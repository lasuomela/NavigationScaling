from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from cv_bridge import CvBridge
from rclpy.duration import Duration

from std_msgs.msg import Float32
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import Twist
from geographic_msgs.msg import GeoPath

import yaml
import numpy as np
import torch
from pathlib import Path
from threading import Lock
import matplotlib.pyplot as plt
import time

import earthrovers
import earthrovers_deployment.policies
from earthrovers_deployment.policies.base_policy import dynamic_load
from earthrovers_deployment.keyboard_killswitch import KeyboardKillswitchController
from earthrovers.common.utils import compute_distance, compute_direction
pkg_top_dir = Path(earthrovers.__file__).parent

class NavigationNode(Node):
    """
    ROS2 Node for robot navigation using a learned policy.
    1. Subscribes to camera images, GPS data, orientation data, and goal coordinates.
    2. Uses a learned navigation policy to compute velocity commands.
    3. Publishes velocity commands to control the robot.

    Use keyboard key 'k' to toggle policy execution (stop/start the robot), and 'q' to quit the program.
    """
    def __init__(self):
        super().__init__('navigation_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('model', 'EarthRovers_DP-Unet_nLoc32_nH32'), #'EarthRovers_MLP-BC_nLoc153_nH-1'),
                ('model_config_path', str(pkg_top_dir / 'deployment/src/earthrovers_deployment/config/huggingface_config.yaml')),
                ('model_weight_dir', str(pkg_top_dir / 'deployment/src/earthrovers_deployment/model_weights')),
                ('device', 'cuda'),
                ('main_loop_frequency', 4.0),
                ('dry_run', False),
                ('goal_orient', False),
                ('start_orient', False),
                ('max_speed', 0.7),
                ('filtered_orientation', True),
            ],
        )
        self._setup_ros()
        self._setup_navigation_policy()

        self.get_logger().info('Navigation node initialized.')

        self._goal_orient = self.get_parameter('goal_orient').get_parameter_value().bool_value
        self._start_orient = self.get_parameter('start_orient').get_parameter_value().bool_value
        self._max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self._reached_last_goal = False
        self._approaching_last_goal = False

        self._state = 'RUN'
        self._state_change_time = self.get_clock().now()
        
        # Parameters for 'manually' orienting the robot towards the goal
        self._orient_start_threshold = 0.5
        self._orient_threshold = 0.2 # can be changed
        self._overtime_orient_threshold = 0.15
        self._orient_minumum_error_yaw = None
        self._orient_start_time = None
        self._orient_max_time = Duration(seconds=25.0)


        # Track how long it takes to reach each goal segment
        self._segment_start_time = None
        self._segment_durations = []
        self._previous_goal_lat_lon = None
        self._segment_toggle_start_time = None
        self._segment_toggle_duration = 0.0

        # Make policy stoppable via keyboard killswitch. Press 'k' to toggle, 'q' to quit.
        self._killswitch_controller = KeyboardKillswitchController(
            logger=self.get_logger(),
            model_name_provider=self._get_model_name,
            segment_durations_provider=lambda: self._segment_durations,
            on_killswitch_toggled=self._on_killswitch_toggled,
            on_shutdown=rclpy.shutdown,
        )
        self._killswitch_controller.start()

    def __del__(self):
        if hasattr(self, '_killswitch_controller'):
            self._killswitch_controller.stop()

    def shutdown(self):
        if hasattr(self, '_killswitch_controller'):
            self._killswitch_controller.stop()

    def _on_killswitch_toggled(self, is_active: bool):
        self._state_change_time = self.get_clock().now()
        if is_active:
            self._segment_toggle_start_time = time.time()
            return

        if self._segment_toggle_start_time is not None:
            self._segment_toggle_duration += time.time() - self._segment_toggle_start_time
            self._segment_toggle_start_time = None
     
    def _get_model_name(self):
        return self.get_parameter('model').get_parameter_value().string_value
    
    def _setup_navigation_policy(self):
        """
        Set up navigation policy.
        """
        self.get_logger().info('Setting up navigation policy...')

        # Load the model config
        model_config_path = Path(self.get_parameter('model_config_path').get_parameter_value().string_value)
        with model_config_path.open(mode="r", encoding="utf-8") as f:
            model_configs = yaml.safe_load(f)
        self.model_config = model_configs[self.get_parameter('model').get_parameter_value().string_value]
        
        # Parse the model weight ckpt
        if 'ckpt_path' in self.model_config:
            self.model_config['ckpt_path'] = Path(
                self.get_parameter('model_weight_dir').get_parameter_value().string_value) / self.model_config['ckpt_path']

        policy_class = dynamic_load(
            earthrovers_deployment.policies,
            self.model_config['model_type'],
        )
        self._navigation_policy = policy_class(
            config=self.model_config,
            device=self.get_parameter('device').get_parameter_value().string_value,
        )

    def _setup_ros(self):
        """
        Set up ROS publishers, subscribers and timer loops.
        """
        self.get_logger().info('Setting up ROS...')
        self._cv_bridge = CvBridge()

        # Subscribers
        self._front_camera_lock = Lock()
        self._front_camera_msg = None
        self._front_camera_sub = self.create_subscription(
            Image,
            '/front_camera/image_raw',
            self._front_camera_callback,
            10,
        )

        self._gps_lock = Lock()
        self._gps_msg = None
        self._gps_sub = self.create_subscription(
            NavSatFix,
            '/gps',
            self._gps_callback,
            10
        )

        orientation_topic = '/orientation_filtered' if self.get_parameter('filtered_orientation').get_parameter_value().bool_value else '/orientation'
        self._orientation_lock = Lock()
        self._orientation_msg = None
        self._orientation_sub = self.create_subscription(
            Float32,
            orientation_topic,
            self._orientation_callback,
            10
        )

        self._goal_lock = Lock()
        self._goal_msg = None
        self._goal_sub = self.create_subscription(
            GeoPath,
            '/checkpoints_gps',
            self._goal_callback,
            10
        )

        # Publishers
        cmd_topic = '/cmd_vel'
        dry_run = self.get_parameter('dry_run').get_parameter_value().bool_value
        if dry_run:
            cmd_topic = '/cmd_vel_dry_run'
        self._cmd_vel_pub = self.create_publisher(
            Twist,
            cmd_topic,
            10
        )

        self._goal_pub = self.create_publisher(
            NavSatFix,
            '/goal',
            10
        )

        # Publisher for the figure that visualizes the commands
        self._cmd_viz_pub = self.create_publisher(
            Image,
            '/cmd_viz',
            10
        )

        # Timer loops
        self.main_loop_frequency = self.get_parameter('main_loop_frequency').get_parameter_value().double_value
        self.main_timer = self.create_timer(
            1/self.main_loop_frequency,
            self.main_loop,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    ### Callbacks
    def _front_camera_callback(self, msg):
        """
        Callback for front camera image.
        """
        with self._front_camera_lock:
            self._front_camera_msg = msg

    def _gps_callback(self, msg):
        """
        Callback for GPS data.
        """
        with self._gps_lock:
            self._gps_msg = msg

    def _orientation_callback(self, msg):
        """
        Callback for orientation data.
        """
        with self._orientation_lock:
            self._orientation_msg = msg

    def _goal_callback(self, msg):
        """
        Callback for goal data.
        """
        with self._goal_lock:
            first_callback = self._goal_msg is None
            goal_lat_lon = (msg.poses[0].pose.position.latitude, msg.poses[0].pose.position.longitude)

            if (first_callback and self._start_orient):
                self._state = 'ORIENT'
                self._state_change_time = self.get_clock().now()
                self._orient_start_time = self.get_clock().now()
                
            elif ( (not first_callback) and self._goal_orient and
                ((msg.poses and self._goal_msg.poses) and
                msg.poses[0].pose.position != self._goal_msg.poses[0].pose.position)
                ):

                # Check if we are already facing the goal
                current_lat_lon = (self._gps_msg.latitude, self._gps_msg.longitude)
                current_orientation = np.deg2rad(self._orientation_msg.data)
                _, goal_direction = self._compute_goal_input(
                    current_position=current_lat_lon,
                    current_orientation=current_orientation,
                    goal_position=goal_lat_lon,
                )
            
                if abs(goal_direction) > self._orient_start_threshold:
                    self._state = 'ORIENT'
                    self._state_change_time = self.get_clock().now()
                    self._orient_start_time = self.get_clock().now()

            self._goal_msg = msg

            if first_callback:
                self._segment_start_time = time.time()
            elif self._reached_last_goal:
                if self._segment_start_time is not None:
                    if self._segment_toggle_start_time is not None:
                        self._segment_toggle_duration = time.time() - self._segment_toggle_start_time

                    segment_duration = time.time() - self._segment_start_time - self._segment_toggle_duration
                    self._segment_durations.append(segment_duration)
                    self.get_logger().info(f"Reached goal. Segment active duration: {segment_duration:.2f} s. Toggle duration: {self._segment_toggle_duration:.2f} s.")

                    self._segment_start_time = None
                    self._segment_toggle_duration = 0.0
                    self._segment_toggle_start_time = None

            elif goal_lat_lon != self._previous_goal_lat_lon:

                if self._segment_toggle_start_time is not None:
                    self._segment_toggle_duration = time.time() - self._segment_toggle_start_time

                segment_duration = time.time() - self._segment_start_time - self._segment_toggle_duration
                self._segment_durations.append(segment_duration)
                self.get_logger().info(f"Reached checkpoint. Segment active duration: {segment_duration:.2f} s. Toggle duration: {self._segment_toggle_duration:.2f} s.")
                self._segment_start_time = time.time()
                self._segment_toggle_duration = 0.0
                self._segment_toggle_start_time = None
            self._previous_goal_lat_lon = goal_lat_lon

    def _fetch_data(self):
        """
        Fetch the latest data from the subscribers.
        """
        with self._front_camera_lock:
            if self._front_camera_msg is None:
                return None
            front_camera_msg = self._front_camera_msg
            # Only process each image once
            self._front_camera_msg = None
        with self._gps_lock:
            if self._gps_msg is None:
                return None
            gps_msg = self._gps_msg
        with self._orientation_lock:
            if self._orientation_msg is None:
                return None
            orientation_msg = self._orientation_msg
        with self._goal_lock:
            if self._goal_msg is None:
                return None
            goal_msg = self._goal_msg

        front_camera_img = self._cv_bridge.imgmsg_to_cv2(front_camera_msg, desired_encoding='rgb8')
        lat_lon = (gps_msg.latitude, gps_msg.longitude)

        if goal_msg.poses:
            if len(goal_msg.poses) == 1:
                self._approaching_last_goal = True

            goal = goal_msg.poses[0].pose.position
            goal_lat_lon = (goal.latitude, goal.longitude)

            # Publish the goal for vizualization
            self._goal_pub.publish(
                NavSatFix(
                    latitude=goal.latitude,
                    longitude=goal.longitude,
                )
            )
        else:
            # No goal coodinates in the message
            return None

        orientation = np.deg2rad(orientation_msg.data) 
        if orientation > np.pi:
            orientation -= 2*np.pi
        assert orientation >= -np.pi and orientation <= np.pi

        goal_distance, goal_direction = self._compute_goal_input(
            current_position=lat_lon,
            current_orientation=orientation,
            goal_position=goal_lat_lon,
        )

        data = {
            'image': front_camera_img,
            'lat_lon': lat_lon,
            'orientation': orientation,
            'goal_lat_lon': goal_lat_lon,
            'goal_distance': goal_distance,
            'goal_direction': goal_direction,
        }
        return data

    def _compute_goal_input(
            self,
            current_position: List[float],
            current_orientation: List[float],
            goal_position: List[float],
        ):
        """
        Compute the distance and bearing to the goal position.

        Args:
            current_position: Current position of the robot. (Latitude, Longitude) in degrees, WGS84 / EPSG:4326.
            current_orientation: Current orientation of the robot. (Heading) in [-pi, +pi] radians, clockwise positive from North.
            goal_position: Goal position of the robot. (Latitude, Longitude) in degrees, WGS84 / EPSG:4326.
        """
        lat1, lon1 = torch.tensor(current_position)
        lat2, lon2 = torch.tensor(goal_position)
        current_orientation = torch.tensor(current_orientation)

        distance = compute_distance(lat1, lon1, lat2, lon2)
        direction = compute_direction(lat1, lon1, lat2, lon2, current_orientation).squeeze()
        self.get_logger().info(f"Distance: {distance:<.3f}, Direction: {torch.rad2deg(direction*torch.pi):<.1f}")
        return distance, direction

    def _goal_orient_cmd(self, data: Dict):
        """
        Orients the robot towards the goal direction using a simple proportional controller.
        """
        direction = data['goal_direction'].item()
        yaw = data['orientation'].item()

        if (self._orient_minumum_error_yaw is None) or (abs(direction) < self._orient_minumum_error_yaw[1]):
            self._orient_minumum_error_yaw = (yaw, abs(direction))

        multiplier = 0.6
        orient_threshold = self._orient_threshold
        if self.get_clock().now() - self._orient_start_time > self._orient_max_time:
            self.get_logger().info('Orientation timeout, orienting to minimum distance.')
            direction = (self._orient_minumum_error_yaw[0] - yaw) / np.pi
            orient_threshold = self._overtime_orient_threshold
            multiplier = 0.4

        raw_ang = -multiplier * direction
        max_ang = 1.0
        angular = max(-max_ang, min(max_ang, raw_ang))

        if abs(direction) < orient_threshold:
            self._state = 'RUN'
            self._state_change_time = self.get_clock().now()
            self._orient_start_time = None
            self._orient_minumum_error_yaw = None
            angular = 0.0

        # 
        return np.array([0.0, angular], dtype=np.float32)

    def _publish_angular_velocity_viz(self, batch):
        """
        Publish a visualization of angular velocities for a batch of predicted commands.
        """
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        for cmd in batch:
            ax.plot(cmd[:, 1], label='Angular Velocity')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_ylim(-1.0, 1.0)

        fig.canvas.draw()

        img_msg = self._cv_bridge.cv2_to_imgmsg(
            np.array(fig.canvas.buffer_rgba()), encoding='rgba8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'cmd_viz'
        self._cmd_viz_pub.publish(img_msg)
        plt.close(fig)

    ### Main loop
    def main_loop(self):
        """
        Main loop for navigation.
        """
        # Get data
        data = self._fetch_data()
        if data is None:
            return
        
        # Process data
        batch = None
        if self._state == 'ORIENT':
            cmd_vel = self._goal_orient_cmd(data)
        else:
            action_chunk = self._navigation_policy(data)
            if len(action_chunk.shape) == 4:
                # If the policy returns a batch of commands, take the first one
                batch = action_chunk[0]
                action_chunk = batch[0]

            # Take the first command from the action chunk
            cmd_vel = action_chunk[0]

        if self._reached_last_goal or (self._approaching_last_goal and data['goal_distance'] < 0.01):
            self.get_logger().info('Reached last goal, stopping.')
            self._reached_last_goal = True
            cmd_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Check killswitch
        if self._killswitch_controller.killswitch_active:
            if not self._killswitch_controller.consume_state_change():
                return
            cmd_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Publish cmd_vel
        cmd_vel_msg = Twist()
        linear = cmd_vel[0].item()
        angular = cmd_vel[1].item()

        print(f"v: {linear:.2f}, w: {angular:.2f}")

        linear = np.clip(linear, -self._max_speed, self._max_speed).item()
        cmd_vel_msg.linear.x = linear if ((linear > 0.08) or (abs(angular) < 0.1)) else 0.
        cmd_vel_msg.angular.z = angular
        self._cmd_vel_pub.publish(cmd_vel_msg)

        # If running a diffusion policy,
        # publish the different angular velocities for visualization
        if batch is not None:
            self._publish_angular_velocity_viz(batch)



def main(args=None):
    rclpy.init(args=args)

    navigation_node = NavigationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(navigation_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        navigation_node.shutdown()
        navigation_node.destroy_node()

if __name__ == '__main__':
    main()
