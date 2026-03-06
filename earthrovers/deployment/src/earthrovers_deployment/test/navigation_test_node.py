import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import numpy as np

from std_msgs.msg import Float32
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import Twist
from geographic_msgs.msg import GeoPoseStamped, GeoPath

class NavigationTestNode(Node):
    def __init__(self):
        super().__init__('navigation_test_node')

        self._cv_bridge = CvBridge()
        self.dummy_image = self._cv_bridge.cv2_to_imgmsg(
            np.zeros((480, 640, 3), np.uint8),
            encoding="rgb8")

        self._front_camera_publisher = self.create_publisher(
            Image, '/front_camera/image_raw', 10)

        self._gps_publisher = self.create_publisher(
            NavSatFix, '/gps', 10)
        
        self._orientation_publisher = self.create_publisher(
            Float32, '/orientation', 10)
        self._filtered_orientation_publisher = self.create_publisher(
            Float32, '/orientation_filtered', 10)
        
        self._dummy_path = GeoPath()
        self._dummy_path.poses.append(GeoPoseStamped())
        self._dummy_path.poses[0].pose.position.latitude = 10.
        self._dummy_path.poses[0].pose.position.longitude = 10.
        self._goal_publisher = self.create_publisher(
            GeoPath, '/checkpoints_gps', 10)

        self._main_timer = self.create_timer(1.0, self._main_timer_callback)

    def _main_timer_callback(self):
        self.get_logger().info('Publishing test data...')
        self._front_camera_publisher.publish(self.dummy_image)
        self._gps_publisher.publish(NavSatFix())
        self._orientation_publisher.publish(Float32())
        self._filtered_orientation_publisher.publish(Float32())
        self._goal_publisher.publish(self._dummy_path)

def main(args=None):
    rclpy.init(args=args)
    navigation_test_node = NavigationTestNode()
    try:
        rclpy.spin(navigation_test_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_test_node.destroy_node()
        
if __name__ == '__main__':
    main()

