import wandb
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as TF
from shapely.geometry import Point, LineString

from earthrovers.data_wrangling.pipeline.state_estimation import wgs84_to_enu, enu_to_wgs84

def plot_obs_and_controls(
        ride_id,
        obs_img,
        goal_input,
        gt_controls,
        pred_controls,
        current_position,
        current_yaw,
        goal_position,
        flipped,
        gt_waypoints=None,
        pred_waypoints=None,
    ):
    """Plot observation image, target and predicted controls."""
    goal_input = goal_input.cpu().numpy()
    gt_controls = gt_controls.cpu().numpy()
    pred_controls = pred_controls.detach().cpu().numpy()
    current_position = current_position.cpu().numpy()
    current_yaw = current_yaw.cpu().numpy()
    goal_position = goal_position.cpu().numpy()

    if gt_waypoints is not None:
        gt_waypoints = gt_waypoints.cpu().numpy()
        if gt_waypoints.shape[-1] == 4:
            gt_waypoints = waypoint_cos_sin_to_yaw(gt_waypoints)            

    if pred_waypoints is not None:
        pred_waypoints = pred_waypoints.detach().cpu().numpy()
        if pred_waypoints.shape[-1] == 4:
            pred_waypoints = waypoint_cos_sin_to_yaw(pred_waypoints)

    # If the image and labels have been flipped horizontally,
    # de-flip labels, predicitons and goal input for visualization.
    # Image has already been flipped
    if flipped:
        goal_input[..., 1] *= -1
        gt_controls[..., 1] *= -1
        pred_controls[..., 1] *= -1

        if gt_waypoints is not None:
            # Flip the y-axis and the theta angle of the waypoints
            gt_waypoints[..., 1] *= -1
            gt_waypoints[..., 2] *= -1

        if pred_waypoints is not None:
            # Flip the y-axis and the theta angle of the waypoints
            pred_waypoints[..., 1] *= -1
            pred_waypoints[..., 2] *= -1

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()

    # Set tight layout
    plt.tight_layout()

    # Plot the observation image
    axes[0].imshow(obs_img)
    axes[0].set_title('Observation Image')
    axes[0].axis('off')

    # Show the goal_input (distance, bearing). Plot the bearing as a 'compass' arrow.
    # The dataloader output is normalized, so we need to unnormalize it
    goal_bearing = goal_input[1] * np.pi

    # Place the origin in the center bottom of the image
    compass_origin = (obs_img.shape[1]//2, obs_img.shape[0]-20)
    circle = plt.Circle(compass_origin, 10, fill=False, color='r', lw=2)
    arrow = plt.Arrow(*compass_origin, 10*np.sin(goal_bearing), -10*np.cos(goal_bearing), color='r', lw=2)
    axes[0].add_artist(circle)
    axes[0].add_artist(arrow)
    axes[0].text(compass_origin[0]+15, compass_origin[1], f'Dist: {goal_input[0]:.3f}', color='r')
    axes[0].text(compass_origin[0]+15, compass_origin[1]+10, f'Bearing: {np.rad2deg(goal_bearing):.0f}', color='r')

    # Plot the target linear and angular velocities
    axes[1].plot(gt_controls[:,0], 'r') # linear
    axes[1].plot(gt_controls[:,1], 'r--') # angular

    # Plot the predicted linear and angular velocities
    if len(pred_controls.shape) == 2:
        # If the prediction is just a single point estimate, add an extra dimension
        # to be compatible with e.g. diffusion models that predict multiple trajectories
        pred_controls = np.expand_dims(pred_controls, axis=0)

    # Create a color map for the predicted controls
    cmap = plt.get_cmap('Pastel2', pred_controls.shape[0])

    for i in range(pred_controls.shape[0]):
        axes[1].plot(pred_controls[i,:,0], 
                    color=cmap(i))
        axes[1].plot(pred_controls[i,:,1],
                    color=cmap(i), linestyle='--')

    # Set axis limits to [1, -1]
    axes[1].set_ylim([-1.05, 1.05])
    axes[1].legend(['target_v', 'target_w', 'pred_v', 'pred_w'])
    axes[1].set_title('Control Prediction')
    axes[1].set_xlabel('Time steps')

    # Plot the current position and goal position with Geopandas to get a map background
    plot_map_locations(axes[2], current_position, current_yaw, goal_position, gt_waypoints, pred_waypoints)

    wandb_fig = wandb.Image(fig, caption=ride_id)
    plt.close(fig)
    return wandb_fig

def waypoint_cos_sin_to_yaw(wps_sin_cos):
    """Convert sin/cos representation to yaw angle in radians."""

    # Calculate the yaw angle using arctan2
    yaw = np.arctan2(wps_sin_cos[..., 3], wps_sin_cos[..., 2])
    wps_yaw = np.concatenate(
        [wps_sin_cos[..., :2], yaw[..., np.newaxis]], axis=-1
    )
    return wps_yaw

def local_to_global_coordinates(local_coords, current_position, current_yaw):
    """Convert local coordinates to global coordinates."""
    # Convert local coordinates to global coordinates
    x, y, theta = local_coords
    global_x = current_position[0] + x * np.cos(current_yaw) - y * np.sin(current_yaw)
    global_y = current_position[1] + x * np.sin(current_yaw) + y * np.cos(current_yaw)
    global_theta = current_yaw + theta
    return global_x, global_y, global_theta

def plot_map_locations(
    ax,
    current_position,
    current_yaw,
    goal_position,
    gt_waypoints=None,
    pred_waypoints=None,
    ):
    """
    Plot current and goal positions on a map background.
    """
    if gt_waypoints is not None:
        # The waypoints are (x, y, theta) in robot local coordinates,
        # so we need to convert them to WGS84 coordinates
        current_pose_enu, tf = wgs84_to_enu(
            pd.DataFrame(
            {
                ('gps', 'latitude'): [current_position[0]],
                ('gps', 'longitude'): [current_position[1]],
                ('compass', 'yaw'): [current_yaw]
            }
        ), orientation_header='compass')

        gt_waypoints_enu = local_to_global_coordinates(
            gt_waypoints.T,
            current_pose_enu[[('enu', 'e'), ('enu', 'n')]].values[0],
            current_pose_enu[('enu', 'yaw')].values[0]
        )
        gt_waypoints_wgs84 = enu_to_wgs84(
            pd.DataFrame(
                {
                    ('enu', 'e'): gt_waypoints_enu[0],
                    ('enu', 'n'): gt_waypoints_enu[1],
                    ('enu', 'u'): np.zeros_like(gt_waypoints_enu[0]),
                    ('enu', 'yaw'): gt_waypoints_enu[2]
                }
            ),
            transformer=tf
        )

        wps = gpd.GeoDataFrame(
            {
                'positions': [Point(p[1], p[0]) for p in gt_waypoints_wgs84['wgs84'].values],
                'labels': ['Ground Truth Waypoint'] * len(gt_waypoints_wgs84)
            },
            crs='EPSG:4326',
            geometry='positions'
        )
        wps.plot(ax=ax, color='green', legend=False, markersize=5)

        if pred_waypoints is not None:
            # The predicted waypoints are (x, y, theta) in robot local coordinates,
            # so we need to convert them to WGS84 coordinates
            pred_waypoints_enu = local_to_global_coordinates(
                pred_waypoints.T,
                current_pose_enu[[('enu', 'e'), ('enu', 'n')]].values[0],
                current_pose_enu[('enu', 'yaw')].values[0]
            )
            pred_waypoints_wgs84 = enu_to_wgs84(
                pd.DataFrame(
                    {
                        ('enu', 'e'): pred_waypoints_enu[0],
                        ('enu', 'n'): pred_waypoints_enu[1],
                        ('enu', 'u'): np.zeros_like(pred_waypoints_enu[0]),
                        ('enu', 'yaw'): pred_waypoints_enu[2]
                    }
                ),
                transformer=tf
            )
            wps_pred = gpd.GeoDataFrame(
                {
                    'positions': [Point(p[1], p[0]) for p in pred_waypoints_wgs84['wgs84'].values],
                    'labels': ['Predicted Waypoint'] * len(pred_waypoints_wgs84)
                },
                crs='EPSG:4326',
                geometry='positions'
            )
            wps_pred.plot(ax=ax, color='orange', legend=False, markersize=5)

    d = {
        'positions': [
            Point(current_position[1], current_position[0]),
            Point(goal_position[1], goal_position[0]),
        ],
        'labels': ['Current Position', 'Goal Position']
    }
    gdf = gpd.GeoDataFrame(d, crs='EPSG:4326', geometry='positions')
    gdf.plot(ax=ax, color=['blue', 'red'], legend=False)

    # Set the axis limits to the bounding box of the points
    bounds = gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds
    y_diff = (ymax - ymin)
    x_diff = (xmax - xmin)
    margin = max(max(y_diff, x_diff) * 0.5, 0.0005)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    # Plot the orientation separately without adding it to the legend
    line_length = np.linalg.norm([y_diff, x_diff]) * 0.1
    line = LineString(
        [
            (current_position[1], current_position[0]),
            (
                current_position[1] + line_length*np.sin(current_yaw),
                current_position[0] + line_length*np.cos(current_yaw),
            )
        ]
    )
    ax.plot(*line.xy, color='black')

    # Add a map background
    try:
        zoom = cx.tile._calculate_zoom(*bounds)
    except OverflowError:
        zoom = 19
    zoom = min(19, zoom)
    
    try:
        cx.add_basemap(ax, crs=gdf.crs.to_string(), zoom=zoom, attribution=False)
    except Exception as e:
        # Check for connection errors
        print(f"Failed to fetch the map background. Error: {e}")

    ax.set_title('Current and Goal Positions')
    ax.set_axis_off()

    # Add a legend manually for the points
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Current Position'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Goal Position')
    ]
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.0))

def unnormalize_img(img, img_mean, img_std, flipped=False):
    """Unnormalize image."""
    rgb_img = TF.normalize(img, [-m/s for m, s in zip(img_mean, img_std)], [1/s for s in img_std])

    if flipped:
        rgb_img = TF.hflip(rgb_img)

    rgb_img = rgb_img.permute(0, 2, 3, 1).cpu().numpy()
    return rgb_img