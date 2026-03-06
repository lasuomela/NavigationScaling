from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

# Get the python package top directory and make paths relative to it
# Note: This will only work properly if the package was installed as editable (pip install -e .)
# Other option would be to make paths relative to ros pkg share directory but would mean that the package
# have to be rebuilt after every change in the config files
from pathlib import Path
import earthrovers
pkg_top_dir = Path(earthrovers.__file__).parent

def generate_launch_description():
    # Declare the launch arguments
    declare_model = DeclareLaunchArgument(
        'model',
        default_value='dummy_model',
        description='Model name to use for navigation policy (default: dummy_model)',
    )
    declare_model_config_path = DeclareLaunchArgument(
        'model_config_path',
        default_value=str(pkg_top_dir / 'deployment/src/earthrovers_deployment/config/models.yaml'),
        description='Path to the model configuration file'
    )
    declare_model_weight_dir = DeclareLaunchArgument(
        'model_weight_dir',
        default_value=str(pkg_top_dir / 'deployment/src/earthrovers_deployment/model_weights'),
        description='Directory where the model weights are stored'
    )
    declare_device = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Compute device (default: cuda)'
    )
    declare_main_loop_frequency = DeclareLaunchArgument(
        'main_loop_frequency',
        default_value='4.0',
        description='Frequency at which to run the navigation policy (default: 4 Hz)'
    )

    navigation_node = Node(
        package='earthrovers_deployment',
        executable='navigation_node',
        name='navigation',
        output='screen',
        parameters=[{
            'model': LaunchConfiguration('model'),
            'model_config_path': LaunchConfiguration('model_config_path'),
            'model_weight_dir': LaunchConfiguration('model_weight_dir'),
            'device': LaunchConfiguration('device'),
            'main_loop_frequency': LaunchConfiguration('main_loop_frequency'),
        }]
    )

    return LaunchDescription([
        declare_model,
        declare_model_config_path,
        declare_model_weight_dir,
        declare_device,
        declare_main_loop_frequency,
        navigation_node,
    ])