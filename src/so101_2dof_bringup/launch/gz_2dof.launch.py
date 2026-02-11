from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import os


def load_urdf(pkg_name: str, urdf_rel: str):
    pkg_share = FindPackageShare(pkg_name).find(pkg_name)
    urdf_path = os.path.join(pkg_share, 'urdf', urdf_rel)
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    # Convert package:// meshes to model:// meshes for Gazebo
    replace_str = 'package://so_100_arm/models/so_100_arm_5dof/meshes'
    with_str = 'model://so_100_arm_5dof/meshes'
    gazebo_urdf_content = urdf_content.replace(replace_str, with_str)

    return (
        ParameterValue(urdf_content, value_type=str),
        ParameterValue(gazebo_urdf_content, value_type=str),
        pkg_share
    )


def generate_launch_description():
    def launch_setup(context, *args, **kwargs):
        # --- Load SO101 description ---
        robot_desc, gz_desc, pkg_share = load_urdf(
            'so101_2dof_bringup',
            'so_100_arm_2dof.urdf'
        )

        # --- Ensure Gazebo can find meshes/models ---
        model_path = os.path.join(os.path.dirname(os.path.dirname(pkg_share)), 'models')
        gz_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
        if model_path not in gz_path.split(':'):
            os.environ['GZ_SIM_RESOURCE_PATH'] = (gz_path + ':' + model_path) if gz_path else model_path

        # --- Gazebo ---
        gz = ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'empty.sdf'],
            output='screen',
            additional_env={'GZ_SIM_RESOURCE_PATH': os.environ['GZ_SIM_RESOURCE_PATH']}
        )

        # --- Bridge (clock + tf) ---
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge',
            output='screen',
            parameters=[{
                'qos_overrides./tf_static.publisher.durability': 'transient_local',
            }],
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
                '/tf_static@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            ],
        )

        # --- robot_state_publisher (namespaced SO101) ---
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace='so101',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        )

        # --- Spawn SO101 robot in Gazebo ---
        spawn = Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_so101',
            output='screen',
            arguments=[
                '-string', gz_desc.value,
                '-name', 'so101_2dof',
                '-allow_renaming', 'true',
                '-x', '0.0', '-y', '0.0', '-z', '0.0'
            ],
        )

        # --- Load controllers into /so101/controller_manager ---
        load_jsb = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', '/so101/controller_manager',
                'joint_state_broadcaster'
            ],
            output='screen'
        )

        load_pos = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', '/so101/controller_manager',
                'arm_position_controller'
            ],
            output='screen'
        )

        # If you really use joint_trajectory_controller instead, swap the name above.

        return [
            gz,
            bridge,
            rsp,
            spawn,

            # After spawn -> load JSB
            RegisterEventHandler(
                OnProcessExit(target_action=spawn, on_exit=[load_jsb])
            ),

            # After JSB -> load position controller
            RegisterEventHandler(
                OnProcessExit(target_action=load_jsb, on_exit=[load_pos])
            ),
        ]

    return LaunchDescription([OpaqueFunction(function=launch_setup)])
