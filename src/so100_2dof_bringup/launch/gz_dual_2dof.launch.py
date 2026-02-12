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

    # convert meshes path for gz if needed (keep your existing replace rule)
    replace_str = 'package://so_100_arm/models/so_100_arm_5dof/meshes'
    with_str = 'model://so_100_arm_5dof/meshes'
    gazebo_urdf_content = urdf_content.replace(replace_str, with_str)

    return ParameterValue(urdf_content, value_type=str), ParameterValue(gazebo_urdf_content, value_type=str), pkg_share

def generate_launch_description():

    def launch_setup(context, *args, **kwargs):
        # ---------- Load both robot descriptions ----------
        so100_robot_desc, so100_gz_desc, so100_pkg = load_urdf('so100_2dof_bringup', 'so_100_arm_2dof.urdf')
        so101_robot_desc, so101_gz_desc, so101_pkg = load_urdf('so101_2dof_bringup', 'so_100_arm_2dof.urdf')

        # ---------- Ensure GZ model path ----------
        # so100_pkg points to .../share/so100_2dof_bringup
        model_path = os.path.join(os.path.dirname(os.path.dirname(so100_pkg)), 'models')
        if 'GZ_SIM_RESOURCE_PATH' in os.environ:
            os.environ['GZ_SIM_RESOURCE_PATH'] += f":{model_path}"
        else:
            os.environ['GZ_SIM_RESOURCE_PATH'] = model_path

        # ---------- Gazebo ----------
        gz = ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'empty.sdf'],
            output='screen',
            additional_env={'GZ_SIM_RESOURCE_PATH': os.environ['GZ_SIM_RESOURCE_PATH']}
        )

        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge',
            parameters=[{'qos_overrides./tf_static.publisher.durability': 'transient_local'}],
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
                '/tf_static@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            ],
        )

        # ---------- robot_state_publisher (namespaced) ----------
        rsp_so100 = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace='so100',
            output='screen',
            parameters=[{'robot_description': so100_robot_desc}]
        )

        rsp_so101 = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace='so101',
            output='screen',
            parameters=[{'robot_description': so101_robot_desc}]
        )

        # ---------- Spawn both robots ----------
        spawn_so100 = Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_so100',
            arguments=[
                '-string', so100_gz_desc.value,
                '-name', 'so100_2dof',
                '-allow_renaming', 'true',
                '-x', '0.0', '-y', '0.0', '-z', '0.0'
            ],
            output='screen'
        )

        spawn_so101 = Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_so101',
            arguments=[
                '-string', so101_gz_desc.value,
                '-name', 'so101_2dof',
                '-allow_renaming', 'true',
                '-x', '0.6', '-y', '0.0', '-z', '0.0'
            ],
            output='screen'
        )

        # ---------- Controller load commands (target the namespaced controller_manager) ----------
        load_jsb_so100 = ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                 '--controller-manager', '/so100/controller_manager',
                 'joint_state_broadcaster'],
            output='screen'
        )
        load_jtc_so100 = ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                 '--controller-manager', '/so100/controller_manager',
                 'joint_trajectory_controller'],
            output='screen'
        )

        load_jsb_so101 = ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                 '--controller-manager', '/so101/controller_manager',
                 'joint_state_broadcaster'],
            output='screen'
        )
        load_jtc_so101 = ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                 '--controller-manager', '/so101/controller_manager',
                 'joint_trajectory_controller'],
            output='screen'
        )

        # Load controllers after spawn
        return [
            gz,
            bridge,
            rsp_so100, rsp_so101,
            spawn_so100, spawn_so101,

            RegisterEventHandler(
                OnProcessExit(target_action=spawn_so100, on_exit=[load_jsb_so100])
            ),
            RegisterEventHandler(
                OnProcessExit(target_action=load_jsb_so100, on_exit=[load_jtc_so100])
            ),

            RegisterEventHandler(
                OnProcessExit(target_action=spawn_so101, on_exit=[load_jsb_so101])
            ),
            RegisterEventHandler(
                OnProcessExit(target_action=load_jsb_so101, on_exit=[load_jtc_so101])
            ),
        ]

    return LaunchDescription([OpaqueFunction(function=launch_setup)])
