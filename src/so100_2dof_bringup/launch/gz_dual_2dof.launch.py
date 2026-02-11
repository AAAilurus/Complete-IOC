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
        # -------- Namespaces and packages --------
        ns1, pkg1, model1 = 'so100', 'so100_2dof_bringup', 'so100_2dof'
        ns2, pkg2, model2 = 'so101', 'so101_2dof_bringup', 'so101_2dof'
        urdf_file = 'so_100_arm_2dof.urdf'

        # -------- Load URDFs --------
        so100_robot_desc, so100_gz_desc, so100_pkg_share = load_urdf(pkg1, urdf_file)
        so101_robot_desc, so101_gz_desc, so101_pkg_share = load_urdf(pkg2, urdf_file)

        # -------- Ensure Gazebo model path --------
        model_path = os.path.join(os.path.dirname(os.path.dirname(so100_pkg_share)), 'models')
        if 'GZ_SIM_RESOURCE_PATH' in os.environ:
            os.environ['GZ_SIM_RESOURCE_PATH'] += f":{model_path}"
        else:
            os.environ['GZ_SIM_RESOURCE_PATH'] = model_path

        # -------- Gazebo --------
        gz = ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'empty.sdf'],
            output='screen',
            additional_env={'GZ_SIM_RESOURCE_PATH': os.environ['GZ_SIM_RESOURCE_PATH']},
        )

        # -------- Bridge --------
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge',
            output='screen',
            parameters=[{'qos_overrides./tf_static.publisher.durability': 'transient_local'}],
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
                '/tf_static@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            ],
        )

        # -------- robot_state_publisher (namespaced) --------
        rsp_so100 = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=ns1,
            output='screen',
            parameters=[{'robot_description': so100_robot_desc}],
        )

        rsp_so101 = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=ns2,
            output='screen',
            parameters=[{'robot_description': so101_robot_desc}],
        )

        # -------- Spawn both robots --------
        spawn_so100 = Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_so100',
            output='screen',
            arguments=[
                '-string', so100_gz_desc.value,
                '-name', model1,
                '-allow_renaming', 'true',
                '-x', '0.0', '-y', '0.0', '-z', '0.0',
            ],
        )

        spawn_so101 = Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_so101',
            output='screen',
            arguments=[
                '-string', so101_gz_desc.value,
                '-name', model2,
                '-allow_renaming', 'true',
                '-x', '0.6', '-y', '0.0', '-z', '0.0',
            ],
        )

        # -------- Load controllers (matches your YAML) --------
        cm_so100 = f'/{ns1}/controller_manager'
        cm_so101 = f'/{ns2}/controller_manager'

        load_jsb_so100 = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', cm_so100,
                'joint_state_broadcaster'
            ],
            output='screen',
        )

        load_pos_so100 = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', cm_so100,
                'arm_position_controller'
            ],
            output='screen',
        )

        load_jsb_so101 = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', cm_so101,
                'joint_state_broadcaster'
            ],
            output='screen',
        )

        load_pos_so101 = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', cm_so101,
                'arm_position_controller'
            ],
            output='screen',
        )

        return [
            gz,
            bridge,
            rsp_so100,
            rsp_so101,
            spawn_so100,
            spawn_so101,

            # SO100 controller chain
            RegisterEventHandler(OnProcessExit(target_action=spawn_so100, on_exit=[load_jsb_so100])),
            RegisterEventHandler(OnProcessExit(target_action=load_jsb_so100, on_exit=[load_pos_so100])),

            # SO101 controller chain
            RegisterEventHandler(OnProcessExit(target_action=spawn_so101, on_exit=[load_jsb_so101])),
            RegisterEventHandler(OnProcessExit(target_action=load_jsb_so101, on_exit=[load_pos_so101])),
        ]

    return LaunchDescription([OpaqueFunction(function=launch_setup)])
