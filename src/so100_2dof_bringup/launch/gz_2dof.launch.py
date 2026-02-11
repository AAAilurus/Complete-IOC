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

    # ROS uses package://, Gazebo wants model://
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
        # -------- Settings (edit these if needed) --------
        ns = 'so100'  # single robot namespace
        pkg_name = 'so100_2dof_bringup'
        urdf_file = 'so_100_arm_2dof.urdf'
        model_name = 'so100_2dof'
        x, y, z = '0.0', '0.0', '0.0'

        # -------- Load URDF --------
        robot_desc, gz_desc, pkg_share = load_urdf(pkg_name, urdf_file)

        # -------- Ensure Gazebo model path (meshes) --------
        # pkg_share = .../install/<pkg>/share/<pkg>
        # models live under .../install/<something>/share/so_100_arm/models
        # But your earlier approach also worked by adding the install "models" folder.
        model_path = os.path.join(os.path.dirname(os.path.dirname(pkg_share)), 'models')
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

        # -------- Bridge (clock/tf) --------
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
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=ns,
            output='screen',
            parameters=[{'robot_description': robot_desc}],
        )

        # -------- Spawn robot --------
        spawn = Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_model',
            output='screen',
            arguments=[
                '-string', gz_desc.value,
                '-name', model_name,
                '-allow_renaming', 'true',
                '-x', x, '-y', y, '-z', z,
            ],
        )

        # -------- Load controllers (IMPORTANT: matches your YAML) --------
        # Controller manager is namespaced by gz_ros2_control <namespace> tag in URDF.
        cm = f'/{ns}/controller_manager'

        load_jsb = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', cm,
                'joint_state_broadcaster'
            ],
            output='screen',
        )

        load_pos = ExecuteProcess(
            cmd=[
                'ros2', 'control', 'load_controller', '--set-state', 'active',
                '--controller-manager', cm,
                'arm_position_controller'
            ],
            output='screen',
        )

        return [
            gz,
            bridge,
            rsp,
            spawn,

            # Spawn controllers only after robot exists
            RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[load_jsb])),
            RegisterEventHandler(OnProcessExit(target_action=load_jsb, on_exit=[load_pos])),
        ]

    return LaunchDescription([OpaqueFunction(function=launch_setup)])
