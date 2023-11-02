import os
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription


def generate_launch_description():

    # urdf_file = os.path.join(get_package_share_directory('arrow_description'), 'urdf', 'arrow.urdf')

    # with open(urdf_file, 'r') as infp:
    #     robot_description_config = infp.read()
    
    # robot_description = {'robot_description': robot_description_config}

    rviz_config_file = os.path.join(get_package_share_directory('manipulator_description'),
                             'rviz', 'config.rviz')
    
    my_node = Node(
        package='motion_programm',
        executable='test_node',
        name='test_node'
    )
    
    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     output='screen',
    #     remappings=[("/robot_description", "/arrow/robot_description")],
    #     parameters=[robot_description],
    # )

    # joint_state_publisher_gui = Node(
    #     package='joint_state_publisher_gui',
    #     executable='joint_state_publisher_gui',
    #     output='screen',
    #     remappings=[("/robot_description", "/arrow/robot_description")],
    #     parameters=[robot_description])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen')


    return LaunchDescription(
        # [robot_state_publisher,
        # joint_state_publisher_gui,
        # rviz,
        # launch_red_model,
        # tf_pose,
        # ]
        [   
            # my_node,
            rviz
        ]
    )