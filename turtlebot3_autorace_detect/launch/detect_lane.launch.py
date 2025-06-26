#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    namespace = 'tb3_0/camera'

    image_proc_container = ComposableNodeContainer(
        name='image_proc_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='rectify_node',
                parameters=[{'queue_size': 20}]
            )
        ],
        output='screen'
    )

    projection_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_camera'),
        'calibration', 'extrinsic_calibration', 'projection.yaml'
    )

    image_projection_node = Node(
        package='turtlebot3_autorace_camera',
        executable='image_projection',
        name='image_projection',
        namespace=namespace,
        output='screen',
        parameters=[projection_param, {'is_extrinsic_camera_calibration_mode': False}],
        remappings=[
            ('/camera/image_input', 'image_rect'),
            ('/camera/image_output', 'image_projected'),
            ('/camera/image_calib', 'image_extrinsic_calib')
        ]
    )

    detect_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_detect'),
        'param', 'lane', 'lane.yaml'
    )

    detect_lane_node = Node(
        package='turtlebot3_autorace_detect',
        executable='detect_lane',
        name='detect_lane',
        namespace=namespace,
        output='screen',
        parameters=[
            {'is_detection_calibration_mode': False},
            detect_param
        ],
        remappings=[
            ('/detect/image_input', 'image_projected'),
            ('/detect/image_output', 'image_lane'),
            ('/detect/image_output_sub1', 'image_white_lane_marker'),
            ('/detect/image_output_sub2', 'image_yellow_lane_marker')
        ]
    )

    return LaunchDescription([
        image_proc_container,
        image_projection_node,
        detect_lane_node
    ])
