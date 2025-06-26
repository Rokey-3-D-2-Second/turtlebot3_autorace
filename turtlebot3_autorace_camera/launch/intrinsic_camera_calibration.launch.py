#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    composable_nodes = [
        ComposableNode(
            package='image_proc',
            plugin='image_proc::RectifyNode',
            name='rectify_node',
            parameters=[{'queue_size': 20}],
            remappings=[
                ('image', 'camera/image'),
                ('camera_info', 'camera/camera_info')
            ]
        ),
        ComposableNode(
            package='image_proc',
            plugin='image_proc::DebayerNode',
            name='debayer_node',
            remappings=[
                ('image_raw', 'image_rect'),
                ('image_color/compressed', 'image_rect_color/compressed')
            ]
        )
    ]

    republish_node = Node(
        package='image_transport',
        executable='republish',
        name='republish',
        output='screen',
        arguments=['compressed', 'raw'],
        remappings=[
            ('in/compressed', '/camera/image_raw/compressed'),
            ('out', '/camera/image')
        ]
    )

    image_proc_container = ComposableNodeContainer(
        name='image_proc_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
    )

    return LaunchDescription([
        republish_node,
        image_proc_container
    ])
