#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    calibration_mode_arg = DeclareLaunchArgument(
        'calibration_mode',
        default_value='False',
        description='calibration mode type [True, False]')
    calibration_mode = LaunchConfiguration('calibration_mode')

    compensation_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_camera'),
        'calibration',
        'extrinsic_calibration',
        'compensation.yaml'
    )

    projection_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_camera'),
        'calibration',
        'extrinsic_calibration',
        'projection.yaml'
    )

    # 네임스페이스/리매핑 없이 단순 토픽만 사용
    remappings_projection = [
        ('/camera/image_input', '/image_rect_color'),
        ('/camera/image_input/compressed', '/image_rect_color/compressed'),
        ('/camera/image_output', '/image_projected'),
        ('/camera/image_output/compressed', '/image_projected/compressed'),
        ('/camera/image_calib', '/image_extrinsic_calib'),
        ('/camera/image_calib/compressed', '/image_extrinsic_calib/compressed')
    ]
    remappings_compensation = [
        ('/camera/image_input', '/image_rect_color'),
        ('/camera/image_input/compressed', '/image_rect_color/compressed'),
        ('/camera/image_output', '/image_compensated'),
        ('/camera/image_output/compressed', '/image_compensated/compressed')
    ]

    image_projection_node = Node(
        package='turtlebot3_autorace_camera',
        executable='image_projection',
        name='image_projection',
        output='screen',
        parameters=[
            projection_param,
            {'is_extrinsic_camera_calibration_mode': calibration_mode}
        ],
        remappings=remappings_projection
    )

    image_compensation_node = Node(
        package='turtlebot3_autorace_camera',
        executable='image_compensation',
        name='image_compensation',
        namespace='',
        output='screen',
        parameters=[{
            'is_extrinsic_camera_calibration_mode': calibration_mode
        },
        compensation_param],
        remappings=remappings_compensation
    )

    return LaunchDescription([
        calibration_mode_arg,
        image_projection_node,
        image_compensation_node
    ])
