#!/usr/bin/env python3

import os
import cv2
import numpy as np

from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Int8
from ament_index_python.packages import get_package_share_directory


class DetectStopSign(Node):
    def __init__(self):
        super().__init__('detect_stop_sign')
        parameter_descriptor_hue = ParameterDescriptor(
            integer_range=[IntegerRange(from_value=0, to_value=179, step=1)],
            description='Hue Value (0~179)'
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)],
            description='Saturation/Lightness Value (0~255)'
        )

        self.declare_parameter(
            'red.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'red.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'red.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.lightness_h', 255, parameter_descriptor_saturation_lightness)
        
        self.declare_parameter('is_calibration_mode', False)

        self.hue_red_l = self.get_parameter(
            'red.hue_l').get_parameter_value().integer_value
        self.hue_red_h = self.get_parameter(
            'red.hue_h').get_parameter_value().integer_value
        self.saturation_red_l = self.get_parameter(
            'red.saturation_l').get_parameter_value().integer_value
        self.saturation_red_h = self.get_parameter(
            'red.saturation_h').get_parameter_value().integer_value
        self.lightness_red_l = self.get_parameter(
            'red.lightness_l').get_parameter_value().integer_value
        self.lightness_red_h = self.get_parameter(
            'red.lightness_h').get_parameter_value().integer_value
        self.is_calibration_mode = self.get_parameter(
            'is_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.get_detect_traffic_light_param)

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        self.counter = 1

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.get_image, 1)
        else:
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.get_image, 1)

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_light = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1)
        else:
            self.pub_image_traffic_light = self.create_publisher(
                Image, '/detect/image_output', 1)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_red_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1)
                self.pub_image_yellow_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1)
                self.pub_image_green_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub3/compressed', 1)
            else:
                self.pub_image_red_light = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1)
                self.pub_image_yellow_light = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1)
                self.pub_image_green_light = self.create_publisher(
                    Image, '/detect/image_output_sub3', 1)

        self.stop_sign_pub = self.create_publisher(Bool, '/stop_sign_detected', 10)
        self.stop_sign_state_pub = self.create_publisher(Int8, '/stop_sign_state', 10)
  
        self.cvBridge = CvBridge()
        self.cv_image = None

        self.is_image_available = False

        self.sift = cv2.SIFT_create()
        self.prepare_stop_image()

        # FLANN 매칭기 설정
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.status = 0
        self.MIN_MATCH_COUNT = 4
        self.MIN_MSE = 50000

        self.red_count = 0
        self.counter = 0
        self.stop_in_progress = False
        self.stop_timer = None
        # 정지 표지판 상태 퍼블리셔 추가
        self.stop_sign_state_publisher = self.create_publisher(
            Int8, '/stop_sign_state', 10)

        self.get_logger().info('🚦 DetectStopSign node initialized.')

    def prepare_stop_image(self):
        try:
            # 소스 디렉토리 기준으로 stop.png 경로 설정
            package_share_dir = get_package_share_directory('turtlebot3_autorace_detect')
            
            stop_path = os.path.join(package_share_dir, 'image', 'stop.png')
            self.img_stop = cv2.imread(stop_path, cv2.IMREAD_GRAYSCALE)
            if self.img_stop is None:
                self.get_logger().error(f'❌ Stop image not found or failed to load: {stop_path}')
                return
            self.kp_stop, self.des_stop = self.sift.detectAndCompute(self.img_stop, None)
            self.get_logger().info(f'📂 Stop image loaded from: {stop_path}')
        except Exception as e:
            self.get_logger().error(f'Exception while loading stop image: {str(e)}')

    def image_callback(self, msg):

        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter = 1

        try:
            if self.image_type == 'compressed':
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image decoding failed: {str(e)}')
            return

        kp1, des1 = self.sift.detectAndCompute(self.cv_image, None)
        if des1 is None or self.des_stop is None:
            return

        matches = self.flann.knnMatch(des1, self.des_stop, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp_stop[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            mse = np.mean((src_pts - dst_pts) ** 2)

            if mse < self.MIN_MSE and not self.stop_in_progress:
                self.get_logger().info("🚫 STOP sign detected! Publishing signal.")
                self.publish_stop_signal(True)
                self.start_stop_timer()

    def publish_stop_signal(self, state: bool):
        bool_msg = Bool()
        bool_msg.data = state
        self.stop_sign_pub.publish(bool_msg)

        int_msg = Int8()
        int_msg.data = 1 if state else 0
        self.stop_sign_state_pub.publish(int_msg)

        self.get_logger().info(f"📡 /stop_sign_detected → {state}, /stop_sign_state → {int_msg.data}")

    def start_stop_timer(self):
        self.stop_in_progress = True
        self.stop_timer = self.create_timer(3.0, self.stop_timer_callback)

    def stop_timer_callback(self):
        self.get_logger().info("✅ 3 seconds passed. Publishing resume signal.")
        self.publish_stop_signal(False)
        self.stop_in_progress = False

        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None


def main(args=None):
    rclpy.init(args=args)
    node = DetectStopSign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()