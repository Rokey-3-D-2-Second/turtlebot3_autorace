#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Leon Jung, Gilbert, Ashe Kim, ChanHyeong Lee

import collections
import time

import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from std_msgs.msg import Bool
# from geometry_msgs.msg import Twist  # Twist 메시지 제거
from std_msgs.msg import Int8  # Int8 메시지 추가


class DetectTrafficLight(Node):

    def __init__(self):
        super().__init__('detect_traffic_light')
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

        self.declare_parameter(
            'yellow.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_h', 255, parameter_descriptor_saturation_lightness)

        self.declare_parameter(
            'green.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'green.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'green.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.lightness_h', 255, parameter_descriptor_saturation_lightness)

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

        self.hue_yellow_l = self.get_parameter(
            'yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'yellow.lightness_h').get_parameter_value().integer_value

        self.hue_green_l = self.get_parameter(
            'green.hue_l').get_parameter_value().integer_value
        self.hue_green_h = self.get_parameter(
            'green.hue_h').get_parameter_value().integer_value
        self.saturation_green_l = self.get_parameter(
            'green.saturation_l').get_parameter_value().integer_value
        self.saturation_green_h = self.get_parameter(
            'green.saturation_h').get_parameter_value().integer_value
        self.lightness_green_l = self.get_parameter(
            'green.lightness_l').get_parameter_value().integer_value
        self.lightness_green_h = self.get_parameter(
            'green.lightness_h').get_parameter_value().integer_value

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

        self.cvBridge = CvBridge()
        self.cv_image = None

        self.is_image_available = False
        self.is_traffic_light_finished = False

        self.status = 0
        self.green_count = 0
        self.yellow_count = 0
        self.red_count = 0
        self.stop_count = 0
        self.off_traffic = False

        # 1. 표지판(차단봉/교차로 등) 최근 감지시각 저장용 deque
        self.traffic_sign_detect_times = collections.deque(maxlen=10)
        # self.level_sign_detect_times = collections.deque(maxlen=10)
        self.level_bar_detect_times = collections.deque(maxlen=10)
        self.check_window_sec = 1.5        # 1.5초 윈도우 내에서 N회 감지되면 신호 무시
        self.check_min_count = 2           # 표지판 감지 N회 이상이면 무시

        # 2. 표지판 토픽 구독 (이미지 → UInt8로 변경)
        self.create_subscription(
            UInt8, '/detect/traffic_sign', self.cb_traffic_sign, 1)
        # self.create_subscription(
        #     CompressedImage, '/detect/image_level/compressed', self.cb_level_sign, 1)
        self.create_subscription(Bool, '/detect/level_bar', self.cb_level_bar, 1)
        # 차량 속도 제어 퍼블리셔 제거
        # self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10) # 제거
        # self.current_speed = 0.0 # 제거

        # 신호등 상태 퍼블리셔 추가
        self.traffic_light_state_publisher = self.create_publisher(
            Int8, '/traffic_light_state', 10)


        time.sleep(1)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def get_detect_traffic_light_param(self, params):
        for param in params:
            if param.name == 'red.hue_l':
                self.hue_red_l = param.value
                self.get_logger().info(f'red.hue_l set to: {param.value}')
            elif param.name == 'red.hue_h':
                self.hue_red_h = param.value
                self.get_logger().info(f'red.hue_h set to: {param.value}')
            elif param.name == 'red.saturation_l':
                self.saturation_red_l = param.value
                self.get_logger().info(f'red.saturation_l set to: {param.value}')
            elif param.name == 'red.saturation_h':
                self.saturation_red_h = param.value
                self.get_logger().info(f'red.saturation_h set to: {param.value}')
            elif param.name == 'red.lightness_l':
                self.lightness_red_l = param.value
                self.get_logger().info(f'red.lightness_l set to: {param.value}')
            elif param.name == 'red.lightness_h':
                self.lightness_red_h = param.value
                self.get_logger().info(f'red.lightness_h set to: {param.value}')
            elif param.name == 'yellow.hue_l':
                self.hue_yellow_l = param.value
                self.get_logger().info(f'yellow.hue_l set to: {param.value}')
            elif param.name == 'yellow.hue_h':
                self.hue_yellow_h = param.value
                self.get_logger().info(f'yellow.hue_h set to: {param.value}')
            elif param.name == 'yellow.saturation_l':
                self.saturation_yellow_l = param.value
                self.get_logger().info(f'yellow.saturation_l set to: {param.value}')
            elif param.name == 'yellow.saturation_h':
                self.saturation_yellow_h = param.value
                self.get_logger().info(f'yellow.saturation_h set to: {param.value}')
            elif param.name == 'yellow.lightness_l':
                self.lightness_yellow_l = param.value
                self.get_logger().info(f'yellow.lightness_l set to: {param.value}')
            elif param.name == 'yellow.lightness_h':
                self.lightness_yellow_h = param.value
                self.get_logger().info(f'yellow.lightness_h set to: {param.value}')
            elif param.name == 'green.hue_l':
                self.hue_green_l = param.value
                self.get_logger().info(f'green.hue_l set to: {param.value}')
            elif param.name == 'green.hue_h':
                self.hue_green_h = param.value
                self.get_logger().info(f'green.hue_h set to: {param.value}')
            elif param.name == 'green.saturation_l':
                self.saturation_green_l = param.value
                self.get_logger().info(f'green.saturation_l set to: {param.value}')
            elif param.name == 'green.saturation_h':
                self.saturation_green_h = param.value
                self.get_logger().info(f'green.saturation_h set to: {param.value}')
            elif param.name == 'green.lightness_l':
                self.lightness_green_l = param.value
                self.get_logger().info(f'green.lightness_l set to: {param.value}')
            elif param.name == 'green.lightness_h':
                self.lightness_green_h = param.value
                self.get_logger().info(f'green.lightness_h set to: {param.value}')
        return SetParametersResult(successful=True)

    def get_image(self, image_msg):
        # Processing every 3 frames to reduce frame processing load
        if self.counter % 1 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            try:
                self.cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')
                return

        self.is_image_available = True

    def timer_callback(self):
        if self.is_image_available:
            self.find_traffic_light()

    # 표지판 메시지 올 때마다 현재 시각 기록 (UInt8만 기록)
    def cb_traffic_sign(self, msg):
        self.traffic_sign_detect_times.append(time.time())

    # 표지판 메시지 올 때마다 현재 시각 기록 (CompressedImage만 기록)
    # def cb_level_sign(self, msg):
    #     self.level_sign_detect_times.append(time.time())
        
    def cb_level_bar(self, msg):
        self.level_bar_detect_times.append(time.time())

    # sliding window 내 감지 횟수 반환
    def get_recent_detect_count(self, detect_times):
        now = time.time()
        return len([t for t in detect_times if now-t < self.check_window_sec])

    # 기존 find_traffic_light()에서 detect_red 등 검사하는 부분에 조건 추가!
    def find_traffic_light(self):
        cv_image_mask_red = self.mask_red_traffic_light()
        cv_image_mask_red = cv2.GaussianBlur(cv_image_mask_red, (5, 5), 0)
        detect_red = self.find_circle_of_traffic_light(cv_image_mask_red, 'red')
        # if detect_red:
        #     cv2.putText(self.cv_image, 'RED', (self.point_x, self.point_y),
        #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        
        red_light_detected = False # 빨간불 감지 여부 플래그
        if detect_red:
            cv2.putText(self.cv_image, 'RED', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            red_light_detected = True # 빨간불 감지됨

        cv_image_mask_yellow = self.mask_yellow_traffic_light()
        cv_image_mask_yellow = cv2.GaussianBlur(cv_image_mask_yellow, (5, 5), 0)
        detect_yellow = self.find_circle_of_traffic_light(cv_image_mask_yellow, 'yellow')

        cv_image_mask_green = self.mask_green_traffic_light()
        cv_image_mask_green = cv2.GaussianBlur(cv_image_mask_green, (5, 5), 0)
        detect_green = self.find_circle_of_traffic_light(cv_image_mask_green, 'green')

        # *** 표지판 감지 검사 조건 추가 ***
        sign_cnt = self.get_recent_detect_count(self.traffic_sign_detect_times)
        # level_cnt = self.get_recent_detect_count(self.level_sign_detect_times)
        level_cnt = self.get_recent_detect_count(self.level_bar_detect_times)
        ignore_sign = (level_cnt >= self.check_min_count) or (sign_cnt >= self.check_min_count)

        if detect_red and not ignore_sign:
            cv2.putText(self.cv_image, 'RED', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            self.get_logger().info("Red light: 신호등으로 최종 인정")
        elif detect_red and ignore_sign:
            self.get_logger().info("Red light: 표지판/차단봉 감지로 신호 무시")

        if detect_yellow and not ignore_sign:
            cv2.putText(self.cv_image, 'YELLOW', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
            self.get_logger().info("Yellow light: 신호등으로 최종 인정")
        elif detect_yellow and ignore_sign:
            self.get_logger().info("Yellow light: 표지판/차단봉 감지로 신호 무시")

        if detect_green and not ignore_sign:
            cv2.putText(self.cv_image, 'GREEN', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
            self.get_logger().info("Green light: 신호등으로 최종 인정")
        elif detect_green and ignore_sign:
            self.get_logger().info("Green light: 표지판/차단봉 감지로 신호 무시")

        # 신호등 상태 메시지 생성 및 발행
        traffic_light_msg = Int8()
        if red_light_detected:
            traffic_light_msg.data = 0  # 빨간불 (정지)
            self.get_logger().info('Red light detected. Publishing state: 0 (STOP)')
        elif detect_green:
            traffic_light_msg.data = 1  # 녹색불 (주행)
            self.get_logger().info('Green light detected. Publishing state: 1 (GO)')
        elif detect_yellow:
            traffic_light_msg.data = 2  # 노란불 (대기/주의)
            self.get_logger().info('Yellow light detected. Publishing state: 2 (CAUTION)')
        else:
            traffic_light_msg.data = 3  # 신호 없음 (기본 주행)
            self.get_logger().info('No specific traffic light detected. Publishing state: 3 (DEFAULT)')

        self.traffic_light_state_publisher.publish(traffic_light_msg)

        # 기존 빨간불 감지에 따른 차량 속도 제어 로직 제거
        # if red_light_detected:
        #     if self.current_speed != 0.0:
        #         self.stop_car()
        # else:
        #     if detect_green and self.current_speed == 0.0:
        #          self.move_car_forward(0.1)
        #     elif not detect_green and not detect_yellow and not detect_red and self.current_speed == 0.0:
        #          self.move_car_forward(0.0)

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_light.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(self.cv_image, 'jpg'))
        else:
            self.pub_image_traffic_light.publish(
                self.cvBridge.cv2_to_imgmsg(self.cv_image, 'bgr8'))

    # stop_car() 및 move_car_forward() 함수 제거
    # def stop_car(self):
    #     """차량을 정지시키는 함수."""
    #     self.get_logger().info('빨간불 감지! 자동차를 정지합니다.')
    #     twist_msg = Twist()
    #     twist_msg.linear.x = 0.0
    #     twist_msg.angular.z = 0.0
    #     self.cmd_vel_publisher.publish(twist_msg)
    #     self.current_speed = 0.0

    # def move_car_forward(self, speed):
    #     """지정된 속도로 차량을 전진시키는 함수."""
    #     self.get_logger().info(f'녹색불 감지 또는 기본 이동: 차량을 {speed} m/s로 움직입니다.')
    #     twist_msg = Twist()
    #     twist_msg.linear.x = float(speed)
    #     twist_msg.angular.z = 0.0
    #     self.cmd_vel_publisher.publish(twist_msg)
    #     self.current_speed = speed


    def mask_red_traffic_light(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([self.hue_red_l, self.saturation_red_l, self.lightness_red_l])
        upper_red = np.array([self.hue_red_h, self.saturation_red_h, self.lightness_red_h])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_red_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_red_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def mask_yellow_traffic_light(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l]
            )
        upper_yellow = np.array(
            [self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h]
            )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_yellow_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_yellow_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def mask_green_traffic_light(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([self.hue_green_l, self.saturation_green_l, self.lightness_green_l])
        upper_green = np.array([self.hue_green_h, self.saturation_green_h, self.lightness_green_h])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_green_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_green_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def find_circle_of_traffic_light(self, mask, color):
        detect_result = False
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 125
        params.maxArea = 600
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.7

        detector = cv2.SimpleBlobDetector_create(params)
        keypts = detector.detect(mask)

        height, width = mask.shape[:2]
        roi_x_start = 2 * width // 3
        roi_x_end = width
        roi_y_start = height // 3
        roi_y_end = 2 * height // 3

        for i in range(len(keypts)):
            self.point_x = int(keypts[i].pt[0])
            self.point_y = int(keypts[i].pt[1])
            if roi_x_start < self.point_x < roi_x_end and roi_y_start < self.point_y < roi_y_end:
                detect_result = True
                self.get_logger().info(f'{color} light detected')
            else:
                detect_result = False

        return detect_result


def main(args=None):
    rclpy.init(args=args)
    node = DetectTrafficLight()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()