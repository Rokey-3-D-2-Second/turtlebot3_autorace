#!/usr/bin/env python
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
# Authors:
#   - Leon Jung, Gilbert, Ashe Kim, ChanHyeong Lee
#   - [AuTURBO] Kihoon Kim (https://github.com/auturbo)

from enum import Enum
import math
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8
from std_msgs.msg import Bool


def fnCalcDistanceDot2Line(a, b, c, x0, y0):
    distance = abs(x0 * a + y0 * b + c) / math.sqrt(a * a + b * b)
    return distance


def fnCalcDistanceDot2Dot(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def fnArrangeIndexOfPoint(arr):
    new_arr = arr[:]
    arr_idx = list(range(len(arr)))
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if new_arr[i] < new_arr[j]:
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
                arr_idx[i], arr_idx[j] = arr_idx[j], arr_idx[i]
    return arr_idx


def fnCheckLinearity(point1, point2, point3):
    threshold_linearity = 50
    x1, y1 = point1
    x2, y2 = point3
    if x2 - x1 != 0:
        a = (y2 - y1) / (x2 - x1)
    else:
        a = 1000
    b = -1
    c = y1 - a * x1
    err = fnCalcDistanceDot2Line(a, b, c, point2[0], point2[1])
    return err < threshold_linearity


def fnCheckDistanceIsEqual(point1, point2, point3):
    threshold_distance_equality = 3
    distance1 = fnCalcDistanceDot2Dot(point1[0], point1[1], point2[0], point2[1])
    distance2 = fnCalcDistanceDot2Dot(point2[0], point2[1], point3[0], point3[1])
    std = np.std([distance1, distance2])
    return std < threshold_distance_equality


# ------------------------
# ROS2 Node: DetectLevelNode
# ------------------------
class DetectLevelNode(Node):

    def __init__(self):
        super().__init__('detect_level')
        self.get_logger().info('Starting detect_level node (ROS2)')

        hue_range = IntegerRange(from_value=0, to_value=179, step=1)
        sat_range = IntegerRange(from_value=0, to_value=255, step=1)
        light_range = IntegerRange(from_value=0, to_value=255, step=1)

        hue_l_descriptor = ParameterDescriptor(
            description='Lower hue threshold',
            integer_range=[hue_range]
        )
        hue_h_descriptor = ParameterDescriptor(
            description='Upper hue threshold',
            integer_range=[hue_range]
        )
        sat_l_descriptor = ParameterDescriptor(
            description='Lower saturation threshold',
            integer_range=[sat_range]
        )
        sat_h_descriptor = ParameterDescriptor(
            description='Upper saturation threshold',
            integer_range=[sat_range]
        )
        light_l_descriptor = ParameterDescriptor(
            description='Lower value (lightness) threshold',
            integer_range=[light_range]
        )
        light_h_descriptor = ParameterDescriptor(
            description='Upper value (lightness) threshold',
            integer_range=[light_range]
        )

        # delcare parameters
        self.declare_parameter('detect.level.red.hue_l', 0, descriptor=hue_l_descriptor)
        self.declare_parameter('detect.level.red.hue_h', 179, descriptor=hue_h_descriptor)
        self.declare_parameter('detect.level.red.saturation_l', 24, descriptor=sat_l_descriptor)
        self.declare_parameter('detect.level.red.saturation_h', 255, descriptor=sat_h_descriptor)
        self.declare_parameter('detect.level.red.lightness_l', 207, descriptor=light_l_descriptor)
        self.declare_parameter('detect.level.red.lightness_h', 255, descriptor=light_h_descriptor)

        self.declare_parameter('is_detection_calibration_mode', False)

        # get parameters
        self.hue_red_l = self.get_parameter('detect.level.red.hue_l').value
        self.hue_red_h = self.get_parameter('detect.level.red.hue_h').value
        self.saturation_red_l = self.get_parameter('detect.level.red.saturation_l').value
        self.saturation_red_h = self.get_parameter('detect.level.red.saturation_h').value
        self.lightness_red_l = self.get_parameter('detect.level.red.lightness_l').value
        self.lightness_red_h = self.get_parameter('detect.level.red.lightness_h').value
        self.is_calibration_mode = self.get_parameter('is_detection_calibration_mode').value

        self.add_on_set_parameters_callback(self.on_parameter_change)

        self.sub_image_type = 'raw'  # 'raw' or 'compressed'
        self.pub_image_type = 'compressed'  # 'raw' or 'compressed'

        self.StepOfLevelCrossing = Enum('StepOfLevelCrossing', 'pass_level exit')

        self.is_level_crossing_finished = False
        self.stop_bar_count = 0
        self.counter = 1
        self.cv_image = None

        self.cv_bridge = CvBridge()

        # create publishers
        if self.pub_image_type == 'compressed':
            self.pub_image_level = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 10)
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 10)
        else:  # raw
            self.pub_image_level = self.create_publisher(
                Image, '/detect/image_output', 10)
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(
                    Image, '/detect/image_output_sub1', 10)
                
        self.pub_level_bar = self.create_publisher(Bool, '/detect/level_bar', 10)

        # create subscribers
        if self.sub_image_type == 'compressed':
            self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.get_image, 10)
        else:  # raw
            self.create_subscription(
                Image, '/detect/image_input', self.get_image, 10)

        self.create_subscription(
            UInt8, '/detect/level_crossing_order', self.level_crossing_order, 10)

        self.timer = self.create_timer(1.0/15.0, self.timer_callback)

        time.sleep(1.0)

    def on_parameter_change(self, params):
        for param in params:
            if param.name == 'detect.level.red.hue_l':
                self.hue_red_l = param.value
            elif param.name == 'detect.level.red.hue_h':
                self.hue_red_h = param.value
            elif param.name == 'detect.level.red.saturation_l':
                self.saturation_red_l = param.value
            elif param.name == 'detect.level.red.saturation_h':
                self.saturation_red_h = param.value
            elif param.name == 'detect.level.red.lightness_l':
                self.lightness_red_l = param.value
            elif param.name == 'detect.level.red.lightness_h':
                self.lightness_red_h = param.value
        self.get_logger().info('Dynamic parameters updated.')
        return SetParametersResult(successful=True)

    def timer_callback(self):
        if self.cv_image is not None:
            self.find_level()

    def get_image(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            try:
                self.cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error('CV Bridge error: %s' % str(e))

    def level_crossing_order(self, order_msg):
        pub_level_crossing_return = UInt8()
        if order_msg.data == self.StepOfLevelCrossing.pass_level.value:
            while rclpy.ok():
                is_level_detected, _, _ = self.find_level()
                rclpy.spin_once(self, timeout_sec=0.01)
                if is_level_detected:
                    self.get_logger().info('Level Detected')
                    max_vel_msg = Float64()
                    max_vel_msg.data = 0.03
                    self.pub_max_vel.publish(max_vel_msg)
                    break

            while rclpy.ok():
                _, is_level_close, _ = self.find_level()
                rclpy.spin_once(self, timeout_sec=0.01)
                if is_level_close:
                    self.get_logger().info('STOP')
                    max_vel_msg = Float64()
                    max_vel_msg.data = 0.0
                    self.pub_max_vel.publish(max_vel_msg)
                    break

            while rclpy.ok():
                _, _, is_level_opened = self.find_level()
                rclpy.spin_once(self, timeout_sec=0.01)
                if is_level_opened:
                    self.get_logger().info('GO')
                    max_vel_msg = Float64()
                    max_vel_msg.data = 0.05
                    self.pub_max_vel.publish(max_vel_msg)
                    break

            pub_level_crossing_return.data = self.StepOfLevelCrossing.exit.value

        self.get_logger().info(pub_level_crossing_return.data)
        time.sleep(3.0)

    def find_level(self):
        mask = self.mask_red_of_level()
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return self.find_rect_of_level(mask)

    def mask_red_of_level(self):
        if self.cv_image is None:
            return None

        image = self.cv_image.copy()
        
        # [ROI 적용: 우측 70%, 상단 60%]
        h, w, _ = image.shape
        roi_x = int(w * 0.3)
        roi_y = 0
        roi_w = int(w * 0.7)
        roi_h = int(h * 0.6)
        roi_img = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # lower_red = np.array([self.hue_red_l, self.saturation_red_l, self.lightness_red_l])
        # upper_red = np.array([self.hue_red_h, self.saturation_red_h, self.lightness_red_h])
        # mask = cv2.inRange(hsv, lower_red, upper_red)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                comp_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(mask, dst_format='jpg')
                self.pub_image_color_filtered.publish(comp_img_msg)
            else:
                img_msg = self.cv_bridge.cv2_to_imgmsg(mask, encoding='mono8')
                self.pub_image_color_filtered.publish(img_msg)

        mask = cv2.bitwise_not(mask)
        return mask

    def find_rect_of_level(self, mask):
        # --- 결과 플래그 초기화 ---
        is_level_detected = False     # 바(bar)가 감지됨
        is_level_close = False        # 바가 가까움 (정지)
        is_level_opened = False       # 바가 열림 (통과 가능)

        # --- Blob(키포인트) 검출 파라미터 세팅 ---
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True        # 영역(크기) 필터 사용
        params.minArea = 200              # Blob 최소 면적(픽셀) (바가 얇으면 더 낮춰야 함)
        params.maxArea = 40000              # Blob 최대 면적(픽셀)        
        params.filterByConvexity = False   # 볼록성(거의 원형에 가까운 것만) 필터
        params.minConvexity = 0.9         # (0~1 사이, 더 낮추면 더 많은 blob 인식)

        # --- Blob(키포인트) 검출기 생성 및 검출 ---
        detector = cv2.SimpleBlobDetector_create(params)
        keypts = detector.detect(mask)    # 마스크에서 Blob(keypoint) 검출

        # --- ROI offset 보정 (마스크가 ROI였으면, 원본 좌표로 변환) ---
        roi_x = int(self.cv_image.shape[1] * 0.3)
        roi_y = 0
        # Blob 키포인트 좌표를 원본 영상 위치로 이동 (cv2.drawKeypoints에서 위치 정확히 매칭)
        keypts = [
            cv2.KeyPoint(
                kp.pt[0] + roi_x,  # X좌표 보정
                kp.pt[1] + roi_y,  # Y좌표 보정
                kp.size,
                kp.angle,
                kp.response,
                kp.octave,
                kp.class_id
            ) for kp in keypts
        ]

        # --- 키포인트(Blob) 시각화: 원본 이미지에 키포인트 그림 ---
        frame = cv2.drawKeypoints(
            self.cv_image,
            keypts,
            np.array([]),
            (0, 255, 255),        # 노란색
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # --- 키포인트 개수 체크 및 바(bar) 상태 추론 ---
        mean_x = 0.0
        mean_y = 0.0

        # 바가 3개 검출되면 (= 건널목바 3개로 판단) 아래 로직 수행
        if len(keypts) == 3:
            # 키포인트들의 무게중심 계산
            for kp in keypts:
                mean_x += kp.pt[0] / 3
                mean_y += kp.pt[1] / 3

            # 각 키포인트와 무게중심간 거리 구하기
            arr_distances = [
                fnCalcDistanceDot2Dot(mean_x, mean_y, kp.pt[0], kp.pt[1]) for kp in keypts
            ]
            # 가장 멀리 떨어진 순서대로 인덱스 정렬
            idx_order = fnArrangeIndexOfPoint(arr_distances)

            # 바(bar) 2개를 선으로 그림(시각화용)
            frame = cv2.line(
                frame,
                (int(keypts[idx_order[0]].pt[0]), int(keypts[idx_order[0]].pt[1])),
                (int(keypts[idx_order[1]].pt[0]), int(keypts[idx_order[1]].pt[1])),
                (255, 0, 0), 5)
            # 무게중심도 표시(시각화용)
            frame = cv2.circle(frame, (int(mean_x), int(mean_y)), 5, (255, 255, 0), -1)

            # 각 bar의 좌표를 추출
            point1 = [int(keypts[idx_order[0]].pt[0]), int(keypts[idx_order[0]].pt[1] - 1)]
            point2 = [int(keypts[idx_order[2]].pt[0]), int(keypts[idx_order[2]].pt[1] - 1)]
            point3 = [int(keypts[idx_order[1]].pt[0]), int(keypts[idx_order[1]].pt[1] - 1)]

            # 수평/수직/기울기 등 계산 (slope)
            dx = point3[0] - point1[0]
            dy = point3[1] - point1[1]
            if dx == 0:
                slope = float('inf')
            else:
                slope = dy / dx

            # --- 상태 판별 로직 ---
            # (1) 세 bar가 수직(혹은 기울기가 매우 큰 경우)면 "열림(open)"
            if dx == 0 or abs(slope) > 2.0:
                is_level_opened = True
                self.stop_bar_state = 'go'
                self.get_logger().info(self.stop_bar_state)
            else:
                # (2) 그 외엔 선형성(linearity), 거리균등성 판별
                is_rects_linear = fnCheckLinearity(point1, point2, point3)
                is_rects_dist_equal = fnCheckDistanceIsEqual(point1, point2, point3)

                # (3) 둘 중 하나라도 맞으면 바(bar) 감지 성공
                if is_rects_linear or is_rects_dist_equal:
                    distance_bar2car = 25 / fnCalcDistanceDot2Dot(
                        point1[0], point1[1], point2[0], point2[1])
                    self.stop_bar_count = 50
                    # 거리 조건에 따라 slowdown/stop을 판별
                    if distance_bar2car > 0.9:
                        is_level_detected = True
                        self.stop_bar_state = 'slowdown'
                        self.get_logger().info(self.stop_bar_state)
                    else:
                        is_level_close = True
                        self.stop_bar_state = 'stop'
                        self.get_logger().info(self.stop_bar_state)
            
            msg = Bool()
            msg.data = True
            self.pub_level_bar.publish(msg)

        # --- 키포인트가 1개 이하일 때(= bar가 없다고 판단): 무조건 open으로 처리 ---
        elif len(keypts) <= 1:
            is_level_opened = True
            self.stop_bar_state = 'go'

        # --- 결과 이미지 퍼블리시(원본에 bar 위치, 상태 시각화) ---
        if self.pub_image_type == 'compressed':
            comp_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
            self.pub_image_level.publish(comp_img_msg)
        else:
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub_image_level.publish(img_msg)

        # --- 상태 반환: (바 감지/정지/열림 여부) ---
        return is_level_detected, is_level_close, is_level_opened


def main(args=None):
    rclpy.init(args=args)
    node = DetectLevelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
