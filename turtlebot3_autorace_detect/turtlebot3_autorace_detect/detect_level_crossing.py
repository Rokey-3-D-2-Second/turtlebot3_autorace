#!/usr/bin/env python
#
# Copyright 2018 ROBOTIS CO., LTD.
# (생략)
#

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


# 점과 직선 사이의 거리 계산 함수
def fnCalcDistanceDot2Line(a, b, c, x0, y0):
    distance = abs(x0 * a + y0 * b + c) / math.sqrt(a * a + b * b)
    return distance


# 두 점 사이의 거리 계산 함수
def fnCalcDistanceDot2Dot(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


# 점 배열을 내림차순으로 정렬하고 정렬된 인덱스 반환
def fnArrangeIndexOfPoint(arr):
    new_arr = arr[:]
    arr_idx = list(range(len(arr)))
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if new_arr[i] < new_arr[j]:
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
                arr_idx[i], arr_idx[j] = arr_idx[j], arr_idx[i]
    return arr_idx


# 세 점이 거의 일직선상에 있는지 확인하는 함수 (오차 범위 내)
def fnCheckLinearity(point1, point2, point3):
    threshold_linearity = 50  # 선형성 판단 임계값
    x1, y1 = point1
    x2, y2 = point3
    if x2 - x1 != 0:
        a = (y2 - y1) / (x2 - x1)
    else:
        a = 1000  # 기울기 무한대 처리
    b = -1
    c = y1 - a * x1
    err = fnCalcDistanceDot2Line(a, b, c, point2[0], point2[1])
    return err < threshold_linearity


# 두 거리의 표준편차가 임계값 이하인지 확인하여 거리들이 거의 같은지 판단하는 함수
def fnCheckDistanceIsEqual(point1, point2, point3):
    threshold_distance_equality = 3  # 거리 동일 판단 임계값
    distance1 = fnCalcDistanceDot2Dot(point1[0], point1[1], point2[0], point2[1])
    distance2 = fnCalcDistanceDot2Dot(point2[0], point2[1], point3[0], point3[1])
    std = np.std([distance1, distance2])
    return std < threshold_distance_equality


# -----------------------------------
# ROS2 노드 클래스: DetectLevelNode
# -----------------------------------
class DetectLevelNode(Node):

    def __init__(self):
        super().__init__('detect_level')
        self.get_logger().info('detect_level 노드 시작 (ROS2)')

        # HSV 색상 범위 파라미터 설명 및 범위 지정
        hue_range = IntegerRange(from_value=0, to_value=179, step=1)
        sat_range = IntegerRange(from_value=0, to_value=255, step=1)
        light_range = IntegerRange(from_value=0, to_value=255, step=1)

        # 파라미터 설명자 생성 (파라미터 변경 시 GUI 또는 CLI에서 도움말 역할)
        hue_l_descriptor = ParameterDescriptor(description='색상 범위 하한 (Hue)')
        hue_h_descriptor = ParameterDescriptor(description='색상 범위 상한 (Hue)')
        sat_l_descriptor = ParameterDescriptor(description='채도 범위 하한 (Saturation)')
        sat_h_descriptor = ParameterDescriptor(description='채도 범위 상한 (Saturation)')
        light_l_descriptor = ParameterDescriptor(description='명도 범위 하한 (Value)')
        light_h_descriptor = ParameterDescriptor(description='명도 범위 상한 (Value)')

        # ROS2 파라미터 선언 (초기값 포함)
        self.declare_parameter('detect.level.red.hue_l', 0, descriptor=hue_l_descriptor)
        self.declare_parameter('detect.level.red.hue_h', 179, descriptor=hue_h_descriptor)
        self.declare_parameter('detect.level.red.saturation_l', 24, descriptor=sat_l_descriptor)
        self.declare_parameter('detect.level.red.saturation_h', 255, descriptor=sat_h_descriptor)
        self.declare_parameter('detect.level.red.lightness_l', 207, descriptor=light_l_descriptor)
        self.declare_parameter('detect.level.red.lightness_h', 255, descriptor=light_h_descriptor)

        # 감지 보정 모드 여부 파라미터 선언 (디버깅용)
        self.declare_parameter('is_detection_calibration_mode', False)

        # 파라미터 값 읽기
        self.hue_red_l = self.get_parameter('detect.level.red.hue_l').value
        self.hue_red_h = self.get_parameter('detect.level.red.hue_h').value
        self.saturation_red_l = self.get_parameter('detect.level.red.saturation_l').value
        self.saturation_red_h = self.get_parameter('detect.level.red.saturation_h').value
        self.lightness_red_l = self.get_parameter('detect.level.red.lightness_l').value
        self.lightness_red_h = self.get_parameter('detect.level.red.lightness_h').value
        self.is_calibration_mode = self.get_parameter('is_detection_calibration_mode').value

        # 파라미터 동적 변경 콜백 함수 등록
        self.add_on_set_parameters_callback(self.on_parameter_change)

        # 구독 및 발행할 이미지 타입 설정 ('raw' 또는 'compressed')
        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        # 레벨 크로싱 단계 상태 정의 enum (통과, 종료)
        self.StepOfLevelCrossing = Enum('StepOfLevelCrossing', 'pass_level exit')

        # 초기 상태 변수들
        self.is_level_crossing_finished = False
        self.stop_bar_count = 0
        self.counter = 1
        self.cv_image = None  # OpenCV 이미지 저장용

        self.cv_bridge = CvBridge()

        # 이미지 결과 출력을 위한 퍼블리셔 생성
        if self.pub_image_type == 'compressed':
            self.pub_image_level = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 10)
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(CompressedImage, '/detect/image_output_sub1/compressed', 10)
        else:  # raw 이미지
            self.pub_image_level = self.create_publisher(Image, '/detect/image_output', 10)
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(
                    Image, '/detect/image_output_sub1', 10)
                
        self.pub_level_bar = self.create_publisher(Bool, '/detect/level_bar', 10)

        # 이미지 입력을 위한 구독자 생성 (raw / compressed 타입에 따라 다름)
        if self.sub_image_type == 'compressed':
            self.create_subscription(CompressedImage, '/detect/image_input/compressed', self.get_image, 10)
        else:
            self.create_subscription(Image, '/detect/image_input', self.get_image, 10)

        # 레벨 크로싱 동작 명령 구독자 생성
        self.create_subscription(UInt8, '/detect/level_crossing_order', self.level_crossing_order, 10)

        # 15Hz 주기로 타이머 콜백 실행 (레벨 크로싱 감지 함수 호출)
        self.timer = self.create_timer(1.0/15.0, self.timer_callback)

        time.sleep(1.0)  # 초기화 대기

    # 파라미터 변경 시 호출되는 콜백 함수
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
        self.get_logger().info('파라미터가 동적으로 업데이트되었습니다.')
        return SetParametersResult(successful=True)

    # 타이머 콜백: 이미지가 있으면 레벨 크로싱 검출 수행
    def timer_callback(self):
        if self.cv_image is not None:
            self.find_level()

    # 이미지 구독 콜백 함수: 3프레임마다 처리 (속도 조절)
    def get_image(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        # 이미지 타입에 따라 OpenCV 이미지로 변환
        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            try:
                self.cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error('CV Bridge 에러: %s' % str(e))

    # 레벨 크로싱 동작 명령 처리 함수
    def level_crossing_order(self, order_msg):
        pub_level_crossing_return = UInt8()
        if order_msg.data == self.StepOfLevelCrossing.pass_level.value:
            # 레벨 감지 후 속도 줄이기
            while rclpy.ok():
                is_level_detected, _, _ = self.find_level()
                rclpy.spin_once(self, timeout_sec=0.01)
                if is_level_detected:
                    self.get_logger().info('레벨 감지됨')
                    max_vel_msg = Float64()
                    max_vel_msg.data = 0.03
                    self.pub_max_vel.publish(max_vel_msg)
                    break

            # 레벨에 가까워지면 멈춤
            while rclpy.ok():
                _, is_level_close, _ = self.find_level()
                rclpy.spin_once(self, timeout_sec=0.01)
                if is_level_close:
                    self.get_logger().info('정지 신호')
                    max_vel_msg = Float64()
                    max_vel_msg.data = 0.0
                    self.pub_max_vel.publish(max_vel_msg)
                    break

            # 레벨이 열리면 출발
            while rclpy.ok():
                _, _, is_level_opened = self.find_level()
                rclpy.spin_once(self, timeout_sec=0.01)
                if is_level_opened:
                    self.get_logger().info('출발 신호')
                    max_vel_msg = Float64()
                    max_vel_msg.data = 0.05
                    self.pub_max_vel.publish(max_vel_msg)
                    break

            pub_level_crossing_return.data = self.StepOfLevelCrossing.exit.value

        self.get_logger().info(pub_level_crossing_return.data)
        time.sleep(3.0)

    # 레벨 크로싱 감지를 위한 메인 처리 함수
    def find_level(self):
        mask = self.mask_red_of_level()
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return self.find_rect_of_level(mask)

    # 빨간색 영역만 필터링 하는 함수 (HSV 색상 범위 내)
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

        # 보정 모드일 때 필터링 이미지 퍼블리시
        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                comp_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(mask, dst_format='jpg')
                self.pub_image_color_filtered.publish(comp_img_msg)
            else:
                img_msg = self.cv_bridge.cv2_to_imgmsg(mask, encoding='mono8')
                self.pub_image_color_filtered.publish(img_msg)

        # 필터링 결과 반전 (검정-흰색 반전)
        mask = cv2.bitwise_not(mask)
        return mask

    # 필터링된 이미지에서 레벨 크로싱 관련 도형 검출 및 상태 판단
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

            # 선형성 및 거리 등 검사에 필요한 3개 점 좌표 설정
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
                is_level_opened = True  # 레벨 크로싱이 열려있음
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
                        is_level_close = True  # 레벨에 가까움 (정지 신호)
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
        node.get_logger().info('키보드 인터럽트(SIGINT) 발생')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
