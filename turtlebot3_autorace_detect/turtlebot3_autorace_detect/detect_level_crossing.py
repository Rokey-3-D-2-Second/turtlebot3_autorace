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

from enum import Enum  # 열거형(Enum)을 사용하기 위한 모듈 임포트
import math  # 수학 함수 (예: sqrt, abs)를 사용하기 위한 모듈 임포트
import time  # 시간 관련 함수 (예: sleep)를 사용하기 위한 모듈 임포트

import cv2  # OpenCV 라이브러리 임포트 (이미지 처리)
from cv_bridge import CvBridge  # ROS 이미지 메시지와 OpenCV 이미지 간의 변환을 위한 브리지 임포트
import numpy as np  # 수치 계산을 위한 NumPy 라이브러리 임포트
from rcl_interfaces.msg import IntegerRange  # ROS 2 파라미터 설명을 위한 IntegerRange 메시지 임포트
from rcl_interfaces.msg import ParameterDescriptor  # ROS 2 파라미터 설명을 위한 ParameterDescriptor 메시지 임포트
from rcl_interfaces.msg import SetParametersResult  # ROS 2 파라미터 설정 결과 메시지 임포트
import rclpy  # ROS 2 Python 클라이언트 라이브러리 임포트
from rclpy.node import Node  # ROS 2 노드 클래스 임포트
from sensor_msgs.msg import CompressedImage  # 압축된 이미지 메시지 타입 임포트
from sensor_msgs.msg import Image  # 일반 이미지 메시지 타입 임포트
from std_msgs.msg import Float64  # 64비트 부동 소수점 메시지 타입 임포트
from std_msgs.msg import UInt8  # 8비트 부호 없는 정수 메시지 타입 임포트
from std_msgs.msg import Int8
from std_msgs.msg import Bool


# 점과 직선 사이의 거리 계산 함수
def fnCalcDistanceDot2Line(a, b, c, x0, y0):
    """
    점 (x0, y0)와 직선 Ax + By + C = 0 사이의 거리를 계산합니다.
    """
    distance = abs(x0 * a + y0 * b + c) / math.sqrt(a * a + b * b)  # 점과 직선 사이의 거리 공식
    return distance  # 계산된 거리 반환


# 두 점 사이의 거리 계산 함수
def fnCalcDistanceDot2Dot(x1, y1, x2, y2):
    """
    두 점 (x1, y1)와 (x2, y2) 사이의 유클리드 거리를 계산합니다.
    """
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # 두 점 사이의 거리 공식
    return distance  # 계산된 거리 반환


# 점 배열을 내림차순으로 정렬하고 정렬된 인덱스 반환
def fnArrangeIndexOfPoint(arr):
    """
    배열의 값을 기준으로 인덱스를 내림차순으로 정렬합니다.
    (예: 거리가 먼 순서대로 점의 인덱스를 반환)
    """
    new_arr = arr[:]  # 원본 배열의 복사본 생성
    arr_idx = list(range(len(arr)))  # 인덱스 리스트 초기화
    for i in range(len(arr)):  # 배열의 각 요소에 대해 반복
        for j in range(i + 1, len(arr)):  # 현재 요소 이후의 요소들에 대해 반복
            if new_arr[i] < new_arr[j]:  # 현재 요소가 다음 요소보다 작으면 (내림차순 정렬)
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]  # 값 교환
                arr_idx[i], arr_idx[j] = arr_idx[j], arr_idx[i]  # 인덱스도 함께 교환
    return arr_idx  # 정렬된 인덱스 리스트 반환


# 세 점이 거의 일직선상에 있는지 확인하는 함수 (오차 범위 내)
def fnCheckLinearity(point1, point2, point3):
    """
    세 점이 거의 일직선상에 있는지 확인합니다.
    두 끝점을 잇는 직선과 중간 점 사이의 거리를 계산하여 임계값보다 작으면 선형으로 간주합니다.
    """
    threshold_linearity = 50  # 선형성 판단을 위한 임계 거리
    x1, y1 = point1  # 첫 번째 점의 좌표
    x2, y2 = point3  # 세 번째 점의 좌표
    if x2 - x1 != 0:  # 분모가 0이 아닌 경우 기울기 계산
        a = (y2 - y1) / (x2 - x1)  # 직선의 기울기 A
    else:
        a = 1000  # x2 - x1이 0이면 수직선이므로 기울기를 매우 큰 값으로 설정
    b = -1  # 직선 방정식 Ax + By + C = 0 에서 B 값
    c = y1 - a * x1  # 직선 방정식 Ax + By + C = 0 에서 C 값
    err = fnCalcDistanceDot2Line(a, b, c, point2[0], point2[1])  # 중간 점과 직선 사이의 거리 계산
    return err < threshold_linearity  # 거리가 임계값보다 작으면 True (선형) 반환


# 두 거리의 표준편차가 임계값 이하인지 확인하여 거리들이 거의 같은지 판단하는 함수
def fnCheckDistanceIsEqual(point1, point2, point3):
    """
    연속된 두 점 쌍 사이의 거리가 거의 동일한지 확인합니다.
    (예: 횡단보도 바의 길이가 일정한지 확인)
    """
    threshold_distance_equality = 3  # 거리 균등성 판단을 위한 표준 편차 임계값
    distance1 = fnCalcDistanceDot2Dot(point1[0], point1[1], point2[0], point2[1])  # 첫 번째 점과 두 번째 점 사이의 거리
    distance2 = fnCalcDistanceDot2Dot(point2[0], point2[1], point3[0], point3[1])  # 두 번째 점과 세 번째 점 사이의 거리
    std = np.std([distance1, distance2])  # 두 거리의 표준 편차 계산
    return std < threshold_distance_equality  # 표준 편차가 임계값보다 작으면 True (거의 동일) 반환


# -----------------------------------
# ROS2 노드 클래스: DetectLevelNode
# -----------------------------------
class DetectLevelNode(Node):
    """
    ROS 2 노드: 건널목(Level Crossing)을 감지하고 상태를 발행합니다.
    """

    def __init__(self):
        """
        DetectLevelNode 클래스의 생성자입니다.
        ROS 2 노드를 초기화하고, 이미지 구독 및 발행, 파라미터 설정 등을 수행합니다.
        """
        super().__init__('detect_level')  # 'detect_level' 이름으로 ROS 2 노드 초기화
        self.get_logger().info('Starting detect_level node (ROS2)')  # 노드 시작 로그 메시지 출력

        # 파라미터 설명 (IntegerRange와 ParameterDescriptor 사용)
        hue_range = IntegerRange(from_value=0, to_value=179, step=1)  # Hue 값 범위 (OpenCV HSV)
        sat_range = IntegerRange(from_value=0, to_value=255, step=1)  # Saturation 값 범위
        light_range = IntegerRange(from_value=0, to_value=255, step=1)  # Lightness (Value) 값 범위

        hue_l_descriptor = ParameterDescriptor(
            description='Lower hue threshold',  # 하위 Hue 임계값 설명
            integer_range=[hue_range]  # Hue 값 범위 지정
        )
        hue_h_descriptor = ParameterDescriptor(
            description='Upper hue threshold',  # 상위 Hue 임계값 설명
            integer_range=[hue_range]  # Hue 값 범위 지정
        )
        sat_l_descriptor = ParameterDescriptor(
            description='Lower saturation threshold',  # 하위 Saturation 임계값 설명
            integer_range=[sat_range]  # Saturation 값 범위 지정
        )
        sat_h_descriptor = ParameterDescriptor(
            description='Upper saturation threshold',  # 상위 Saturation 임계값 설명
            integer_range=[sat_range]  # Saturation 값 범위 지정
        )
        light_l_descriptor = ParameterDescriptor(
            description='Lower value (lightness) threshold',  # 하위 Lightness (Value) 임계값 설명
            integer_range=[light_range]  # Lightness (Value) 값 범위 지정
        )
        light_h_descriptor = ParameterDescriptor(
            description='Upper value (lightness) threshold',  # 상위 Lightness (Value) 임계값 설명
            integer_range=[light_range]  # Lightness (Value) 값 범위 지정
        )

        # delcare parameters
        self.declare_parameter('detect.level.red.hue_l', 0, descriptor=hue_l_descriptor)
        self.declare_parameter('detect.level.red.hue_h', 179, descriptor=hue_h_descriptor)
        self.declare_parameter('detect.level.red.saturation_l', 24, descriptor=sat_l_descriptor)
        self.declare_parameter('detect.level.red.saturation_h', 255, descriptor=sat_h_descriptor)
        self.declare_parameter('detect.level.red.lightness_l', 207, descriptor=light_l_descriptor)
        self.declare_parameter('detect.level.red.lightness_h', 255, descriptor=light_h_descriptor)

        self.declare_parameter('is_detection_calibration_mode', False)  # 캘리브레이션 모드 여부 파라미터

        # 파라미터 값 가져오기
        self.hue_yellow_l = self.get_parameter('detect.level.yellow.hue_l').value  # 노란색 하위 Hue 값
        self.hue_yellow_h = self.get_parameter('detect.level.yellow.hue_h').value  # 노란색 상위 Hue 값
        self.saturation_yellow_l = self.get_parameter('detect.level.yellow.saturation_l').value  # 노란색 하위 Saturation 값
        self.saturation_yellow_h = self.get_parameter('detect.level.yellow.saturation_h').value  # 노란색 상위 Saturation 값
        self.lightness_yellow_l = self.get_parameter('detect.level.yellow.lightness_l').value  # 노란색 하위 Lightness 값
        self.lightness_yellow_h = self.get_parameter('detect.level.yellow.lightness_h').value  # 노란색 상위 Lightness 값
        self.is_calibration_mode = self.get_parameter('is_detection_calibration_mode').value  # 캘리브레이션 모드 여부

        self.add_on_set_parameters_callback(self.on_parameter_change)  # 파라미터 변경 콜백 함수 등록

        self.sub_image_type = 'raw'  # 구독할 이미지 메시지 타입 ('raw' 또는 'compressed')
        self.pub_image_type = 'compressed'  # 발행할 이미지 메시지 타입 ('raw' 또는 'compressed')

        self.StepOfLevelCrossing = Enum('StepOfLevelCrossing', 'pass_level exit')  # 건널목 통과 단계 열거형 정의

        self.is_level_crossing_finished = False  # 건널목 통과 완료 여부
        self.stop_bar_count = 0  # 정지 바 감지 카운터
        self.counter = 1  # 프레임 스킵을 위한 카운터
        self.cv_image = None  # 현재 OpenCV 이미지

        self.cv_bridge = CvBridge()  # CvBridge 객체 생성
        # __init__ 메서드 내에 추가
        self.pub_level_crossing_state = self.create_publisher(Int8, '/level_crossing_state', 10)
        self.pub_level_crossing_state_publisher = self.create_publisher(
            Int8, '/pub_level_crossing_state', 10)
        # 이미지 발행자 생성
        if self.pub_image_type == 'compressed':
            self.pub_image_level = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 10)  # 압축 이미지 발행자
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 10)  # 색상 필터링된 이미지 발행자 (캘리브레이션 모드)
        else:  # raw
            self.pub_image_level = self.create_publisher(
                Image, '/detect/image_output', 10)  # 원본 이미지 발행자
            if self.is_calibration_mode:
                self.pub_image_color_filtered = self.create_publisher(
                    Image, '/detect/image_output_sub1', 10)
                
        self.pub_level_bar = self.create_publisher(Bool, '/detect/level_bar', 10)

        # 이미지 구독자 생성
        if self.sub_image_type == 'compressed':
            self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.get_image, 10)  # 압축 이미지 구독자
        else:  # raw
            self.create_subscription(
                Image, '/detect/image_input', self.get_image, 10)  # 원본 이미지 구독자

        self.create_subscription(
            Int8, '/detect/level_crossing_order', self.level_crossing_order, 10)  # 건널목 통과 명령 구독자

        self.timer = self.create_timer(1.0/15.0, self.timer_callback)  # 1초에 15번 (약 66ms 간격) 타이머 콜백 호출

        time.sleep(1.0)  # 초기화 후 1초 대기

    # 파라미터 변경 시 호출되는 콜백 함수
    def on_parameter_change(self, params):
        """
        파라미터가 변경될 때 호출되는 콜백 함수입니다.
        """
        for param in params:  # 변경된 각 파라미터에 대해 반복
            if param.name == 'detect.level.yellow.hue_l':
                self.hue_yellow_l = param.value  # 노란색 하위 Hue 값 업데이트
            elif param.name == 'detect.level.yellow.hue_h':
                self.hue_yellow_h = param.value  # 노란색 상위 Hue 값 업데이트
            elif param.name == 'detect.level.yellow.saturation_l':
                self.saturation_yellow_l = param.value  # 노란색 하위 Saturation 값 업데이트
            elif param.name == 'detect.level.yellow.saturation_h':
                self.saturation_yellow_h = param.value  # 노란색 상위 Saturation 값 업데이트
            elif param.name == 'detect.level.yellow.lightness_l':
                self.lightness_yellow_l = param.value  # 노란색 하위 Lightness 값 업데이트
            elif param.name == 'detect.level.yellow.lightness_h':
                self.lightness_yellow_h = param.value  # 노란색 상위 Lightness 값 업데이트
        self.get_logger().info('Dynamic parameters updated.')  # 파라미터 업데이트 로그 메시지 출력
        return SetParametersResult(successful=True)  # 파라미터 설정 성공 결과 반환

    def timer_callback(self):
        if self.cv_image is not None:
            self.find_level()

    def get_image(self, image_msg):
        """
        이미지 메시지를 구독하는 콜백 함수입니다.
        처리 속도 향상을 위해 프레임을 스킵할 수 있습니다.
        """
        if self.counter % 1 != 0:  # 3프레임마다 한 번씩만 처리
            self.counter += 1  # 카운터 증가
            return  # 현재 프레임 스킵
        else:
            self.counter = 1  # 카운터 리셋

        if self.sub_image_type == 'compressed':  # 압축 이미지인 경우
            np_arr = np.frombuffer(image_msg.data, np.int8)  # 바이트 배열을 NumPy 배열로 변환
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # NumPy 배열을 OpenCV 이미지로 디코딩
        else:  # 원본 이미지인 경우
            try:
                self.cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')  # ROS 이미지 메시지를 BGR8 형식의 OpenCV 이미지로 변환
            except Exception as e:
                self.get_logger().error('CV Bridge error: %s' % str(e))  # 변환 오류 발생 시 에러 로그 출력

    # 레벨 크로싱 동작 명령 처리 함수
    def level_crossing_order(self, order_msg):
        """
        건널목 통과 명령 메시지를 수신하는 콜백 함수입니다.
        수신된 명령에 따라 건널목 통과 절차를 진행합니다.
        """
        pub_level_crossing_return = Int8()  # 반환 메시지 객체 생성
        pub_level_crossing_msg = Int8()
        if order_msg.data == self.StepOfLevelCrossing.pass_level.value:  # 'pass_level' 명령이 수신된 경우
            while rclpy.ok():  # ROS 2가 실행 중인 동안 반복
                is_level_detected, _, _ = self.find_level()  # 건널목 감지 상태 확인
                rclpy.spin_once(self, timeout_sec=0.01)  # 짧은 시간 동안 ROS 2 콜백 처리
                if is_level_detected:  # 건널목이 감지되면
                    self.get_logger().info('Level Detected')  # 감지 로그 출력
                    max_vel_msg = Float64()  # 최대 속도 메시지 생성
                    max_vel_msg.data = 0.03  # 속도 설정
                    # self.pub_max_vel.publish(max_vel_msg)  # (주석 처리됨) 최대 속도 발행
                    break  # 루프 종료

            while rclpy.ok():  # ROS 2가 실행 중인 동안 반복
                _, is_level_close, _ = self.find_level()  # 건널목 근접 상태 확인
                rclpy.spin_once(self, timeout_sec=0.01)  # 짧은 시간 동안 ROS 2 콜백 처리
                if is_level_close:  # 건널목이 가까워지면
                    self.get_logger().info('STOP')  # 'STOP' 로그 출력
                    max_vel_msg = Float64()  # 최대 속도 메시지 생성
                    max_vel_msg.data = 0.0  # 속도를 0으로 설정 (정지)
                    # self.pub_max_vel.publish(max_vel_msg)  # (주석 처리됨) 최대 속도 발행
                    pub_level_crossing_msg.data = 4
                    break  # 루프 종료

            while rclpy.ok():  # ROS 2가 실행 중인 동안 반복
                _, _, is_level_opened = self.find_level()  # 건널목 열림 상태 확인
                rclpy.spin_once(self, timeout_sec=0.01)  # 짧은 시간 동안 ROS 2 콜백 처리
                if is_level_opened:  # 건널목이 열리면
                    self.get_logger().info('GO')  # 'GO' 로그 출력
                    max_vel_msg = Float64()  # 최대 속도 메시지 생성
                    max_vel_msg.data = 0.05  # 속도 설정
                    # self.pub_max_vel.publish(max_vel_msg)  # (주석 처리됨) 최대 속도 발행
                    break  # 루프 종료

            self.pub_level_crossing_state_publisher.publish(pub_level_crossing_msg)
            pub_level_crossing_return.data = self.StepOfLevelCrossing.exit.value  # 'exit' 단계로 반환 메시지 설정

        self.get_logger().info(pub_level_crossing_return.data)
        time.sleep(3.0)

    def find_level(self):
        """
        이미지에서 건널목을 찾기 위한 주 함수입니다.
        먼저 노란색 마스크를 생성하고, 블러 처리 후 사각형을 감지합니다.
        """
        mask = self.mask_yellow_of_level()  # 노란색 영역 마스크 생성
        if mask is None: # 마스크가 None이면 (이미지가 없으면) 처리 중단
            return False, False, False
        mask = cv2.GaussianBlur(mask, (5, 5), 0)  # 가우시안 블러를 적용하여 노이즈 감소
        return self.find_rect_of_level(mask)  # 마스크에서 사각형을 찾아 건널목 상태 반환

    def mask_yellow_of_level(self):
        """
        이미지에서 노란색 영역을 마스크로 추출합니다.
        캘리브레이션 모드인 경우 필터링된 마스크 이미지를 발행합니다.
        """
        if self.cv_image is None:  # OpenCV 이미지가 없으면 None 반환
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

        if self.is_calibration_mode:  # 캘리브레이션 모드인 경우
            if self.pub_image_type == 'compressed':
                comp_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(mask, dst_format='jpg')  # 마스크 이미지를 JPG로 압축하여 변환
                self.pub_image_color_filtered.publish(comp_img_msg)  # 압축된 마스크 이미지 발행
            else:
                img_msg = self.cv_bridge.cv2_to_imgmsg(mask, encoding='mono8')  # 마스크 이미지를 mono8 (흑백)으로 변환
                self.pub_image_color_filtered.publish(img_msg)  # 원본 마스크 이미지 발행

        mask = cv2.bitwise_not(mask)  # 마스크를 반전 (노란색이 아닌 영역을 흰색으로)
        return mask  # 반전된 마스크 반환

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
        msg_state = Int8() # 메시지 객체를 미리 생성

        # 바가 3개 검출되면 (= 건널목바 3개로 판단) 아래 로직 수행
        if len(keypts) == 3:
            # 키포인트들의 무게중심 계산
            for kp in keypts:
                mean_x += kp.pt[0] / 3  # x 좌표 합산 및 평균 계산
                mean_y += kp.pt[1] / 3  # y 좌표 합산 및 평균 계산

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
            point2 = [int(keypts[idx_order[2]].pt[0]), int(keypts[idx_order[2]].pt[1] - 1)] # 가장 가까운 점
            point3 = [int(keypts[idx_order[1]].pt[0]), int(keypts[idx_order[1]].pt[1] - 1)]

            # 수평/수직/기울기 등 계산 (slope)
            dx = point3[0] - point1[0]
            dy = point3[1] - point1[1]
            if dx == 0:
                slope = float('inf')
            else:
                slope = dy / dx  # 기울기 계산

            # 기울기를 기반으로 건널목 바의 방향을 결정
            if dx == 0 or abs(slope) > 2.0:  # 수직에 가깝거나 기울기가 매우 큰 경우 (바가 세워진 상태로 간주)
                is_level_opened = True  # 건널목이 열렸다고 판단
                self.stop_bar_state = 'go'  # 정지 바 상태를 'go'로 설정
                self.get_logger().info(self.stop_bar_state)  # 상태 로그 출력
                msg_state.data = 1 # 'go' 상태를 나타내는 값
                self.pub_level_crossing_state.publish(msg_state) # 발행 추가

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
                        msg_state.data = 2 # 'slowdown' 상태를 나타내는 값
                        self.pub_level_crossing_state.publish(msg_state) # 발행 추가
                    else:
                        is_level_close = True  # 레벨에 가까움 (정지 신호)
                        self.stop_bar_state = 'stop'
                        self.get_logger().info(self.stop_bar_state)
                        msg_state.data = 4 # 'stop' 상태 (control_lane에서 정지시키는 값)
                        self.pub_level_crossing_state.publish(msg_state) # 발행 추가
            
            msg = Bool()
            msg.data = True
            self.pub_level_bar.publish(msg)

        # --- 키포인트가 1개 이하일 때(= bar가 없다고 판단): 무조건 open으로 처리 ---
        elif len(keypts) <= 1:
            is_level_opened = True
            self.stop_bar_state = 'go'
            self.get_logger().info(self.stop_bar_state)
            msg_state.data = 3 # 키포인트 부족 시 'go' 또는 기본 주행 상태
            self.pub_level_crossing_state.publish(msg_state) # 발행 추가

        # --- 결과 이미지 퍼블리시(원본에 bar 위치, 상태 시각화) ---
        if self.pub_image_type == 'compressed':
            comp_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')  # JPG로 압축하여 변환
            self.pub_image_level.publish(comp_img_msg)  # 압축 이미지 발행
        else:
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')  # BGR8 형식으로 변환
            self.pub_image_level.publish(img_msg)  # 원본 이미지 발행

        # --- 상태 반환: (바 감지/정지/열림 여부) ---
        return is_level_detected, is_level_close, is_level_opened


def main(args=None):
    """
    ROS 2 노드를 실행하는 메인 함수입니다.
    """
    rclpy.init(args=args)  # ROS 2 Python 클라이언트 라이브러리 초기화
    node = DetectLevelNode()  # DetectLevelNode 노드 객체 생성
    try:
        rclpy.spin(node)  # 노드가 종료될 때까지 메시지 콜백을 계속 처리
    except KeyboardInterrupt:  # Ctrl+C (SIGINT) 예외 처리
        node.get_logger().info('Keyboard Interrupt (SIGINT)')  # 키보드 인터럽트 로그 출력
    finally:
        node.destroy_node()  # 노드 객체 소멸 (리소스 해제)
        rclpy.shutdown()  # ROS 2 시스템 종료


if __name__ == '__main__':
    main()  # 스크립트가 직접 실행될 때 main 함수 호출