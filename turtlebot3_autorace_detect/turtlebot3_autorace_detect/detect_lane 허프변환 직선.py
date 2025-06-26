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
# Authors:
#   - Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
#   - Special Thanks : Roger Sacchelli

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8
import math
import random


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        # 허프 변환 관련 파라미터만 선언
        self.declare_parameters(
            namespace='',
            parameters=[
                # 이전에 성공했던 hough_transform.py의 값들을 초기값으로 설정
                ('hough.min_line_length', 10),  # hough_transform.py에서 20
                ('hough.max_line_gap', 5),      # hough_transform.py에서 2
                ('hough.threshold', 3),        # hough_transform.py에서 10
                # 이전에 성공했던 hough_transform.py의 값들을 초기값으로 설정
                ('canny.low_threshold', 60),    # hough_transform.py에서 50
                ('canny.high_threshold', 160),  # hough_transform.py에서 200
                ('blur.kernel_size', 7),
                ('roi.offset_y', 350), # ROI 시작 Y 좌표 (조감도 이미지에 맞게 넓게 설정)
                ('roi.height', 150),   # ROI 높이 (조감도 이미지에 맞게 넓게 설정)
                ('lane_separation.min_slope', 0),
                ('lane_separation.max_slope', 10),
                ('lane_separation.x_offset', 90), # 좌우 차선 분류 시 이미지 중앙에서 벗어나는 정도
                ('is_detection_calibration_mode', False)
            ]
        )

        # 파라미터 로드
        self.hough_min_line_length = self.get_parameter('hough.min_line_length').get_parameter_value().integer_value
        self.hough_max_line_gap = self.get_parameter('hough.max_line_gap').get_parameter_value().integer_value
        self.hough_threshold = self.get_parameter('hough.threshold').get_parameter_value().integer_value
        self.canny_low_threshold = self.get_parameter('canny.low_threshold').get_parameter_value().integer_value
        self.canny_high_threshold = self.get_parameter('canny.high_threshold').get_parameter_value().integer_value
        self.blur_kernel_size = self.get_parameter('blur.kernel_size').get_parameter_value().integer_value
        self.roi_offset_y = self.get_parameter('roi.offset_y').get_parameter_value().integer_value
        self.roi_height = self.get_parameter('roi.height').get_parameter_value().integer_value
        self.lane_sep_min_slope = self.get_parameter('lane_separation.min_slope').get_parameter_value().integer_value
        self.lane_sep_max_slope = self.get_parameter('lane_separation.max_slope').get_parameter_value().integer_value
        self.lane_sep_x_offset = self.get_parameter('lane_separation.x_offset').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter('is_detection_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        # 원본 이미지 구독
        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1
                )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1
                )

        # 최종 처리 이미지 발행
        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1
                )
        elif self.pub_image_type == 'raw':
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1
                )

        # Canny Edge 이미지 발행 (디버깅용)
        if self.pub_image_type == 'compressed':
            self.pub_image_canny_edge = self.create_publisher(
                CompressedImage, '/detect/image_canny_edge/compressed', 1
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_canny_edge = self.create_publisher(
                Image, '/detect/image_canny_edge', 1
            )

        # 차선 중앙 위치 발행 (Float64)
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)

        # 차선 상태 발행 (UInt8)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()
        self.counter = 1

    def cbGetDetectLaneParam(self, parameters):
        for param in parameters:
            if param.name == 'hough.min_line_length':
                self.hough_min_line_length = param.value
            elif param.name == 'hough.max_line_gap':
                self.hough_max_line_gap = param.value
            elif param.name == 'hough.threshold':
                self.hough_threshold = param.value
            elif param.name == 'canny.low_threshold':
                self.canny_low_threshold = param.value
            elif param.name == 'canny.high_threshold':
                self.canny_high_threshold = param.value
            elif param.name == 'blur.kernel_size':
                self.blur_kernel_size = param.value
            elif param.name == 'roi.offset_y':
                self.roi_offset_y = param.value
            elif param.name == 'roi.height':
                self.roi_height = param.value
            elif param.name == 'lane_separation.min_slope':
                self.lane_sep_min_slope = param.value
            elif param.name == 'lane_separation.max_slope':
                self.lane_sep_max_slope = param.value
            elif param.name == 'lane_separation.x_offset':
                self.lane_sep_x_offset = param.value
            return SetParametersResult(successful=True)

    def draw_lines(self, img, lines, offset_y):
        """감지된 허프 선분들을 이미지에 그립니다."""
        for line in lines:
            x1, y1, x2, y2 = line[0]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # ROI offset_y를 더하여 원본 이미지 좌표에 맞게 그립니다.
            img = cv2.line(img, (x1, y1 + offset_y), (x2, y2 + offset_y), color, 2)
        return img

    def divide_left_right(self, lines, width, min_slope_th, max_slope_th, x_offset):
        """선분들을 왼쪽 차선과 오른쪽 차선으로 분류합니다."""
        slopes = []
        new_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                slope = 0
            else:
                slope = float(y2 - y1) / float(x2 - x1)
            if (abs(slope) > min_slope_th) and (abs(slope) < max_slope_th):
                slopes.append(slope)
                new_lines.append(line[0])

        left_lines = []
        right_lines = []
        for j in range(len(slopes)):
            line = new_lines[j]
            slope = slopes[j]
            x1, y1, x2, y2 = line
            # 기울기 음수 (왼쪽 위-오른쪽 아래) 및 이미지 왼쪽 절반에 위치
            if (slope < 0) and (x2 < width/2 - x_offset):
                left_lines.append([line.tolist()])
            # 기울기 양수 (왼쪽 아래-오른쪽 위) 및 이미지 오른쪽 절반에 위치
            elif (slope > 0) and (x1 > width/2 + x_offset):
                right_lines.append([line.tolist()])
        return left_lines, right_lines

    def get_line_params(self, lines):
        """선분들의 평균 기울기와 y절편을 계산합니다."""
        x_sum = 0.0
        y_sum = 0.0
        m_sum = 0.0
        size = len(lines)
        if size == 0:
            return 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / float(x2 - x1)
        x_avg = x_sum / (size * 2)
        y_avg = y_sum / (size * 2)
        m = m_sum / size
        b = y_avg - m * x_avg
        return m, b

    def cbFindLane(self, image_msg):
        if self.counter % 0.5 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        current_image_height, current_image_width = cv_image.shape[:2]

        # 1. Grayscale 변환
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian Blur 적용
        blur_gray = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # 3. Canny Edge 검출
        edge_img = cv2.Canny(np.uint8(blur_gray), self.canny_low_threshold, self.canny_high_threshold)

        # Canny Edge 이미지 발행 (디버깅용)
        if self.pub_image_type == 'compressed':
            self.pub_image_canny_edge.publish(self.cvBridge.cv2_to_compressed_imgmsg(edge_img, '.png'))
        elif self.pub_image_type == 'raw':
            self.pub_image_canny_edge.publish(self.cvBridge.cv2_to_imgmsg(edge_img, 'mono8'))

        # 4. ROI (Region of Interest) 설정
        # 엣지 이미지에서 ROI 영역만 추출 (Y축 offset_y에서 높이 roi_height만큼)
        roi = edge_img[self.roi_offset_y : self.roi_offset_y + self.roi_height, 0 : current_image_width]

        # 5. Hough Line Transform (P: Probabilistic)
        # ROI에서 직선 성분 검출
        all_lines_hough = cv2.HoughLinesP(
            roi,
            1, # rho: 픽셀 해상도 (1픽셀 단위)
            math.pi/180, # theta: 라디안 해상도 (1도 단위)
            self.hough_threshold, # threshold: 직선을 구성하는 최소 교차점 개수
            self.hough_min_line_length, # minLineLength: 직선으로 인식할 최소 길이
            self.hough_max_line_gap # maxLineGap: 같은 직선으로 간주할 최대 간격
        )

        centerx_to_publish = current_image_width / 2.0 # 기본값 설정
        lane_state_data = 0 # 0: none, 1: left, 2: both, 3: right

        # 시각화용 이미지 복사
        cv_image_with_hough_lines = cv_image.copy()

        # 허프 변환으로 선분이 감지된 경우
        if all_lines_hough is not None:
            # 6. 좌우 차선 분류
            left_lines, right_lines = self.divide_left_right(
                all_lines_hough, current_image_width,
                self.lane_sep_min_slope, self.lane_sep_max_slope, self.lane_sep_x_offset
            )

            # 허프 라인 그리기
            cv_image_with_hough_lines = self.draw_lines(cv_image_with_hough_lines, left_lines, self.roi_offset_y)
            cv_image_with_hough_lines = self.draw_lines(cv_image_with_hough_lines, right_lines, self.roi_offset_y)

            # 좌우 차선의 평균 기울기 및 y절편 계산
            m_left, b_left = self.get_line_params(left_lines)
            m_right, b_right = self.get_line_params(right_lines)

            # 차선 존재 여부 판단
            is_left_line_exist = (m_left != 0 or b_left != 0)
            is_right_line_exist = (m_right != 0 or b_right != 0)

            # 차선 상태 및 centerx 계산
            y_calc_point = self.roi_offset_y + self.roi_height # ROI의 가장 아래쪽 Y 좌표를 기준으로 centerx 계산

            if is_left_line_exist and is_right_line_exist:
                x_left_at_point = (y_calc_point - b_left) / m_left if m_left != 0 else 0
                x_right_at_point = (y_calc_point - b_right) / m_right if m_right != 0 else current_image_width
                centerx_to_publish = (x_left_at_point + x_right_at_point) / 2.0
                lane_state_data = 2

            elif is_left_line_exist and not is_right_line_exist:
                x_left_at_point = (y_calc_point - b_left) / m_left if m_left != 0 else 0
                centerx_to_publish = x_left_at_point + 280 # 기존 코드의 280 offset 사용
                lane_state_data = 1

            elif not is_left_line_exist and is_right_line_exist:
                x_right_at_point = (y_calc_point - b_right) / m_right if m_right != 0 else current_image_width
                centerx_to_publish = x_right_at_point - 280 # 기존 코드의 280 offset 사용
                lane_state_data = 3
            else:
                # all_lines_hough는 None이 아니지만, 유효한 좌우 차선이 분류되지 않은 경우
                lane_state_data = 0 # No lane detected

        # === ROI 시각화 추가 (이 부분에 추가) ===
        # ROI 영역을 cv_image_with_hough_lines에 녹색 사각형으로 그립니다.
        cv2.rectangle(
            cv_image_with_hough_lines,
            (0, self.roi_offset_y),
            (current_image_width, self.roi_offset_y + self.roi_height),
            (0, 255, 0), # Green color (BGR)
            2 # Thickness
        )
        # ====================================

        # 최종 이미지 (허프 라인 + ROI 표시)를 발행 (all_lines_hough 여부와 상관없이 항상 발행)
        if self.pub_image_type == 'compressed':
            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(cv_image_with_hough_lines, 'jpg'))
        elif self.pub_image_type == 'raw':
            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(cv_image_with_hough_lines, 'bgr8'))

        # /detect/lane 토픽 발행 (Float64)
        if centerx_to_publish is not None:
            msg_desired_center = Float64()
            msg_desired_center.data = centerx_to_publish
            self.pub_lane.publish(msg_desired_center)

        # lane_state 토픽 발행 (UInt8)
        msg_lane_state = UInt8()
        msg_lane_state.data = lane_state_data
        self.pub_lane_state.publish(msg_lane_state)
        self.get_logger().info(f'Lane state: {lane_state_data}, CenterX: {centerx_to_publish}')

# main 함수는 변경 없음
def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()