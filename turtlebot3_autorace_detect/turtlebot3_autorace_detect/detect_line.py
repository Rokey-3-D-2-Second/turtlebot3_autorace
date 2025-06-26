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
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8
import math
import random # draw_lines 함수에서 사용 (랜덤 색상)

class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        # 기존 detect_line.py의 파라미터는 차선 감지 방식이 변경되므로 대부분 제거 또는 조정 필요
        # 여기서는 Hough 변환에 직접적으로 필요하지 않은 파라미터는 제거했습니다.
        # 필요에 따라 Canny 임계값, Hough 변환 임계값 등을 ROS 파라미터로 추가할 수 있습니다.
        self.declare_parameters(
            namespace='',
            parameters=[
                ('is_detection_calibration_mode', False)
            ]
        )

        self.is_calibration_mode = self.get_parameter(
            'is_detection_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        self.sub_image_type = 'raw'         # you can choose image type 'compressed', 'raw'
        self.pub_image_type = 'compressed'  # you can choose image type 'compressed', 'raw'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1
                )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1
                )

        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1
                )
        elif self.pub_image_type == 'raw':
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1
                )

        # Hough 변환 방식에서는 이 발행자들은 직접적으로 사용되지 않을 수 있습니다.
        # calibration mode를 유지하려면 이 부분의 로직을 변경해야 합니다.
        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1
                    )
            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1
                    )

        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8, '/detect/yellow_line_reliability', 1
            )
        self.pub_white_line_reliability = self.create_publisher(
            UInt8, '/detect/white_line_reliability', 1
            )
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()

        self.counter = 1

        # 이미지 크기는 cbFindLane에서 동적으로 설정
        self.Width = 640
        self.Height = 480
        self.Offset = 420
        self.Gap = 40

        self.reliability_white_line = 100
        self.reliability_yellow_line = 100

        # Hough 변환 방식에서는 이전 프레임의 차선 정보 유지 로직이 다를 수 있음
        # 여기서는 초기값을 None으로 설정하여 첫 프레임에서 차선 감지가 실패할 경우를 대비합니다.
        self.lpos = None
        self.rpos = None

    def cbGetDetectLaneParam(self, parameters):
        # 파라미터가 변경되면 업데이트하는 로직 (Hough 변환 관련 파라미터 추가 시 여기에 반영)
        for param in parameters:
            if param.name == 'is_detection_calibration_mode':
                self.is_calibration_mode = param.value
            # 다른 Hough 관련 파라미터 (Canny 임계값 등)가 있다면 여기에 추가
            return SetParametersResult(successful=True)

    # draw lines - Hough 변환 코드에서 가져옴
    def draw_lines(self, img, lines):
        # global Offset
        for line in lines:
            x1, y1, x2, y2 = line[0]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = cv2.line(img, (x1, y1 + self.Offset), (x2, y2 + self.Offset), color, 2)
        return img

    # draw rectangle - Hough 변환 코드에서 가져옴
    def draw_rectangle(self, img, lpos, rpos, offset=0):
        center = (lpos + rpos) / 2
        cv2.rectangle(img, (lpos - 5, 15 + offset), (lpos + 5, 25 + offset), (0, 255, 0), 2)
        cv2.rectangle(img, (rpos - 5, 15 + offset), (rpos + 5, 25 + offset), (0, 255, 0), 2)
        cv2.rectangle(img, (int(center) - 5, 15 + offset), (int(center) + 5, 25 + offset), (0, 255, 0), 2)
        cv2.rectangle(img, (int(self.Width / 2) - 5, 15 + offset), (int(self.Width / 2) + 5, 25 + offset), (0, 0, 255), 2)
        return img

    # left lines, right lines - Hough 변환 코드에서 가져옴
    def divide_left_right(self, lines):
        # global Width
        low_slope_threshold = 0.1 # 0으로 설정하면 거의 수평선도 포함될 수 있음. 약간의 임계값 설정
        high_slope_threshold = 10 # 너무 가파른 선 제외 (노이즈)

        slopes = []
        new_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                slope = 999.0  # 수직선에 가까운 경우 매우 큰 기울기 할당
            else:
                slope = float(y2 - y1) / float(x2 - x1)
            if (abs(slope) > low_slope_threshold) and (abs(slope) < high_slope_threshold):
                slopes.append(slope)
                new_lines.append(line[0])

        left_lines = []
        right_lines = []

        for j in range(len(slopes)):
            Line = new_lines[j]
            slope = slopes[j]

            x1, y1, x2, y2 = Line

            # y축이 아래로 증가하는 OpenCV 좌표계에서 기울기 판단
            # 왼쪽 차선: 음수 기울기, 이미지 왼쪽 절반
            if (slope < 0) and (x2 < self.Width / 2): # x2를 사용하여 선의 끝점이 왼쪽 절반에 있는지 확인 (좀 더 견고)
                left_lines.append([Line.tolist()])
            # 오른쪽 차선: 양수 기울기, 이미지 오른쪽 절반
            elif (slope > 0) and (x1 > self.Width / 2): # x1을 사용하여 선의 시작점이 오른쪽 절반에 있는지 확인
                right_lines.append([Line.tolist()])

        return left_lines, right_lines

    # get average m, b of lines - Hough 변환 코드에서 가져옴
    def get_line_params(self, lines):
        x_sum = 0.0
        y_sum = 0.0
        m_sum = 0.0

        size = len(lines)
        if size == 0:
            return 0.0, 0.0 # float으로 반환

        for line in lines:
            x1, y1, x2, y2 = line[0]

            x_sum += x1 + x2
            y_sum += y1 + y2
            if x2 - x1 == 0: # 분모가 0이 되는 경우 방지
                m_sum += 999.0 if y2 > y1 else -999.0
            else:
                m_sum += float(y2 - y1) / float(x2 - x1)

        x_avg = x_sum / (size * 2)
        y_avg = y_sum / (size * 2)
        m = m_sum / size

        if m == 0: # 기울기가 0인 경우 b 계산 시 오류 방지
            b = y_avg # 거의 수평선이므로 y절편은 y_avg가 됨
        else:
            b = y_avg - m * x_avg

        return m, b

    # get lpos, rpos - Hough 변환 코드에서 가져옴
    def get_line_pos(self, img, lines, left=False, right=False):
        # global Width, Height
        # global Offset, Gap

        m, b = self.get_line_params(lines)

        if m == 0.0 and b == 0.0: # 선을 못 찾으면
            if left:
                pos = 0
            elif right:
                pos = self.Width
        else:
            y = self.Gap / 2
            # y = mx + b  =>  x = (y - b) / m
            if m == 0: # 기울기가 0인 경우 수직선이므로 x는 일정
                pos = int(b) # 사실 이 경우 y=b가 아니라 x=b (y-intercept가 아니라 x-intercept)
            else:
                pos = int((y - b) / m)

            # 차선이 그려지는 영역 (시각화를 위해)
            # OpenCV 좌표계에서 y1은 이미지 하단, y2는 이미지 중간 근처
            y1_line = self.Height
            y2_line = int(self.Height / 2) # 이미지 중간

            # x = (y - b_actual) / m
            b_actual = b + self.Offset # ROI offset을 고려한 실제 y절편

            if m == 0:
                x1_line = int(pos) # 수직선
                x2_line = int(pos)
            else:
                x1_line = int((y1_line - b_actual) / m)
                x2_line = int((y2_line - b_actual) / m)

            # 라인 그리기 (디버깅용)
            # cv2.line(img, (x1_line, y1_line), (x2_line, y2_line), (255, 0, 0), 3)

        return img, pos

    # process_image - Hough 변환 코드에서 가져와 cbFindLane에 통합
    def process_image_hough(self, frame):
        # global Width, Offset, Gap

        # GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # BLUR
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # CANNY EDGE
        low_threshold = 60
        high_threshold = 70
        edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

        # ROI - 현재 ROI는 원본 이미지에 대한 상대적인 위치로 가정 (높이 480 기준)
        # 이미지의 크기가 변경될 경우 이 부분도 동적으로 조정되어야 합니다.
        # 현재는 self.Height와 self.Width를 사용하여 동적으로 조정됩니다.
        roi = edge_img[self.Offset : self.Offset + self.Gap, 0 : self.Width]

        # HoughLinesP
        # param 4: threshold, param 5: minLineLength, param 6: maxLineGap
        all_lines = cv2.HoughLinesP(roi, 1, math.pi / 180, 30, minLineLength=30, maxLineGap=10)

        # Divide left, right lines
        if all_lines is None:
            # 선을 찾지 못했을 경우, 이전 lpos, rpos를 사용하거나 기본값을 반환
            # 여기서는 차선 없음을 알리기 위해 0, Width를 반환
            return 0, self.Width, None # 차선 시각화 정보도 함께 반환

        left_lines, right_lines = self.divide_left_right(all_lines)

        # Get center of lines and draw lines
        # 원본 frame에 선을 그리기 위해 frame을 인자로 전달
        # lpos, rpos는 ROI 내에서의 x 좌표
        draw_frame_left, lpos = self.get_line_pos(frame.copy(), left_lines, left=True) # 복사본에 그려야 원본에 영향 안 줌
        draw_frame_right, rpos = self.get_line_pos(frame.copy(), right_lines, right=True)

        # 병합된 시각화 이미지 생성 (옵션)
        # 모든 HoughLineP로 찾은 선을 알록달록하게 그리기
        visualized_hough_lines = frame.copy()
        visualized_hough_lines = self.draw_lines(visualized_hough_lines, all_lines)
        # 차선 평균 선 (파란색) 그리기 (get_line_pos 내부에서 그려지도록 함)
        # cv2.line(visualized_hough_lines, (int(x1_left), self.Height), (int(x2_left), self.Height//2), (255, 0,0), 3) # 이 부분은 get_line_pos에서 처리
        # cv2.line(visualized_hough_lines, (int(x1_right), self.Height), (int(x2_right), self.Height//2), (255, 0,0), 3)

        # draw rectangle
        final_frame_with_rectangles = self.draw_rectangle(frame.copy(), lpos, rpos, offset=self.Offset)
        # 화면 중앙에 수평선 그리기
        cv2.line(final_frame_with_rectangles, (230, int(self.Height/2 - 5)), (410, int(self.Height/2 - 5)), (255, 255, 255), 2)


        return lpos, rpos, final_frame_with_rectangles


    def draw_steer(self, image, steer_angle) :
        # steer_arrow.png 파일이 프로젝트 내에 존재해야 합니다.
        try:
            arrow_pic = cv2.imread("steer_arrow.png", cv2.IMREAD_COLOR)
            if arrow_pic is None:
                self.get_logger().warn("steer_arrow.png 이미지를 찾을 수 없습니다. 시각화가 제한됩니다.")
                return

            origin_Height = arrow_pic.shape[0]
            origin_Width = arrow_pic.shape[1]
            steer_wheel_center = origin_Height * 0.74
            arrow_Height = int(self.Height / 2) # 이미지 전체 높이의 절반으로 조정
            arrow_Width = int((arrow_Height * 462) / 728)

            matrix = cv2.getRotationMatrix2D((origin_Width / 2, steer_wheel_center), (steer_angle) * 2.5, 0.7)

            arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width + 60, origin_Height))
            arrow_pic = cv2.resize(arrow_pic, dsize = (arrow_Width, arrow_Height), interpolation = cv2.INTER_AREA)

            gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

            # ROI 계산을 이미지 크기에 맞춰 동적으로
            arrow_roi_y_start = self.Height - arrow_Height
            arrow_roi_y_end = self.Height
            arrow_roi_x_start = int(self.Width / 2 - arrow_Width / 2)
            arrow_roi_x_end = int(self.Width / 2 + arrow_Width / 2)

            # 이미지 범위 벗어남 방지
            arrow_roi_y_start = max(0, arrow_roi_y_start)
            arrow_roi_x_start = max(0, arrow_roi_x_start)
            arrow_roi_y_end = min(self.Height, arrow_roi_y_end)
            arrow_roi_x_end = min(self.Width, arrow_roi_x_end)

            # ROI 크기 확인 및 조정
            if arrow_roi_y_end - arrow_roi_y_start != arrow_pic.shape[0] or \
               arrow_roi_x_end - arrow_roi_x_start != arrow_pic.shape[1]:
                # ROI 크기가 arrow_pic과 다르면 resize
                arrow_pic_resized = cv2.resize(arrow_pic, (arrow_roi_x_end - arrow_roi_x_start, arrow_roi_y_end - arrow_roi_y_start))
                mask_resized = cv2.resize(mask, (arrow_roi_x_end - arrow_roi_x_start, arrow_roi_y_end - arrow_roi_y_start))
            else:
                arrow_pic_resized = arrow_pic
                mask_resized = mask


            arrow_roi = image[arrow_roi_y_start:arrow_roi_y_end, arrow_roi_x_start:arrow_roi_x_end]

            # add 함수 대신 copyTo를 사용하거나 직접 마스킹하여 오버레이
            # alpha blending for better integration
            for c in range(0, 3):
                arrow_roi[:, :, c] = np.where(mask_resized == 0,
                                              arrow_pic_resized[:, :, c],
                                              arrow_roi[:, :, c])

            image[arrow_roi_y_start:arrow_roi_y_end, arrow_roi_x_start:arrow_roi_x_end] = arrow_roi

            cv2.imshow('steer', image)
        except Exception as e:
            self.get_logger().error(f"Error drawing steer image: {e}")


    def cbFindLane(self, image_msg):
        if self.counter % 1 != 0: # 프레임 스킵 로직
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        if cv_image is None:
            self.get_logger().warn("Failed to decode image or image is empty.")
            return

        # 이미지 크기 동적 설정 (처음 한 번만)
        if self.Width != cv_image.shape[1] or self.Height != cv_image.shape[0]:
            self.Width = cv_image.shape[1]
            self.Height = cv_image.shape[0]
            # ROI Offset과 Gap도 이미지 Height에 비례하여 조정될 수 있도록 로직 추가
            # 예: self.Offset = int(self.Height * 0.875) # 480 -> 420 (420/480 = 0.875)
            # self.Gap = int(self.Height * 0.0833) # 480 -> 40 (40/480 = 0.0833)
            # 여기서는 일단 고정 값 유지. 필요시 주석 해제 및 비율 계산 적용.

        # Hough 변환 기반 차선 감지 로직 적용
        # lpos, rpos는 ROI 내에서의 x 좌표
        lpos, rpos, processed_frame = self.process_image_hough(cv_image.copy())

        self.lpos = lpos
        self.rpos = rpos

        # 차선 신뢰도 계산 (간단한 예시)
        # Hough 변환으로 차선이 감지되면 신뢰도 높음, 아니면 낮음
        if lpos != 0: # 왼쪽 차선이 감지되면 (초기 0이 아닌 경우)
            self.reliability_yellow_line = 100
        else:
            self.reliability_yellow_line = 0

        if rpos != self.Width: # 오른쪽 차선이 감지되면 (초기 Width가 아닌 경우)
            self.reliability_white_line = 100
        else:
            self.reliability_white_line = 0

        msg_yellow_line_reliability = UInt8()
        msg_yellow_line_reliability.data = self.reliability_yellow_line
        self.pub_yellow_line_reliability.publish(msg_yellow_line_reliability)

        msg_white_line_reliability = UInt8()
        msg_white_line_reliability.data = self.reliability_white_line
        self.pub_white_line_reliability.publish(msg_white_line_reliability)

        # 차선 상태 결정 및 발행
        lane_state = UInt8()
        if lpos != 0 and rpos != self.Width: # 양쪽 차선 감지
            lane_state.data = 2
            centerx = (lpos + rpos) / 2
        elif lpos != 0 and rpos == self.Width: # 왼쪽 차선만 감지
            lane_state.data = 1
            centerx = lpos + int(self.Width / 4) # 대략적인 중앙 추정 (오른쪽으로 치우쳐서 보정)
        elif lpos == 0 and rpos != self.Width: # 오른쪽 차선만 감지
            lane_state.data = 3
            centerx = rpos - int(self.Width / 4) # 대략적인 중앙 추정 (왼쪽으로 치우쳐서 보정)
        else: # 차선 없음
            lane_state.data = 0
            centerx = self.Width / 2 # 중앙으로 가정

        self.pub_lane_state.publish(lane_state)
        self.get_logger().info(f'Lane state: {lane_state.data}, lpos: {lpos}, rpos: {rpos}')

        # 조향각 계산 및 시각화
        # 여기서 steer_angle을 계산할 때, lpos와 rpos는 ROI 내의 좌표임을 기억해야 합니다.
        # 따라서 centerx는 전체 이미지의 중앙에 대한 상대적인 위치로 변환되어야 합니다.
        # hough 변환 코드의 '320 - center' 로직은 Width 640의 중앙 320을 기준으로 한 것입니다.
        # 현재 centerx는 ROI 내의 중심이므로, 전체 이미지의 중앙을 기준으로 다시 계산해야 합니다.
        # lpos, rpos는 Offset만큼 아래로 내려간 ROI에서의 X좌표이므로, 실제 이미지의 X좌표로 사용해도 무방합니다.
        
        # '320 - center' 로직을 유지하기 위해, image의 중앙을 320으로 고정하거나,
        # self.Width / 2를 사용하여 동적으로 계산할 수 있습니다.
        # 여기서는 self.Width / 2를 사용하여 일반화합니다.
        
        # lpos, rpos는 ROI 안에서의 x 좌표이므로, 이를 활용하여 전체 이미지에서의 중심을 계산.
        # `lpos`와 `rpos`가 ROI에서의 x좌표이므로, 이들의 중간값 `centerx`도 ROI에서의 x좌표입니다.
        # 이를 `self.Width / 2` (전체 이미지의 중앙 x좌표)와 비교하여 조향각을 계산합니다.
        
        # `hough_track.avi` 예제에서 사용된 `320`은 `Width/2` (640/2)와 같습니다.
        # 따라서 `angle = (self.Width / 2) - centerx` 로 변경합니다.
        angle = (self.Width / 2) - centerx
        steer_angle = angle * 0.4 # hough_track.avi 예제 비율 유지

        self.draw_steer(processed_frame.copy(), steer_angle) # processed_frame에 조향각 시각화

        # 최종 이미지 발행
        if self.pub_image_type == 'compressed':
            msg_desired_center = Float64()
            msg_desired_center.data = centerx
            self.pub_lane.publish(msg_desired_center)
            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(processed_frame, 'jpeg'))
        elif self.pub_image_type == 'raw':
            msg_desired_center = Float64()
            msg_desired_center.data = centerx
            self.pub_lane.publish(msg_desired_center)
            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(processed_frame, 'bgr8'))

        # calibration mode 시 추가 이미지 발행 (기존 로직 유지)
        if self.is_calibration_mode:
            # Hough 변환 방식에서는 white_lane, yellow_lane 마스크를 직접적으로 생성하지 않으므로,
            # 이 부분을 비활성화하거나, Hough 변환 과정에서 생성된 중간 이미지를 발행하도록 수정해야 합니다.
            # 여기서는 일단 비활성화 (주석 처리) 합니다.
            pass
            # if self.pub_image_type == 'compressed':
            #     self.pub_image_white_lane.publish(
            #         self.cvBridge.cv2_to_compressed_imgmsg(np.zeros((self.Height, self.Width, 1), dtype=np.uint8), 'jpg')
            #         )
            #     self.pub_image_yellow_lane.publish(
            #         self.cvBridge.cv2_to_compressed_imgmsg(np.zeros((self.Height, self.Width, 1), dtype=np.uint8), 'jpg')
            #         )
            # elif self.pub_image_type == 'raw':
            #     self.pub_image_white_lane.publish(
            #         self.cvBridge.cv2_to_imgmsg(np.zeros((self.Height, self.Width, 1), dtype=np.uint8), 'bgr8')
            #         )
            #     self.pub_image_yellow_lane.publish(
            #         self.cvBridge.cv2_to_imgmsg(np.zeros((self.Height, self.Width, 1), dtype=np.uint8), 'bgr8')
            #         )

        # `cv2.waitKey(1)`은 ROS 노드에서는 일반적으로 사용하지 않지만,
        # `cv2.imshow`와 함께 디버깅 목적으로 사용할 수 있습니다.
        # ROS spin 루프가 이미 메시지 처리를 담당하므로 `cv2.waitKey(1)`이 필요 없을 수도 있습니다.
        # 그러나 창이 업데이트되려면 필요합니다.
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()