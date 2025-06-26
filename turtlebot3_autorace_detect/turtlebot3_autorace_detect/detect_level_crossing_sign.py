#!/usr/bin/env python3

# 이 코드는 ROS2 노드에서 STOP 교통 표지판을 인식하고, 인식되었을 때 토픽을 통해 알리는 기능을 수행함.
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Leon Jung, Gilbert, Ashe Kim, Jun

from enum import Enum
import os

import cv2  # OpenCV 사용
from cv_bridge import CvBridge  # ROS 이미지 메시지를 OpenCV 이미지로 변환
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from std_msgs.msg import Int8 # Int8 메시지 임포트 (control_lane과 통신용)


class DetectSign(Node):
    """
    ROS 2 노드: 교통 표지판을 감지하고 관련 정보를 발행합니다.
    주로 '정지' 표지판 감지에 중점을 둡니다.
    """

    def __init__(self):
        """
        DetectSign 클래스의 생성자입니다.
        ROS 2 노드를 초기화하고, 이미지 구독 및 발행, SIFT 특징점 감지기 등을 설정합니다.
        """
        super().__init__('detect_sign')

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        # 이미지 입력 구독자 설정
        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage,
                '/detect/image_input/compressed',
                self.cbFindTrafficSign,
                10
            )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image,
                '/detect/image_input',
                self.cbFindTrafficSign,
                10
            )

        # 기존 교통 표지판 감지 결과 발행자 (UInt8)
        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        
        # 건널목 상태 퍼블리셔 추가 (control_lane.py 와 연동)
        self.level_crossing_state_publisher = self.create_publisher(
            Int8, '/level_crossing_state', 10)

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign = self.create_publisher(
                CompressedImage,
                '/detect/image_output/compressed', 10
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign = self.create_publisher(
                Image, '/detect/image_output', 10
            )

        self.cvBridge = CvBridge()  # ROS 이미지 ↔ OpenCV 변환용 브릿지

        # 표지판 종류 정의 (지금은 stop 하나만 사용)
        self.TrafficSign = Enum('TrafficSign', 'stop')

        self.counter = 1  # 프레임 수 카운터 (프레임 드랍용)

        self.fnPreproc()  # SIFT 및 학습 이미지 전처리

        self.get_logger().info('DetectSign Node Initialized')

    def fnPreproc(self):
        """
        SIFT 특징점 감지기 초기화 및 참조 교통 표지판 이미지 로드 및 특징점 계산
        """
        self.sift = cv2.SIFT_create()

        # 이미지 경로 설정 및 stop 표지판 이미지 불러오기
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        dir_path = os.path.join(dir_path, 'image')

        self.img_stop = cv2.imread(dir_path + '/stop3.png', 0)
        
        self.kp_stop, self.des_stop = self.sift.detectAndCompute(self.img_stop, None)

        # FLANN 매칭 설정
        FLANN_INDEX_KDTREE = 0
        index_params = {
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5
        }

        search_params = {
            'checks': 50
        }

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        """
        두 배열 간의 평균 제곱 오차(MSE)를 계산합니다.
        주로 특징점 매칭 후 변환된 점들의 오차를 측정하는 데 사용됩니다.
        """
        squared_diff = (arr1 - arr2) ** 2
        total_sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1]
        err = total_sum / num_all
        return err

    def cbFindTrafficSign(self, image_msg):
        """
        이미지 메시지를 구독하고 교통 표지판을 감지하는 콜백 함수입니다.
        SIFT를 사용하여 이미지에서 '정지' 표지판을 찾습니다.
        """
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        # 메시지를 OpenCV 이미지로 변환
        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
        
        # -------- [1] ROI: 위쪽 절반만 사용 --------
        h, w, _ = cv_image_input.shape
        # roi_y = 0               # 시작 y좌표
        # roi_h = int(h * 0.4)    # 상단 40%
        # cv_image_roi = cv_image_input[roi_y:roi_y+roi_h, :]
        roi_w = int(w * 0.5)  # ROI 너비 (50%)
        roi_h = int(h * 0.6)  # ROI 높이 (60%)
        roi_x = w - roi_w     # 오른쪽에서 시작 (전체 가로 - ROI 너비)
        roi_y = 0             # 맨 위에서 시
        cv_image_roi = cv_image_input[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        MIN_MATCH_COUNT = 5
        MIN_MSE_DECISION = 50000

        # find the keypoints and descriptors with SIFT
        # kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)
        kp1, des1 = self.sift.detectAndCompute(cv_image_roi, None)

        # 건널목 상태 메시지 초기화 (기본: 신호 없음)
        level_crossing_msg = Int8()
        level_crossing_msg.data = 5  # 3: 신호 없음/기본 주행

        image_out_num = 1  # 기본 출력 이미지 설정 (인식 못함)

        # des1이 None이 아닌 경우에만 매칭을 시도
        if des1 is not None and len(des1) > 0:
            matches_stop = self.flann.knnMatch(des1, self.des_stop, k=2)

            good_stop = []
            for m, n in matches_stop:
                if m.distance < 0.7 * n.distance:
                    good_stop.append(m)

            if len(good_stop) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_stop]).reshape(-1, 1, 2)
                dst_pts = np.float32([
                    self.kp_stop[m.trainIdx].pt for m in good_stop
                ]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matches_stop = mask.ravel().tolist()

                mse = self.fnCalcMSE(src_pts, dst_pts)
                if mse < MIN_MSE_DECISION:
                    # '정지' 표지판 감지!
                    msg_sign = UInt8()
                    msg_sign.data = self.TrafficSign.stop.value
                    self.pub_traffic_sign.publish(msg_sign)
                    self.get_logger().info('Stop sign detected!')
                    
                    # control_lane.py로 건널목 정지 상태 (4) 발행
                    level_crossing_msg.data = 4 # 4: 건널목 정지
                    self.get_logger().info('Publishing level crossing state: 4 (STOP)')
                    image_out_num = 2
                else:
                    self.get_logger().info('Stop sign detected, but MSE too high. Publishing default state.')
            else:
                matches_stop = None
                self.get_logger().info('Not enough matches for stop sign. Publishing default state.')
        else:
            self.get_logger().info('No descriptors found in the input image. Publishing default state.')
        
        # 건널목 상태 메시지 발행
        self.level_crossing_state_publisher.publish(level_crossing_msg)

        # 이미지 퍼블리시 (인식 여부에 따라 다르게)
        if image_out_num == 1:
            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(cv_image_input, 'jpg')
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(cv_image_input, 'bgr8')
                )

        elif image_out_num == 2:
            # 매칭 시각화
            draw_params2 = {
                'matchColor': (0, 0, 255),
                'singlePointColor': None,
                'matchesMask': matches_stop,
                'flags': 2
            }
            # kp1 좌표를 원본 이미지 기준으로 변환
            kp1_on_input = [
                cv2.KeyPoint(
                    kp.pt[0] + roi_x,  # x좌표 보정
                    kp.pt[1] + roi_y,  # y좌표 보정
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id
                ) for kp in kp1
            ]
            final_stop = cv2.drawMatches(
                cv_image_input,
                kp1_on_input,
                self.img_stop,
                self.kp_stop,
                good_stop,
                None,
                **draw_params2
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(final_stop, 'jpg')
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(final_stop, 'bgr8')
                )


def main(args=None):
    """
    ROS 2 노드를 실행하는 메인 함수입니다.
    """
    rclpy.init(args=args)
    node = DetectSign()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# 이 노드 파일이 직접 실행되었을 때 main() 실행
if __name__ == '__main__':
    main()