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
# Author: Leon Jung, Gilbert, Ashe Kimm Jun

from enum import Enum
import os

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8


class DetectSign(Node):

    def __init__(self):
        super().__init__('detect_sign')

        self.sub_image_type = 'raw'         # you can choose image type 'compressed', 'raw'
        self.pub_image_type = 'compressed'  # you can choose image type 'compressed', 'raw'

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

        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign = self.create_publisher(
                CompressedImage,
                '/detect/image_output/compressed', 10
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign = self.create_publisher(
                Image, '/detect/image_output', 10
            )

        self.cvBridge = CvBridge()
        self.TrafficSign = Enum('TrafficSign', 'construction')
        self.counter = 1

        self.fnPreproc()

        self.get_logger().info('DetectSign Node Initialized')

    def fnPreproc(self):
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()

        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        dir_path = os.path.join(dir_path, 'image')

        self.img_construction = cv2.imread(dir_path + '/construction.png', 0)
        self.kp_construction, self.des_construction = self.sift.detectAndCompute(
            self.img_construction, None
        )

        FLANN_INDEX_KDTREE = 0
        index_params = {
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5
        }

        search_params = {
            'checks': 50
        }

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # # 1. 이미지 불러오기 (Gray가 아니라 BGR로 읽는게 전처리에 더 좋음)
        # img_bgr = cv2.imread(dir_path + '/construction.png', cv2.IMREAD_COLOR)

        # # 2. [ROI 적용] - 입력과 동일한 ROI 설정 (예시)
        # # h, w, _ = img_bgr.shape
        # # roi_x = int(w * 0.2)
        # # roi_y = int(h * 0.0)
        # # roi_w = int(w * 0.6)
        # # roi_h = int(h * 0.6)
        # # img_roi = img_bgr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        # img_roi = img_bgr

        # # 3. [색상공간 보정] LAB 변환 + CLAHE (명암 대비 향상)
        # lab = cv2.cvtColor(img_roi, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # l = clahe.apply(l)
        # lab = cv2.merge((l, a, b))
        # img_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # # 4. [블러] Gaussian blur 적용
        # img_preproc = cv2.GaussianBlur(img_lab, (5, 5), 0)

        # # 5. [Gray 변환] SIFT는 gray에서 작동
        # img_gray = cv2.cvtColor(img_preproc, cv2.COLOR_BGR2GRAY)
        # self.img_construction = img_gray  # 전처리된 템플릿 저장

        # # 6. [Keypoint/Descriptor 미리 추출]
        # self.kp_construction, self.des_construction = self.sift.detectAndCompute(self.img_construction, None)

        # # FLANN 매칭 초기화 (이하 동일)
        # FLANN_INDEX_KDTREE = 0
        # index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
        # search_params = {'checks': 50}
        # self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        squared_diff = (arr1 - arr2) ** 2
        total_sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1]  # cv_image_input and 2 should have same shape
        err = total_sum / num_all
        return err

    def cbFindTrafficSign(self, image_msg):
        # 프레임 처리 속도 제한 (3프레임 중 1프레임만 처리, 나머지는 패스)
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        # 이미지 타입에 따라 ROS 이미지 메시지를 OpenCV 이미지로 변환
        if self.sub_image_type == 'compressed':
            # 압축 이미지 → OpenCV 이미지
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            # raw 이미지 → OpenCV 이미지
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
        
        # -------- [1] ROI 설정 (예시: 상단 중앙 60% 영역) --------
        h, w, _ = cv_image_input.shape
        roi_w = int(w * 0.5)  # ROI 너비 (50%)
        roi_h = int(h * 0.6)  # ROI 높이 (60%)
        roi_x = w - roi_w     # 오른쪽에서 시작 (전체 가로 - ROI 너비)
        roi_y = 0             # 맨 위에서 시
        cv_image_roi = cv_image_input[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # # -------- [2] 색상공간 변환 및 밝기 보정 --------
        # # 예: LAB로 변환 후, L채널 CLAHE 적용(명암대비 향상)
        # lab = cv2.cvtColor(cv_image_roi, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # l = clahe.apply(l)
        # lab = cv2.merge((l, a, b))
        # cv_image_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # # -------- [3] 블러/노이즈 제거 --------
        # # Gaussian blur 적용
        # cv_image_preproc = cv2.GaussianBlur(cv_image_lab, (5, 5), 0)

        # # -------- [4] SIFT/FLANN 연산 (기존 알고리즘 그대로) --------
        # # SIFT는 gray image에서 사용
        # gray = cv2.cvtColor(cv_image_preproc, cv2.COLOR_BGR2GRAY)
        # kp1, des1 = self.sift.detectAndCompute(gray, None)

        # SIFT 매칭 기준값 설정
        MIN_MATCH_COUNT = 8
        MIN_MSE_DECISION = 50000

        # 입력 이미지에서 SIFT 특징점(키포인트) 및 디스크립터 추출
        # kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)
        kp1, des1 = self.sift.detectAndCompute(cv_image_roi, None)

        # 입력 이미지와 construction 표지판 이미지의 디스크립터를 FLANN으로 매칭 (k=2)
        matches_construction = self.flann.knnMatch(des1, self.des_construction, k=2)

        image_out_num = 1  # 기본 출력: 원본 이미지

        # 좋은 매칭점(good matches)만 추출 (거리 비율 테스트)
        good_construction = []
        for m, n in matches_construction:
            if m.distance < 0.7*n.distance:
                good_construction.append(m)

        # 좋은 매칭점이 충분하면 표지판 인식 성공으로 판단
        if len(good_construction) > MIN_MATCH_COUNT:
            # 매칭된 점들의 좌표 추출
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_construction]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_construction[m.trainIdx].pt for m in good_construction
            ]).reshape(-1, 1, 2)

            # 호모그래피 계산 (RANSAC)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_construction = mask.ravel().tolist()

            # 매칭된 점들의 MSE(평균제곱오차) 계산
            mse = self.fnCalcMSE(src_pts, dst_pts)
            if mse < MIN_MSE_DECISION:
                # 표지판 인식 결과 메시지 발행
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.construction.value
                self.pub_traffic_sign.publish(msg_sign)
                self.get_logger().info('construction')
                image_out_num = 2  # 매칭 결과 이미지 출력
        else:
            matches_construction = None
            # self.get_logger().info('not found')

        # 결과 이미지 퍼블리시 (원본 or 매칭 결과)
        if image_out_num == 1:
            # 표지판 인식 실패: 원본 이미지를 퍼블리시
            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(
                        cv_image_input, 'jpg'
                    )
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(
                        cv_image_input, 'bgr8'
                    )
                )
        elif image_out_num == 2:
            # 표지판 인식 성공: 매칭 결과 이미지를 퍼블리시
            draw_params_construction = {
                'matchColor': (255, 0, 0),  # 매칭점 색상
                'singlePointColor': None,
                'matchesMask': matches_construction,  # inlier만 그림
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
            final_construction = cv2.drawMatches(
                cv_image_input,
                # kp1,
                kp1_on_input,
                self.img_construction,
                self.kp_construction,
                good_construction,
                None,
                **draw_params_construction
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(
                        final_construction, 'jpg'
                    )
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(
                        final_construction, 'bgr8'
                    )
                )


def main(args=None):
    rclpy.init(args=args)
    node = DetectSign()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
