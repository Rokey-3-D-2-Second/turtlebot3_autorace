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
# Author: Leon Jung, Gilbert, Ashe Kim, Jun

from enum import Enum  # 열거형(Enum)을 사용하기 위한 모듈 임포트
import os  # 운영 체제와 상호 작용하기 위한 모듈 임포트 (파일 경로 등)

import cv2  # OpenCV 라이브러리 임포트 (이미지 처리)
from cv_bridge import CvBridge  # ROS 이미지 메시지와 OpenCV 이미지 간의 변환을 위한 브리지 임포트
import numpy as np  # 수치 계산을 위한 NumPy 라이브트러리 임포트
import rclpy  # ROS 2 Python 클라이언트 라이브러리 임포트
from rclpy.node import Node  # ROS 2 노드 클래스 임포트
from sensor_msgs.msg import CompressedImage  # 압축된 이미지 메시지 타입 임포트
from sensor_msgs.msg import Image  # 일반 이미지 메시지 타입 임포트
from std_msgs.msg import UInt8  # 8비트 부호 없는 정수 메시지 타입 임포트


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
        super().__init__('detect_sign')  # 'detect_sign' 이름으로 ROS 2 노드를 초기화합니다.

        self.sub_image_type = 'raw'  # 구독할 이미지 메시지 타입 (압축: 'compressed', 원본: 'raw')
        self.pub_image_type = 'compressed'  # 발행할 이미지 메시지 타입 (압축: 'compressed', 원본: 'raw')

        # 이미지 구독자 설정 (선택된 이미지 타입에 따라 다름)
        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage,
                '/detect/image_input/compressed',  # 압축 이미지 토픽 이름
                self.cbFindTrafficSign,  # 콜백 함수 지정
                10  # QoS 큐 크기
            )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image,
                '/detect/image_input',  # 원본 이미지 토픽 이름
                self.cbFindTrafficSign,  # 콜백 함수 지정
                10  # QoS 큐 크기
            )

        # 교통 표지판 감지 결과 발행자 설정 (UInt8 메시지)
        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        
        # 감지된 이미지를 발행하는 발행자 설정 (선택된 이미지 타입에 따라 다름)
        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign = self.create_publisher(
                CompressedImage,
                '/detect/image_output/compressed', 10
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign = self.create_publisher(
                Image, '/detect/image_output', 10
            )

        self.cvBridge = CvBridge()  # CvBridge 객체 생성 (ROS 이미지와 OpenCV 이미지 변환용)
        self.TrafficSign = Enum('TrafficSign', 'stop')  # 'stop'이라는 교통 표지판 열거형 정의
        self.counter = 1  # 프레임 스킵을 위한 카운터

        self.fnPreproc()  # SIFT 및 정지 표지판 이미지 로딩 등 전처리 함수 호출

        self.get_logger().info('DetectSign Node Initialized')  # 노드 초기화 완료 로그 메시지 출력

    def fnPreproc(self):
        """
        SIFT 특징점 감지기 초기화 및 참조 교통 표지판 이미지 로드 및 특징점 계산
        """
        # SIFT 감지기 초기화 (OpenCV 4.x 이상에서는 cv2.SIFT_create() 사용)
        self.sift = cv2.SIFT_create()

        # 정지 표지판 이미지 파일 경로 설정
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        dir_path = os.path.join(dir_path, 'image')

        # 정지 표지판 이미지 로드 (흑백 이미지로 로드)
        self.img_stop = cv2.imread(dir_path + '/stop.png', 0)  # trainImage1 (훈련 이미지)
        
        # 정지 표지판 이미지에서 SIFT 특징점(kp_stop)과 기술자(des_stop)를 계산
        self.kp_stop, self.des_stop = self.sift.detectAndCompute(self.img_stop, None)

        # FLANN 매칭을 위한 파라미터 설정
        FLANN_INDEX_KDTREE = 0  # K-D 트리 알고리즘 사용
        index_params = {
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5  # K-D 트리 개수
        }

        search_params = {
            'checks': 50  # 검색 재확인 횟수
        }

        # FLANN 기반 매처 초기화 (고속 근사 최근접 이웃 매칭)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        """
        두 배열 간의 평균 제곱 오차(MSE)를 계산합니다.
        주로 특징점 매칭 후 변환된 점들의 오차를 측정하는 데 사용됩니다.
        """
        squared_diff = (arr1 - arr2) ** 2  # 두 배열의 차이를 제곱
        total_sum = np.sum(squared_diff)  # 제곱된 차이의 합계
        num_all = arr1.shape[0] * arr1.shape[1]  # 배열의 전체 요소 개수
        err = total_sum / num_all  # 평균 제곱 오차 계산
        return err

    def cbFindTrafficSign(self, image_msg):
        """
        이미지 메시지를 구독하고 교통 표지판을 감지하는 콜백 함수입니다.
        SIFT를 사용하여 이미지에서 '정지' 표지판을 찾습니다.
        """
        # 처리 속도 향상을 위해 프레임을 1/3로 드롭 (컴퓨터 성능에 따라 조절)
        if self.counter % 3 != 0:
            self.counter += 1
            return  # 현재 프레임은 스킵
        else:
            self.counter = 1  # 카운터 리셋

        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        if self.sub_image_type == 'compressed':
            # 압축된 이미지 데이터를 NumPy 배열로 변환 후 OpenCV로 디코딩
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')  # 원본 이미지 (BGR8 형식)

        MIN_MATCH_COUNT = 4  # 호모그래피를 계산하기 위한 최소 매치 수
        MIN_MSE_DECISION = 50000  # MSE 기반으로 표지판 감지를 결정하는 임계값

        # 입력 이미지에서 SIFT 특징점(kp1)과 기술자(des1)를 찾습니다.
        kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)

        # FLANN 매처를 사용하여 입력 이미지의 기술자(des1)와 정지 표지판의 기술자(des_stop)를 매칭합니다.
        # k=2는 각 기술자에 대해 가장 가까운 2개의 이웃을 찾도록 합니다.
        matches_stop = self.flann.knnMatch(des1, self.des_stop, k=2)

        image_out_num = 1  # 출력 이미지 유형을 결정하는 플래그 (1: 원본, 2: 매칭 결과)

        good_stop = []  # 좋은 매치들을 저장할 리스트
        for m, n in matches_stop:
            # Lowe의 비율 테스트: 두 번째로 가까운 매치보다 충분히 가까운 매치만 '좋은 매치'로 간주
            if m.distance < 0.7 * n.distance:
                good_stop.append(m)

        # 충분한 좋은 매치가 있는 경우
        if len(good_stop) > MIN_MATCH_COUNT:
            # 좋은 매치에서 원본 이미지의 특징점(src_pts)과 정지 표지판 이미지의 특징점(dst_pts)을 추출
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_stop]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_stop[m.trainIdx].pt for m in good_stop
            ]).reshape(-1, 1, 2)

            # RANSAC 알고리즘을 사용하여 호모그래피 행렬(M)과 인라이어 마스크(mask)를 계산
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_stop = mask.ravel().tolist()  # 마스크를 1차원 리스트로 변환

            # 변환된 특징점들의 MSE를 계산하여 표지판 감지 신뢰도를 평가
            mse = self.fnCalcMSE(src_pts, dst_pts)
            if mse < MIN_MSE_DECISION:
                msg_sign = UInt8()  # UInt8 메시지 객체 생성
                msg_sign.data = self.TrafficSign.stop.value  # 메시지 데이터에 'stop' 값 할당

                self.pub_traffic_sign.publish(msg_sign)  # 교통 표지판 감지 메시지 발행

                self.get_logger().info('stop')  # 'stop' 감지 로그 출력
                image_out_num = 2  # 출력 이미지를 매칭 결과로 설정
        else:
            matches_stop = None  # 좋은 매치가 충분하지 않으면 매치 마스크를 None으로 설정
            # self.get_logger().info('nothing')  # 아무것도 감지되지 않았을 때 로그 (선택 사항)

        # 결과 이미지 발행
        if image_out_num == 1:  # 표지판이 감지되지 않았거나 MSE 기준 미달인 경우 원본 이미지 발행
            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(
                        cv_image_input, 'jpg'  # JPG 형식으로 압축하여 발행
                    )
                )

            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(
                        cv_image_input, 'bgr8'  # BGR8 형식으로 원본 이미지 발행
                    )
                )
        elif image_out_num == 2:  # '정지' 표지판이 감지된 경우 매칭 결과 이미지 발행
            draw_params2 = {
                'matchColor': (0, 0, 255),  # 매치 라인 색상 (파란색)
                'singlePointColor': None,  # 단일 특징점 색상 (None: 기본값 사용)
                'matchesMask': matches_stop,  # 인라이어(inliers)만 그리기 위한 마스크
                'flags': 2  # 매칭 그리기 플래그 (cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            }

            # 입력 이미지와 정지 표지판 이미지 간의 매치 결과를 그립니다.
            final_stop = cv2.drawMatches(
                cv_image_input,
                kp1,
                self.img_stop,
                self.kp_stop,
                good_stop,
                None,
                **draw_params2
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(
                        final_stop, 'jpg'  # JPG 형식으로 압축하여 발행
                    )
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(
                        final_stop, 'bgr8'  # BGR8 형식으로 원본 이미지 발행
                    )
                )


def main(args=None):
    """
    ROS 2 노드를 실행하는 메인 함수입니다.
    """
    rclpy.init(args=args)  # ROS 2 Python 클라이언트 라이브러리 초기화
    node = DetectSign()  # DetectSign 노드 객체 생성
    rclpy.spin(node)  # 노드가 종료될 때까지 메시지 콜백을 계속 처리
    node.destroy_node()  # 노드 객체 소멸 (리소스 해제)
    rclpy.shutdown()  # ROS 2 시스템 종료


if __name__ == '__main__':
    main() # 스크립트가 직접 실행될 때 main 함수 호출