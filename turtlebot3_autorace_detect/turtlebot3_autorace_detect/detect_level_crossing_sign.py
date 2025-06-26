#!/usr/bin/env python3

# 이 코드는 ROS2 노드에서 STOP 교통 표지판을 인식하고, 인식되었을 때 토픽을 통해 알리는 기능을 수행함.

from enum import Enum
import os

import cv2  # OpenCV 사용
from cv_bridge import CvBridge  # ROS 이미지 메시지를 OpenCV 이미지로 변환
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image  # 이미지 관련 ROS 메시지 타입
from std_msgs.msg import UInt8  # 표지판 인식 결과를 퍼블리시할 때 사용


class DetectSign(Node):
    def __init__(self):
        super().__init__('detect_sign')  # 노드 이름

        # 이미지 타입 설정
        self.sub_image_type = 'raw'  # 입력 이미지 타입 ('compressed' or 'raw')
        self.pub_image_type = 'compressed'  # 출력 이미지 타입

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

        # 표지판 결과 퍼블리셔 (정수 타입)
        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)

        # 인식 결과 이미지 퍼블리셔 (표지판 매칭 결과 시각화용)
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
        # SIFT (Scale-Invariant Feature Transform) 검출기 초기화
        self.sift = cv2.SIFT_create()

        # 이미지 경로 설정 및 stop 표지판 이미지 불러오기
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        dir_path = os.path.join(dir_path, 'image')

        self.img_stop = cv2.imread(dir_path + '/stop.png', 0)  # stop 표지판 이미지 (grayscale)
        self.kp_stop, self.des_stop = self.sift.detectAndCompute(self.img_stop, None)  # keypoint 및 descriptor 추출

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
        # Mean Squared Error 계산
        squared_diff = (arr1 - arr2) ** 2
        total_sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1]
        err = total_sum / num_all
        return err

    def cbFindTrafficSign(self, image_msg):
        # 프레임 드랍: 처리 속도를 위한 프레임 스킵
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

        MIN_MATCH_COUNT = 7
        MIN_MSE_DECISION = 50000

        # find the keypoints and descriptors with SIFT
        # kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)
        kp1, des1 = self.sift.detectAndCompute(cv_image_roi, None)

        # FLANN으로 stop 이미지와 매칭
        matches_stop = self.flann.knnMatch(des1, self.des_stop, k=2)

        image_out_num = 1  # 기본 출력 이미지 설정 (인식 못함)

        # Lowe’s ratio test로 좋은 매칭 필터링
        good_stop = []
        for m, n in matches_stop:
            if m.distance < 0.7 * n.distance:
                good_stop.append(m)

        # 충분히 매칭되면 STOP 표지판으로 판단
        if len(good_stop) > MIN_MATCH_COUNT:
            # 매칭된 좌표들 추출
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_stop]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp_stop[m.trainIdx].pt for m in good_stop]).reshape(-1, 1, 2)

            # 호모그래피 추정
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_stop = mask.ravel().tolist()

            # MSE 계산으로 최종 판단
            mse = self.fnCalcMSE(src_pts, dst_pts)
            if mse < MIN_MSE_DECISION:
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.stop.value

                # STOP 표지판이라고 퍼블리시
                self.pub_traffic_sign.publish(msg_sign)
                self.get_logger().info('stop')

                image_out_num = 2  # 매칭 시각화 이미지 출력

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
    rclpy.init(args=args)
    node = DetectSign()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# 이 노드 파일이 직접 실행되었을 때 main() 실행
if __name__ == '__main__':
    main()
