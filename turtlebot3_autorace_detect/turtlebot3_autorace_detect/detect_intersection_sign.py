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

        self.sub_image_type = 'raw'  # you can choose image type 'compressed', 'raw'
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
        self.TrafficSign = Enum('TrafficSign', 'intersection left right')
        self.counter = 1

        self.fnPreproc()

        self.get_logger().info('DetectSign Node Initialized')

    def fnPreproc(self):
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()

        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        dir_path = os.path.join(dir_path, 'image')

        self.img_intersection = cv2.imread(dir_path + '/intersection.png', 0)
        self.img_left = cv2.imread(dir_path + '/left.png', 0)
        self.img_right = cv2.imread(dir_path + '/right.png', 0)

        self.kp_intersection, self.des_intersection = self.sift.detectAndCompute(
            self.img_intersection, None
        )
        self.kp_left, self.des_left = self.sift.detectAndCompute(self.img_left, None)
        self.kp_right, self.des_right = self.sift.detectAndCompute(self.img_right, None)

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
        squared_diff = (arr1 - arr2) ** 2
        total_sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1]  # cv_image_input and 2 should have same shape
        err = total_sum / num_all
        return err

    def cbFindTrafficSign(self, image_msg):
        # drop the frame to 1/5 (6fps) because of the processing speed.
        # This is up to your computer's operating power.
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            # converting compressed image to opencv image
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # -------- [1] ROI: 위쪽 절반만 사용 --------
        h, w, _ = cv_image_input.shape
        roi_y = 0               # 시작 y좌표
        roi_h = int(h * 0.5)    # 상단 40%
        cv_image_roi = cv_image_input[roi_y:roi_y+roi_h, :]

        MIN_MATCH_COUNT = 5
        MIN_MSE_DECISION = 50000

        # find the keypoints and descriptors with SIFT
        # kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)
        kp1, des1 = self.sift.detectAndCompute(cv_image_roi, None)

        matches_intersection = self.flann.knnMatch(des1, self.des_intersection, k=2)
        matches_left = self.flann.knnMatch(des1, self.des_left, k=2)
        matches_right = self.flann.knnMatch(des1, self.des_right, k=2)

        image_out_num = 1

        good_intersection = []
        for m, n in matches_intersection:
            if m.distance < 0.7*n.distance:
                good_intersection.append(m)
        if len(good_intersection) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_intersection]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_intersection[m.trainIdx].pt for m in good_intersection
            ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_intersection = mask.ravel().tolist()

            mse = self.fnCalcMSE(src_pts, dst_pts)
            if mse < MIN_MSE_DECISION:
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.intersection.value

                self.pub_traffic_sign.publish(msg_sign)
                self.get_logger().info('Detect intersection sign')
                image_out_num = 2

        good_left = []
        for m, n in matches_left:
            if m.distance < 0.7*n.distance:
                good_left.append(m)
        if len(good_left) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_left]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_left[m.trainIdx].pt for m in good_left
            ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_left = mask.ravel().tolist()

            mse = self.fnCalcMSE(src_pts, dst_pts)
            if mse < MIN_MSE_DECISION:
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.left.value

                self.pub_traffic_sign.publish(msg_sign)
                self.get_logger().info('Detect left sign')
                image_out_num = 3
        else:
            matches_left = None

        good_right = []
        for m, n in matches_right:
            if m.distance < 0.7*n.distance:
                good_right.append(m)
        if len(good_right) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_right]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_right[m.trainIdx].pt for m in good_right
            ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_right = mask.ravel().tolist()

            mse = self.fnCalcMSE(src_pts, dst_pts)
            if mse < MIN_MSE_DECISION:
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.right.value

                self.pub_traffic_sign.publish(msg_sign)
                self.get_logger().info('Detect right sign')
                image_out_num = 4
        else:
            matches_right = None

        if image_out_num == 1:
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
            draw_params_intersection = {
                'matchColor': (255, 0, 0),  # draw matches in green color
                'singlePointColor': None,
                'matchesMask': matches_intersection,  # draw only inliers
                'flags': 2
            }
            # kp1 좌표를 원본 이미지 기준으로 변환
            kp1_on_input = [
                cv2.KeyPoint(
                    kp.pt[0],
                    kp.pt[1] + roi_y,  # y좌표 보정
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id
                ) for kp in kp1
            ]
            final_intersection = cv2.drawMatches(
                cv_image_input,
                kp1_on_input,
                self.img_intersection,
                self.kp_intersection,
                good_intersection,
                None,
                **draw_params_intersection
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(
                        final_intersection, 'jpg'
                    )
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(
                        final_intersection, 'bgr8'
                    )
                )
        elif image_out_num == 3:
            draw_params_left = {
                'matchColor': (255, 0, 0),  # draw matches in green color
                'singlePointColor': None,
                'matchesMask': matches_left,  # draw only inliers
                'flags': 2
            }
            # kp1 좌표를 원본 이미지 기준으로 변환
            kp1_on_input = [
                cv2.KeyPoint(
                    kp.pt[0],
                    kp.pt[1] + roi_y,  # y좌표 보정
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id
                ) for kp in kp1
            ]
            final_left = cv2.drawMatches(
                cv_image_input,
                kp1_on_input,
                self.img_left,
                self.kp_left,
                good_left,
                None,
                **draw_params_left
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(
                        final_left, 'jpg'
                    )
                )

            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(
                        final_left, 'bgr8'
                    )
                )
        elif image_out_num == 4:
            draw_params_right = {
                'matchColor': (255, 0, 0),  # draw matches in green color
                'singlePointColor': None,
                'matchesMask': matches_right,  # draw only inliers
                'flags': 2
            }
            # kp1 좌표를 원본 이미지 기준으로 변환
            kp1_on_input = [
                cv2.KeyPoint(
                    kp.pt[0],
                    kp.pt[1] + roi_y,  # y좌표 보정
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id
                ) for kp in kp1
            ]
            final_right = cv2.drawMatches(
                cv_image_input,
                kp1_on_input,
                self.img_right,
                self.kp_right,
                good_right,
                None,
                **draw_params_right
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(
                        final_right, 'jpg'
                    )
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(
                        final_right, 'bgr8'
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
