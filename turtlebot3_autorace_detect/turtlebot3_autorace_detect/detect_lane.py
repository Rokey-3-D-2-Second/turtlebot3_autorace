#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64, UInt8
import math


class DetectLane(Node):
    def __init__(self):
        super().__init__('detect_lane')

        self.declare_parameters('', [
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
        ])

        self.hough_min_line_length = self.get_parameter('hough.min_line_length').get_parameter_value().integer_value
        self.hough_max_line_gap = self.get_parameter('hough.max_line_gap').get_parameter_value().integer_value
        self.hough_threshold = self.get_parameter('hough.threshold').get_parameter_value().integer_value
        self.canny_low_threshold = self.get_parameter('canny.low_threshold').get_parameter_value().integer_value
        self.canny_high_threshold = self.get_parameter('canny.high_threshold').get_parameter_value().integer_value
        self.blur_kernel_size = self.get_parameter('blur.kernel_size').get_parameter_value().integer_value
        self.roi_offset_y = self.get_parameter('roi.offset_y').get_parameter_value().integer_value
        self.roi_height = self.get_parameter('roi.height').get_parameter_value().integer_value
        self.lane_sep_min_slope = self.get_parameter('lane_separation.min_slope').get_parameter_value().double_value
        self.lane_sep_max_slope = self.get_parameter('lane_separation.max_slope').get_parameter_value().double_value
        self.lane_sep_x_offset = self.get_parameter('lane_separation.x_offset').get_parameter_value().integer_value

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1)
        else:
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1)

        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1)
            self.pub_image_canny_edge = self.create_publisher(
                CompressedImage, '/detect/image_canny_edge/compressed', 1)
        else:
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1)
            self.pub_image_canny_edge = self.create_publisher(
                Image, '/detect/image_canny_edge', 1)

        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()
        self.last_valid_cx = None
        self.alpha = 0.5

    def divide_left_right(self, lines, width):
        slopes, new_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if self.lane_sep_min_slope < abs(slope) < self.lane_sep_max_slope:
                slopes.append(slope)
                new_lines.append(line[0])

        left, right = [], []
        for idx, slope in enumerate(slopes):
            x1, y1, x2, y2 = new_lines[idx]
            if slope < 0 and x2 < width / 2 - self.lane_sep_x_offset:
                left.append([new_lines[idx].tolist()])
            elif slope > 0 and x1 > width / 2 + self.lane_sep_x_offset:
                right.append([new_lines[idx].tolist()])
        return left, right

    def get_line_params(self, lines):
        if not lines:
            return 0, 0
        x_sum, y_sum, m_sum = 0, 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += (y2 - y1) / (x2 - x1)
        n = len(lines)
        x_avg = x_sum / (2 * n)
        y_avg = y_sum / (2 * n)
        m = m_sum / n
        b = y_avg - m * x_avg
        return m, b

    def draw_lines(self, img, lines, offset_y):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1 + offset_y), (x2, y2 + offset_y), (0, 255, 255), 2)
        return img

    def cbFindLane(self, msg):
        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            frame = self.cvBridge.imgmsg_to_cv2(msg, 'bgr8')

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        edges = cv2.Canny(blur, self.canny_low_threshold, self.canny_high_threshold)

        if self.pub_image_type == 'compressed':
            self.pub_image_canny_edge.publish(self.cvBridge.cv2_to_compressed_imgmsg(edges, '.png'))
        else:
            self.pub_image_canny_edge.publish(self.cvBridge.cv2_to_imgmsg(edges, 'mono8'))

        result = frame.copy()
        centers, weights = [], []
        left_detected, right_detected = [False] * 5, [False] * 5

        roi_gap = self.roi_height // 5
        for i, weight in enumerate([0.3, 0.5, 0.7, 0.9, 1.0]):
            offset = self.roi_offset_y + i * roi_gap
            roi = edges[offset:offset + roi_gap, :]

            lines = cv2.HoughLinesP(roi, 1, math.pi / 180, self.hough_threshold,
                                    self.hough_min_line_length, self.hough_max_line_gap)

            if lines is not None:
                left, right = self.divide_left_right(lines, width)
                result = self.draw_lines(result, left, offset)
                result = self.draw_lines(result, right, offset)

                if left:
                    left_detected[i] = True
                if right:
                    right_detected[i] = True

                m_l, b_l = self.get_line_params(left)
                m_r, b_r = self.get_line_params(right)

                y = offset + roi_gap
                has_left = m_l != 0 or b_l != 0
                has_right = m_r != 0 or b_r != 0

                if has_left and has_right:
                    x_l = (y - b_l) / m_l if m_l != 0 else 0
                    x_r = (y - b_r) / m_r if m_r != 0 else width
                    if abs(x_l - x_r) < width * 0.25:
                        center = (x_l + x_r) / 2
                    else:
                        center = (x_l * 0.4 + x_r * 0.6)
                    centers.append(center)
                    weights.append(weight)
                elif has_left:
                    centers.append(width / 2 + width * 0.08)
                    weights.append(weight * 0.5)
                elif has_right:
                    centers.append(width / 2 - width * 0.08)
                    weights.append(weight * 0.5)

            cv2.rectangle(result, (0, offset), (width, offset + roi_gap), (0, 255, 0), 2)

        left_count = sum(left_detected)
        right_count = sum(right_detected)

        if centers:
            cx = np.average(centers, weights=weights)
            if left_count >= 4 and right_count <= 1:
                cx += width * 0.03
            elif right_count >= 4 and left_count <= 1:
                cx -= width * 0.03
            if self.last_valid_cx is not None:
                cx = self.alpha * cx + (1 - self.alpha) * self.last_valid_cx
            self.last_valid_cx = cx
            state = 2 if len(centers) >= 2 else 1
        else:
            cx = self.last_valid_cx if self.last_valid_cx is not None else width / 2
            state = 0

        if self.pub_image_type == 'compressed':
            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(result, 'jpg'))
        else:
            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(result, 'bgr8'))

        self.pub_lane.publish(Float64(data=cx))
        self.pub_lane_state.publish(UInt8(data=state))
        self.get_logger().info(f'[LaneDetect] State: {state}, CenterX: {cx:.2f}, L:{left_count}, R:{right_count}')


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()