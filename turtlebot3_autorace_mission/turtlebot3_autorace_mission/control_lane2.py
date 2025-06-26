#!/usr/bin/env python3

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, UInt8

class ControlLane(Node):
    def __init__(self):
        super().__init__('control_lane')

        # 구독자
        self.sub_lane = self.create_subscription(
            Float64, '/control/lane', self.callback_follow_lane, 1)
        self.sub_max_vel = self.create_subscription(
            Float64, '/control/max_vel', self.callback_get_max_vel, 1)
        self.sub_sign = self.create_subscription(
            UInt8, '/detect/traffic_sign', self.callback_sign_detected, 1)

        # 퍼블리셔
        self.pub_cmd_vel = self.create_publisher(Twist, '/control/cmd_vel', 1)

        # 내부 변수 초기화
        self.last_error = 0.0
        self.MAX_VEL = 0.1

        # STOP 신호 처리 변수
        self.stopping = False
        self.stop_end_time = 0.0
        self.stop_duration = 3.0  # 멈춤 지속 시간 (초)

        self.get_logger().info('ControlLane Node Initialized (STOP 표지판 감지 전용)')

    def callback_get_max_vel(self, msg):
        self.MAX_VEL = msg.data

    def callback_sign_detected(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9

        if msg.data == 0:  # STOP 표지판 인식 (UInt8 enum 값 0 가정)
            self.stopping = True
            self.stop_end_time = now + self.stop_duration
            self.get_logger().info('🛑 STOP 표지판 인식됨! 정지합니다.')

    def callback_follow_lane(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        twist = Twist()

        if self.stopping:
            if now < self.stop_end_time:
                # 멈춤 상태 유지
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)
                return
            else:
                # 정지 완료, 주행 재개
                self.stopping = False
                self.get_logger().info('✅ 정지 완료, 주행 재개')

        # 차선 추종 PID 제어
        center = msg.data
        error = center - 500
        Kp = 0.0025
        Kd = 0.007
        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05)
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)

        self.pub_cmd_vel.publish(twist)

    def shut_down(self):
        self.get_logger().info('Shutting down.')
        self.pub_cmd_vel.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
