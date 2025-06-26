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
# Author: Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
# Modified by: Google (for PID linear velocity)

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Int8
from nav_msgs.msg import Odometry

class ControlLane(Node):

    def __init__(self):
        super().__init__('control_lane')

        self.sub_lane = self.create_subscription(
            Float64,
            '/control/lane',
            self.callback_follow_lane,
            1
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.callback_get_max_vel,
            1
        )
        self.sub_avoid_cmd = self.create_subscription(
            Twist,
            '/avoid_control',
            self.callback_avoid_cmd,
            1
        )
        self.sub_avoid_active = self.create_subscription(
            Bool,
            '/avoid_active',
            self.callback_avoid_active,
            1
        )
        # Odometry 구독 추가 (선형 속도 피드백용)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # 신호등 상태 구독 추가
        self.traffic_light_state_subscriber = self.create_subscription(
            Int8,
            '/traffic_light_state',
            self.traffic_light_state_callback,
            10
        )
        self.current_traffic_light_state = 3 # 초기값: 3 (신호 없음/기본 주행)

        # 교차로 건널목 상태 구독 추가
        self.level_crossing_state_subscriber = self.create_subscription(
            Int8,
            '/level_crossing_state',
            self.level_crossing_state_callback,
            10
        )
        self.current_level_crossing_state = 3 # 초기값: 3 (건널목 상태 없음/기본 주행)

        # 정지 표지판 상태 구독 추가
        self.stop_sign_state_subscriber = self.create_subscription(
            Int8,
            '/stop_sign_state',  # 정지 표지판 상태 토픽
            self.stop_sign_state_callback,
            10
        )
        self.current_stop_sign_state = 0  # 0: 없음, 1: 정지 표지판 감지

        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/cmd_vel',
            1
        )

        # PD control related variables for angular velocity (original)
        self.last_error = 0 # This is for angular velocity PD

        # PID control related variables for linear velocity (newly added)
        self.Kp_linear = 0.5  # 비례 이득
        self.Ki_linear = 0.01 # 적분 이득
        self.Kd_linear = 0.05 # 미분 이득

        self.target_linear_velocity = 0.22  # 목표 선형 속도 (m/s), 초기값
        self.current_linear_velocity = 0.0 # 현재 선형 속도 (오도메트리에서 수신)
        self.linear_error = 0.0
        self.last_linear_error = 0.0
        self.linear_integral = 0.0
        self.linear_integral_limit = 0.5 # 적분 오버슈트 방지를 위한 적분 항 제한

        # 제어 주기 (초) - PID 계산에 필요
        self.timer_period = 0.1 # 10 Hz
        self.dt = self.timer_period # 시간 간격 (초)

        # Avoidance mode related variables (original)
        self.avoid_active = False
        self.avoid_twist = Twist()

        self.MAX_ANGULAR_Z = 2.0 # Maximum angular velocity
        self.MAX_ROBOT_LINEAR_X = 0.22 # 터틀봇3 최대 선형 속도
        self.MIN_ROBOT_LINEAR_X = -0.22 # 터틀봇3 최소 선형 속도


    def odom_callback(self, msg):
        self.current_linear_velocity = msg.twist.twist.linear.x

    def traffic_light_state_callback(self, msg):
        self.current_traffic_light_state = msg.data
        self.get_logger().info(f'Received traffic light state: {self.current_traffic_light_state}')

    def level_crossing_state_callback(self, msg):
        self.current_level_crossing_state = msg.data
        self.get_logger().info(f'Received level crossing state: {self.current_level_crossing_state}')

    def stop_sign_state_callback(self, msg):
        self.current_stop_sign_state = msg.data
        self.get_logger().info(f'Received stop sign state: {self.current_stop_sign_state}')

    def calculate_linear_pid(self, target_vel):
        self.linear_error = target_vel - self.current_linear_velocity
        self.linear_integral += self.linear_error * self.dt
        self.linear_integral = max(min(self.linear_integral, self.linear_integral_limit), -self.linear_integral_limit)
        linear_derivative = (self.linear_error - self.last_linear_error) / self.dt
        output = (self.Kp_linear * self.linear_error) + \
                 (self.Ki_linear * self.linear_integral) + \
                 (self.Kd_linear * linear_derivative)
        self.last_linear_error = self.linear_error
        return output

    def callback_get_max_vel(self, max_vel_msg):
        self.target_linear_velocity = max_vel_msg.data
        self.get_logger().info(f"Updated base target linear velocity to: {self.target_linear_velocity} m/s")

    def callback_follow_lane(self, desired_center):
        if self.avoid_active:
            return

        twist = Twist()
        effective_target_linear_vel = self.target_linear_velocity

        # 정지 표지판 상태 우선 체크
        if self.current_stop_sign_state == 1:
            effective_target_linear_vel = 0.0
            self.get_logger().info('Stop sign detected. Forcing linear speed to 0.')
            self.linear_integral = 0.0
            self.last_linear_error = 0.0

        # 신호등 및 건널목 상태에 따른 선형 속도 결정
        elif self.current_level_crossing_state == 4:
            effective_target_linear_vel = 0.0
            self.get_logger().info('Level crossing STOP detected. Forcing linear speed to 0.')
            self.linear_integral = 0.0
            self.last_linear_error = 0.0
        elif self.current_traffic_light_state == 0:
            effective_target_linear_vel = 0.0
            self.get_logger().info('Red light detected. Forcing linear speed to 0.')
            self.linear_integral = 0.0
            self.last_linear_error = 0.0
        elif self.current_traffic_light_state == 1:
            effective_target_linear_vel = self.MAX_ROBOT_LINEAR_X
            self.get_logger().info('Green light detected. Default speed or MAX_ROBOT_LINEAR_X.')
        elif self.current_traffic_light_state == 2:
            effective_target_linear_vel = self.MAX_ROBOT_LINEAR_X
            self.get_logger().info('Yellow light detected. Default speed or MAX_ROBOT_LINEAR_X.')
        else:
            effective_target_linear_vel = self.MAX_ROBOT_LINEAR_X
            self.get_logger().info('No traffic light/crossing state. Default driving speed.')

        pid_linear_output = self.calculate_linear_pid(effective_target_linear_vel)
        twist.linear.x = max(min(pid_linear_output, self.MAX_ROBOT_LINEAR_X), self.MIN_ROBOT_LINEAR_X)

        center = desired_center.data
        error = center - 500

        Kp_angular = 0.005
        Kd_angular = 0.007

        angular_z = Kp_angular * error + Kd_angular * (error - self.last_error)
        self.last_error = error
        
        twist.angular.z = -max(angular_z, -self.MAX_ANGULAR_Z) if angular_z < 0 else -min(angular_z, self.MAX_ANGULAR_Z)

        self.pub_cmd_vel.publish(twist)

        self.get_logger().info(
            f"StopSign: {self.current_stop_sign_state}, TL_State: {self.current_traffic_light_state}, "
            f"LC_State: {self.current_level_crossing_state}, "
            f"Cmd Linear X: {twist.linear.x:.3f}, Cmd Angular Z: {twist.angular.z:.3f}"
        )

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg
        if self.avoid_active:
            self.pub_cmd_vel.publish(self.avoid_twist)

    def callback_avoid_active(self, bool_msg):
        self.avoid_active = bool_msg.data
        if self.avoid_active:
            self.get_logger().info('Avoidance mode activated.')
        else:
            self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')
            self.linear_integral = 0.0
            self.last_linear_error = 0.0
            self.last_error = 0.0

    def shut_down(self):
        self.get_logger().info('Shutting down. cmd_vel will be 0')
        twist = Twist()
        self.pub_cmd_vel.publish(twist)


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
