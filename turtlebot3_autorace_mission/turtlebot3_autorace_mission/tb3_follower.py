#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from tf_transformations import euler_from_quaternion
from collections import deque
from copy import deepcopy

class StableWaypointFollower(Node):
    def __init__(self):
        super().__init__('stable_waypoint_follower')
        self.leader_sub = self.create_subscription(Odometry, '/tb3_0/odom', self.leader_odom_cb, 10)
        self.follower_sub = self.create_subscription(Odometry, '/tb3_1/odom', self.follower_odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/tb3_1/cmd_vel', 10)
        self.waypoints = deque()  # (pose, timestamp)
        self.follower_pose = None
        self.current_idx = 0

        # 안정화 파라미터
        self.WAYPOINT_DIST_THRESHOLD = 0.03  # 3cm
        self.WAYPOINT_ANGLE_THRESHOLD = 0.1  # 약 5.7도
        self.MAX_WAYPOINTS = 1000  # 메모리 관리
        
        # 속도 제한 ↓
        self.MAX_LIN = 0.08  # 8cm/s (기존 18cm/s)
        self.MAX_ANG = 0.15  # 0.15 rad/s (기존 0.25)
        
        # 게인 ↓
        self.LIN_GAIN = 0.3  
        self.ANG_GAIN = 0.15  
        
        # 저역통과 필터
        self.ALPHA = 0.3  # 필터 계수
        self.prev_linear = 0.0
        self.prev_angular = 0.0

        self.leader_pose = None
        self.start_time = None
        self.timer = self.create_timer(0.1, self.update) 

    def leader_odom_cb(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        self.leader_pose = deepcopy(msg.pose.pose)
        self.waypoints.append((deepcopy(msg.pose.pose), now))
        if len(self.waypoints) > self.MAX_WAYPOINTS:
            self.waypoints.popleft()

    def follower_odom_cb(self, msg):
        self.follower_pose = msg.pose.pose

    def get_yaw_from_pose(self, pose):
        q = [pose.orientation.x, pose.orientation.y, 
             pose.orientation.z, pose.orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def update(self):
        if self.follower_pose is None or len(self.waypoints) < 2 or self.leader_pose is None:
            return

        # 시간 기반: 팔로워가 첫 waypoint 추종 시작한 시간 기록
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
            self.first_wp_time = self.waypoints[0][1]

        # 현재 목표 waypoint는 시간 동기화로 결정
        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self.start_time

        # 리더가 각 waypoint에 도달한 시간과 동일한 시간차로 따라가게
        target_idx = self.current_idx
        for i in range(self.current_idx, len(self.waypoints)):
            wp_time = self.waypoints[i][1]
            if wp_time - self.first_wp_time > elapsed:
                break
            target_idx = i

        if target_idx >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return

        target_pose, _ = self.waypoints[target_idx]
        fx = self.follower_pose.position.x
        fy = self.follower_pose.position.y
        fyaw = self.get_yaw_from_pose(self.follower_pose)
        tx = target_pose.position.x
        ty = target_pose.position.y
        tyaw = self.get_yaw_from_pose(target_pose)

        # 거리 및 각도 차이 계산
        distance = self.get_distance(tx, ty, fx, fy)
        angle_diff = tyaw - fyaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 도달 조건(위치+방향) 모두 만족하면 다음 waypoint로
        if (distance < self.WAYPOINT_DIST_THRESHOLD and abs(angle_diff) < self.WAYPOINT_ANGLE_THRESHOLD):
            self.current_idx = min(target_idx + 1, len(self.waypoints) - 1)
            if self.current_idx == len(self.waypoints) - 1:
                self.cmd_pub.publish(Twist())
                return

        # 제어 명령 생성
        cmd = Twist()
        if distance > self.WAYPOINT_DIST_THRESHOLD:
            # 속도 게인
            raw_linear = min(self.MAX_LIN, distance * self.LIN_GAIN)
            
            # 방향 계산
            target_angle = math.atan2(ty - fy, tx - fx)
            angle_error = target_angle - fyaw
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi
            # 각속도 게인
            raw_angular = max(min(angle_error * self.ANG_GAIN, self.MAX_ANG), -self.MAX_ANG)
        else:
            raw_linear = 0.0
            raw_angular = max(min(angle_diff * self.ANG_GAIN, self.MAX_ANG), -self.MAX_ANG)

        # 저역통과 필터 적용 (소프트하게잉)
        cmd.linear.x = self.ALPHA * raw_linear + (1 - self.ALPHA) * self.prev_linear
        cmd.angular.z = self.ALPHA * raw_angular + (1 - self.ALPHA) * self.prev_angular
        
        # 이전 값 업데이트
        self.prev_linear = cmd.linear.x
        self.prev_angular = cmd.angular.z

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = StableWaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
