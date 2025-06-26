#!/usr/bin/env python3
# -*- coding: utf-8 -*- # 한글 주석을 위한 인코딩 명시

import os
import cv2
import numpy as np

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult, IntegerRange

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Int8
from ament_index_python.packages import get_package_share_directory
from enum import Enum


class StopSignState(Enum):
    """정지 표지판 감지 및 제어를 위한 상태 정의."""
    DRIVING = 0             # 기본 주행 상태 (정지 표지판 감지 안 됨)
    APPROACHING_SIGN = 1    # 정지 표지판 감지 및 접근 중 (제어 노드에 감속/정지 신호)
    STOPPED_AT_SIGN = 2     # 정지 표지판 앞에서 정지 중 (제어 노드에 정지 신호 유지)
    RESUMING_DRIVING = 3    # 정지 후 재출발 (제어 노드에 주행 가능 신호)


class DetectStopSign(Node):
    def __init__(self):
        super().__init__('detect_stop_sign')

        self.get_logger().info('🚦 DetectStopSign 노드 초기화 중...')

        # 1. ROS 파라미터 선언 및 초기화
        self.declare_parameter('stop_sign.min_match_count', 4,
                               descriptor=ParameterDescriptor(description='SIFT 매칭 최소 개수'))
        self.declare_parameter('stop_sign.min_mse', 50000,
                               descriptor=ParameterDescriptor(description='SIFT 매칭 MSE 임계값 (현재는 사용 안 함)'))
        self.declare_parameter('stop_sign.stop_duration_sec', 3.0,
                               descriptor=ParameterDescriptor(description='정지 표지판 앞에서 대기할 시간 (초)'))
        self.declare_parameter('is_detection_calibration_mode', True,
                               descriptor=ParameterDescriptor(description='감지 보정 모드 활성화 여부'))
        self.declare_parameter('stop_sign.detection_threshold_frames', 3,
                               descriptor=ParameterDescriptor(description='연속 감지해야 할 프레임 수'))
        self.declare_parameter('stop_sign.ransac_reproj_threshold', 5.0,
                               descriptor=ParameterDescriptor(description='Homography RANSAC 재투영 임계값'))
        self.declare_parameter('stop_sign.lowe_ratio_threshold', 0.6,
                               descriptor=ParameterDescriptor(description='FLANN Lowe\'s Ratio Test 임계값'))
        # --- (새로 추가된 파라미터) ---
        self.declare_parameter('stop_sign.reset_flag_delay_sec', 3.0,
                               descriptor=ParameterDescriptor(description='정지 후 stop_sign_passed_flag를 리셋할 시간 (초)'))


        # 선언된 파라미터 값 가져오기
        self.MIN_MATCH_COUNT = self.get_parameter('stop_sign.min_match_count').value
        self.MIN_MSE = self.get_parameter('stop_sign.min_mse').value
        self.stop_duration_sec = self.get_parameter('stop_sign.stop_duration_sec').value
        self.is_calibration_mode = self.get_parameter('is_detection_calibration_mode').value
        self.detection_threshold_frames = self.get_parameter('stop_sign.detection_threshold_frames').value
        self.RANSAC_REPROJ_THRESHOLD = self.get_parameter('stop_sign.ransac_reproj_threshold').value
        self.LOWE_RATIO_THRESHOLD = self.get_parameter('stop_sign.lowe_ratio_threshold').value
        # --- (새로 추가된 파라미터 값) ---
        self.reset_flag_delay_sec = self.get_parameter('stop_sign.reset_flag_delay_sec').value


        # 파라미터 변경 콜백 등록
        self.add_on_set_parameters_callback(self.on_parameter_change)

        # 2. ROS 토픽 구독 및 발행 설정
        self.image_type = 'raw' 
        if self.image_type == 'compressed':
            self.image_sub = self.create_subscription(
                CompressedImage,
                '/detect/image_input/compressed',
                self.image_callback,
                10)
        else:
            self.image_sub = self.create_subscription(
                Image,
                '/camera/image_compensated',
                self.image_callback,
                10)

        self.stop_sign_pub = self.create_publisher(Bool, '/stop_sign_detected', 10)
        self.stop_sign_state_pub = self.create_publisher(Int8, '/stop_sign_state', 10)
        
        # --- (선택 사항) 미션 시작/종료 신호를 받을 구독자 ---
        # 이 토픽은 로봇이 새로운 미션 구간으로 진입하거나 미션이 재시작될 때
        # self.stop_sign_passed_flag를 초기화하는 데 사용될 수 있습니다.
        # 예: /mission_start 또는 /reset_mission 등의 토픽
        # self.mission_reset_sub = self.create_subscription(
        #     Bool,
        #     '/mission_reset', # 예시 토픽 이름
        #     self.mission_reset_callback,
        #     1
        # )

        self.bridge = CvBridge()
        self.cv_image = None

        # 3. SIFT 및 FLANN 매칭기 초기화
        self.sift = cv2.SIFT_create()
        self.prepare_stop_image()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 4. 상태 머신 및 감지 로직 관련 변수
        self.current_state = StopSignState.DRIVING
        self.stop_timer = None # 정지 지속 시간을 위한 타이머
        self.stop_sign_detected_in_frame = False 
        self.consecutive_detection_count = 0 

        # --- (핵심 변수) ---
        # 이 플래그가 True면 정지 표지판 감지를 일시적으로 무시합니다.
        # 일정 시간 후 False로 다시 리셋됩니다.
        self.stop_sign_passed_flag = False 
        self.reset_passed_flag_timer = None # stop_sign_passed_flag를 리셋할 타이머

        # 이 플래그는 정지 표지판에 이미 접근 중이거나 정지해있는 상태일 때
        # 중복된 정지/감속 명령을 막기 위해 사용됩니다. (상태 머신 내부 플래그)
        self.is_approaching_active = False 

        # 5. 주기적인 이미지 처리 타이머
        self.processing_timer = self.create_timer(1.0/15.0, self.timer_callback)

    def on_parameter_change(self, params):
        """ROS 파라미터 변경 시 호출되는 콜백 함수."""
        for param in params:
            if param.name == 'stop_sign.min_match_count':
                self.MIN_MATCH_COUNT = param.value
                self.get_logger().info(f'파라미터 업데이트: min_match_count = {self.MIN_MATCH_COUNT}')
            elif param.name == 'stop_sign.min_mse':
                self.MIN_MSE = param.value
                self.get_logger().info(f'파라미터 업데이트: min_mse = {self.MIN_MSE} (감지 결정에 사용되지 않음)')
            elif param.name == 'stop_sign.stop_duration_sec':
                self.stop_duration_sec = param.value
                self.get_logger().info(f'파라미터 업데이트: stop_duration_sec = {self.stop_duration_sec}')
            elif param.name == 'is_detection_calibration_mode':
                self.is_calibration_mode = param.value
                self.get_logger().info(f'파라미터 업데이트: is_calibration_mode = {self.is_calibration_mode}')
            elif param.name == 'stop_sign.detection_threshold_frames':
                self.detection_threshold_frames = param.value
                self.get_logger().info(f'파라미터 업데이트: detection_threshold_frames = {self.detection_threshold_frames}')
            elif param.name == 'stop_sign.ransac_reproj_threshold':
                self.RANSAC_REPROJ_THRESHOLD = param.value
                self.get_logger().info(f'파라미터 업데이트: ransac_reproj_threshold = {self.RANSAC_REPROJ_THRESHOLD}')
            elif param.name == 'stop_sign.lowe_ratio_threshold':
                self.LOWE_RATIO_THRESHOLD = param.value
                self.get_logger().info(f'파라미터 업데이트: lowe_ratio_threshold = {self.LOWE_RATIO_THRESHOLD}')
            elif param.name == 'stop_sign.reset_flag_delay_sec':
                self.reset_flag_delay_sec = param.value
                self.get_logger().info(f'파라미터 업데이트: reset_flag_delay_sec = {self.reset_flag_delay_sec}')
        return SetParametersResult(successful=True)

    def prepare_stop_image(self):
        """참조 정지 표지판 이미지(stop.png)를 로드하고 SIFT 특징점을 계산합니다."""
        try:
            package_share_dir = get_package_share_directory('turtlebot3_autorace_detect')
            stop_path = os.path.join(package_share_dir, 'image', 'stop.png')
            
            self.img_stop = cv2.imread(stop_path, cv2.IMREAD_GRAYSCALE)
            
            if self.img_stop is None:
                self.get_logger().error(f'❌ 정지 표지판 이미지 로드 실패: {stop_path}. 경로와 파일 존재 여부를 확인하세요.')
                self.kp_stop = None
                self.des_stop = None
                return
            
            self.kp_stop, self.des_stop = self.sift.detectAndCompute(self.img_stop, None)
            
            if self.kp_stop is not None:
                self.get_logger().info(f'📂 정지 표지판 이미지 로드 성공: {stop_path}. 특징점 개수: {len(self.kp_stop)}')
            else:
                 self.get_logger().error(f'❌ 정지 표지판 이미지에서 특징점 감지 실패: {stop_path}.')
        except Exception as e:
            self.get_logger().error(f'정지 표지판 이미지 로드 또는 특징점 감지 중 예외 발생: {str(e)}')
            self.kp_stop = None
            self.des_stop = None

    def image_callback(self, msg):
        """ROS 이미지 메시지를 받아서 OpenCV 이미지로 변환하고 전처리를 수행합니다."""
        try:
            if self.image_type == 'compressed':
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image_raw = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            if cv_image_raw is None:
                self.get_logger().warn("빈 이미지 프레임 수신.")
                self.cv_image = None
                return

            img_yuv = cv2.cvtColor(cv_image_raw, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            self.cv_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        except Exception as e:
            self.get_logger().warn(f'이미지 디코딩 또는 전처리 실패: {str(e)}')
            self.cv_image = None

    # --- (선택 사항) 미션 리셋 콜백 (필요에 따라 주석 해제 후 구현) ---
    # def mission_reset_callback(self, msg):
    #     if msg.data: # 미션 리셋 신호가 들어오면
    #         self.get_logger().info("미션 리셋 신호 수신. 정지 표지판 통과 플래그 초기화.")
    #         self.stop_sign_passed_flag = False
    #         self.is_approaching_active = False # 접근 상태도 초기화
    #         if self.reset_passed_flag_timer: # 리셋 타이머가 동작 중이면 취소
    #             self.reset_passed_flag_timer.cancel()
    #             self.reset_passed_flag_timer = None
    #         # 필요한 경우 상태를 DRIVING으로 강제 전환
    #         if self.current_state != StopSignState.DRIVING:
    #             self.current_state = StopSignState.DRIVING
    #             self.publish_stop_signal(False) # 정지 신호 해제

    def timer_callback(self):
        """주기적으로 실행되며, 이미지 처리 및 상태 업데이트를 수행합니다."""
        if self.cv_image is None:
            return

        # 현재 프레임에서 정지 표지판 감지 여부 확인
        # (단, `stop_sign_passed_flag`가 True인 동안에는 감지 시도 자체를 하지 않습니다.)
        if not self.stop_sign_passed_flag: 
            is_detected_this_frame = self.process_stop_sign_detection(self.cv_image.copy())
        else: # `stop_sign_passed_flag`가 True면 감지되지 않은 것으로 처리
            is_detected_this_frame = False 
            # 이 경우 연속 감지 카운트는 0으로 유지 (혹시라도 이전 값 남아있을까봐)
            self.consecutive_detection_count = 0

        # 연속 감지 카운트 업데이트
        # `stop_sign_passed_flag`가 True인 동안에는 `is_detected_this_frame`이 항상 False이므로,
        # 이 블록은 그 상태에서는 `consecutive_detection_count`를 0으로 유지합니다.
        if is_detected_this_frame:
            self.consecutive_detection_count += 1
        else:
            self.consecutive_detection_count = 0

        # 연속 감지 임계값을 넘었는지 확인하여 최종 감지 상태 결정
        self.stop_sign_detected_in_frame = (self.consecutive_detection_count >= self.detection_threshold_frames)

        # 상태 머신 업데이트
        self.update_state()

        # 현재 상태 및 감지 정보 로깅
        self.get_logger().info(
            f"현재 상태: {self.current_state.name}, "
            f"최종 감지 (연속): {self.stop_sign_detected_in_frame}, "
            f"이번 프레임 감지: {is_detected_this_frame}, "
            f"연속 감지 횟수: {self.consecutive_detection_count}, "
            f"통과 플래그: {self.stop_sign_passed_flag}" # 통과 플래그 상태 로깅
        )


    def process_stop_sign_detection(self, image_frame):
        """
        주어진 이미지 프레임에서 정지 표지판을 SIFT 및 FLANN 매칭으로 감지합니다.
        Homography 계산 성공 시 True 반환, 실패 시 False 반환.
        """
        if self.des_stop is None or self.kp_stop is None:
            self.get_logger().warn("정지 표지판 참조 이미지가 로드되지 않았거나 특징점을 찾을 수 없습니다. 감지할 수 없습니다.")
            return False

        gray_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # 노이즈 감소 및 특징점 안정화
        
        kp_frame, des_frame = self.sift.detectAndCompute(gray_image, None)
        self.get_logger().debug(f"프레임 특징점 개수: {len(kp_frame) if kp_frame is not None else 0}")
        
        if des_frame is None or len(kp_frame) < 2:
            self.get_logger().debug("현재 프레임에서 충분한 특징점이 감지되지 않았습니다. 감지를 건너ym니다.")
            return False

        try:
            matches = self.flann.knnMatch(des_frame, self.des_stop, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < self.LOWE_RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)
        except cv2.error as e:
            self.get_logger().warn(f"FLANN 매칭 오류 발생: {e}")
            return False

        self.get_logger().debug(f"찾은 유효 매칭 개수 (Good Matches): {len(good_matches)}")

        if self.is_calibration_mode and len(good_matches) >= self.MIN_MATCH_COUNT:
            img_matches = cv2.drawMatches(image_frame, kp_frame, self.img_stop, self.kp_stop,
                                            good_matches, None,
                                            matchColor=(0, 255, 0), singlePointColor=None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('SIFT Matches', img_matches)
            cv2.waitKey(1)

        if len(good_matches) >= self.MIN_MATCH_COUNT:
            self.get_logger().debug(f"충분한 유효 매칭 ({len(good_matches)} >= {self.MIN_MATCH_COUNT})으로 Homography 계산 시도!")
            
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp_stop[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.RANSAC_REPROJ_THRESHOLD)

            if M is not None and mask is not None:
                matches_mask = mask.ravel().tolist()
                inlier_count = sum(matches_mask)

                if inlier_count >= self.MIN_MATCH_COUNT:
                    self.get_logger().debug(f"✅ Homography 계산 성공! Inlier 개수: {inlier_count} (최소 {self.MIN_MATCH_COUNT})")
                    return True
                else:
                    self.get_logger().debug(f"Homography는 찾았으나 Inlier 개수 ({inlier_count})가 최소 매칭 개수 ({self.MIN_MATCH_COUNT}) 미만입니다. 감지 실패.")
            else:
                self.get_logger().debug("Homography 계산 실패 (M 또는 mask가 None입니다). 감지 실패.")
        else:
            self.get_logger().debug(f"유효 매칭 개수 부족 ({len(good_matches)} < {self.MIN_MATCH_COUNT}). 감지 실패.")
        return False

    def update_state(self):
        """정지 표지판 감지 결과에 따라 노드의 상태를 업데이트합니다."""
        new_state = self.current_state

        # 현재 상태가 DRIVING일 때
        if self.current_state == StopSignState.DRIVING:
            # 최종 감지 플래그가 True이고, `stop_sign_passed_flag`가 False인 경우에만 정지 로직 시작
            if self.stop_sign_detected_in_frame and not self.stop_sign_passed_flag: 
                new_state = StopSignState.APPROACHING_SIGN
                self.get_logger().info("🚦 정지 표지판 감지됨! APPROACHING_SIGN으로 상태 전환.")
                self.publish_stop_signal(True)
                self.is_approaching_active = True # 접근/정지 상태 진입
            elif self.stop_sign_passed_flag:
                # `stop_sign_passed_flag`가 True이면 감지되어도 무시하고 DRIVING 유지
                self.get_logger().debug("`stop_sign_passed_flag`가 활성화되어 정지 표지판 감지 무시. DRIVING 유지.")

        # 현재 상태가 APPROACHING_SIGN일 때
        elif self.current_state == StopSignState.APPROACHING_SIGN:
            if self.stop_sign_detected_in_frame: 
                new_state = StopSignState.STOPPED_AT_SIGN
                self.get_logger().info("🛑 정지 표지판에 도달했습니다. STOPPED_AT_SIGN으로 상태 전환.")
                self.stop_timer = self.create_timer(self.stop_duration_sec, self.stop_timer_callback)
                self.publish_stop_signal(True)
            else: # 감지 플래그가 False로 바뀌면 (표지판이 시야에서 사라지거나 감지 실패)
                new_state = StopSignState.DRIVING
                self.get_logger().warn("⚠️ APPROACHING_SIGN 도중 정지 표지판 감지가 끊겼습니다. DRIVING으로 복귀.")
                self.publish_stop_signal(False)
                self.is_approaching_active = False # 접근/정지 상태 해제
                if self.stop_timer:
                    self.stop_timer.cancel()
                    self.stop_timer = None

        # 현재 상태가 STOPPED_AT_SIGN일 때 (타이머 만료 대기)
        elif self.current_state == StopSignState.STOPPED_AT_SIGN:
            pass # 이 상태에서는 타이머가 만료될 때까지 대기

        # 현재 상태가 RESUMING_DRIVING일 때 (정지 후 재출발)
        elif self.current_state == StopSignState.RESUMING_DRIVING:
            new_state = StopSignState.DRIVING
            self.get_logger().info("🟢 주행 재개. DRIVING으로 상태 전환.")
            self.publish_stop_signal(False)
            self.is_approaching_active = False # 접근/정지 상태 해제

            # --- (추가) `stop_sign_passed_flag`를 True로 설정하고, 일정 시간 후 리셋 타이머 시작 ---
            self.stop_sign_passed_flag = True # 플래그를 True로 설정하여 일시적으로 감지 무시
            self.get_logger().info(f"✨ `stop_sign_passed_flag` 활성화 (다음 {self.reset_flag_delay_sec}초 동안 정지 표지판 감지 무시).")
            
            # 이전에 리셋 타이머가 동작 중이었다면 취소 (중복 실행 방지)
            if self.reset_passed_flag_timer is not None:
                self.reset_passed_flag_timer.cancel()
            
            # `reset_flag_delay_sec` 후에 `stop_sign_passed_flag`를 False로 리셋하는 타이머 생성
            self.reset_passed_flag_timer = self.create_timer(self.reset_flag_delay_sec, self.reset_stop_sign_passed_flag_callback)

        # 상태가 변경되었다면 현재 상태 업데이트 및 로깅
        if new_state != self.current_state:
            self.current_state = new_state
            self.get_logger().info(f"상태 변경됨: {self.current_state.name}")

    def publish_stop_signal(self, is_stop_active: bool):
        """정지 신호 및 정지 상태를 ROS 토픽으로 발행합니다."""
        bool_msg = Bool()
        bool_msg.data = is_stop_active

        int_msg = Int8()
        # ControlLane 노드가 사용하는 Int8 값과 일치하도록 조정 (Enum 값 사용)
        if is_stop_active:
            int_msg.data = StopSignState.APPROACHING_SIGN.value 
        else:
            int_msg.data = StopSignState.DRIVING.value

        self.stop_sign_pub.publish(bool_msg)
        self.stop_sign_state_pub.publish(int_msg)

        self.get_logger().info(f"📡 /stop_sign_detected 발행: {is_stop_active}, /stop_sign_state 발행: {int_msg.data} ({StopSignState(int_msg.data).name})")

    def stop_timer_callback(self):
        """정지 타이머가 만료되었을 때 호출되는 콜백 함수."""
        self.get_logger().info("✅ 정지 시간이 경과했습니다. 주행 재개 준비 중.")
        self.current_state = StopSignState.RESUMING_DRIVING
        
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

    # --- (새로 추가된 콜백 함수) `stop_sign_passed_flag` 리셋 타이머 콜백 ---
    def reset_stop_sign_passed_flag_callback(self):
        """`stop_sign_passed_flag`를 False로 리셋하는 콜백 함수."""
        self.get_logger().info("✅ `stop_sign_passed_flag` 리셋. 다시 정지 표지판 감지 및 정지 가능.")
        self.stop_sign_passed_flag = False
        
        # 타이머 완료 후에는 타이머 객체 정리
        if self.reset_passed_flag_timer:
            self.reset_passed_flag_timer.cancel()
            self.reset_passed_flag_timer = None


def main(args=None):
    rclpy.init(args=args)
    node = DetectStopSign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트(SIGINT) 감지. 노드 종료.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()