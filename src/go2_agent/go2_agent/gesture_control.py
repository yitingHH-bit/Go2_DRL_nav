#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # 相机常用 QoS（BEST_EFFORT）

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import cv2
import mediapipe as mp


class GestureController(Node):
    def __init__(self):
        super().__init__('gesture_controller')

        # --- 参数 ---
        self.image_topic = self.declare_parameter(
            'image_topic', '/camera/color/image_raw'  # 视你的系统可改为 /camera/camera/color/image_raw
        ).get_parameter_value().string_value
        self.enable_viz = self.declare_parameter('enable_viz', True).get_parameter_value().bool_value
        self.enable_cmd_vel = self.declare_parameter('enable_cmd_vel', False).get_parameter_value().bool_value

        # 锁存相关参数
        self.latch_enabled = self.declare_parameter('latch_enabled', True).get_parameter_value().bool_value
        self.latched_state = self.declare_parameter('latched_initial', False).get_parameter_value().bool_value  # 启动默认 False=停
        self.idle_auto_stop_ms = float(self.declare_parameter('idle_auto_stop_ms', 0).get_parameter_value().integer_value)
        # idle_auto_stop_ms=0 表示禁用“无手超时自动停”；否则超过该毫秒没看到手就强制 False

        # --- 发布器 ---
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_flag = self.create_publisher(Bool, '/forward_or_stop', 10)
        self.pub_use_gesture = self.create_publisher(Bool, '/use_gesture', 10)

        # --- 订阅图像（相机 QoS） ---
        self.sub_img = self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)

        # --- CvBridge & MediaPipe ---
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 记录最近一次看到手的时间
        self.last_hand_seen_ts = time.monotonic()

        self.get_logger().info(
            f"GestureController started. Subscribing: {self.image_topic} | viz={self.enable_viz} "
            f"| enable_cmd_vel={self.enable_cmd_vel} | latch={self.latch_enabled} "
            f"| init_state={self.latched_state} | idle_auto_stop_ms={self.idle_auto_stop_ms}"
        )

    # ----------------- 图像回调 -----------------
    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge 转换失败: {e}')
            return

        # 镜像翻转（更符合人面对摄像头的直觉）
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # 先把“将要发布”的状态设为“当前锁存状态”
        current_state = bool(self.latched_state)
        cmd = Twist()
        used_gesture = False
        detected_hand = False

        # —— 手势识别 ——（只处理一只手）
        if results.multi_hand_landmarks:
            detected_hand = True
            self.last_hand_seen_ts = time.monotonic()

            for hand_landmarks in results.multi_hand_landmarks:
                # 画关键点（可选）
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # 指尖索引（参照 MediaPipe Hands）
                tip_ids = [4, 8, 12, 16, 20]
                fingers_up = []

                # 拇指（水平判断）——左右手/镜像可能需要依场景调整
                fingers_up.append(hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x)

                # 其余 4 指（竖直判断）
                for i in range(1, 5):
                    fingers_up.append(hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y)

                total_fingers = fingers_up.count(True)

                # === 手势 → 状态变更（仅对 ✊ / 🖐 生效；其它手势不改变锁存） ===
                if total_fingers == 0:
                    # ✊ Fist → 前进(True)
                    if self.latch_enabled:
                        self.latched_state = True
                        current_state = True
                    else:
                        current_state = True
                    used_gesture = True
                    if self.enable_cmd_vel:
                        cmd.linear.x = 0.3
                    self.get_logger().info("✊ Fist → forward (latched=True)")
                elif total_fingers == 5:
                    # 🖐 Open palm → 停止(False)
                    if self.latch_enabled:
                        self.latched_state = False
                        current_state = False
                    else:
                        current_state = False
                    used_gesture = True
                    if self.enable_cmd_vel:
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
                    self.get_logger().info("🖐 Open palm → stop (latched=False)")
                else:
                    # 其它/未知 → 不改锁存；只打印
                    self.get_logger().info("✋ Other/unknown → keep last state")
                break  # 单手就够了

        # —— 无手可见时的处理：保持锁存；可选超时自动停 —— 
        if not detected_hand and self.idle_auto_stop_ms > 0:
            dt_ms = (time.monotonic() - self.last_hand_seen_ts) * 1000.0
            if dt_ms >= self.idle_auto_stop_ms:
                if self.latch_enabled:
                    if self.latched_state:  # 只有在当前是 True 时打印一次
                        self.get_logger().warn(f"No hand for {dt_ms:.0f}ms → auto stop")
                    self.latched_state = False
                current_state = False

        # 发布 forward_or_stop（Bool）—— 始终发布“当前锁存/计算后的状态”
        self.pub_flag.publish(Bool(data=bool(current_state)))
        # “正在用手势控制”的标志（可选）
        # self.pub_use_gesture.publish(Bool(data=used_gesture))

        # 可选发布 /cmd_vel（默认关闭）
        # if self.enable_cmd_vel:
        #     self.pub_cmd.publish(cmd)

        # 可视化
        if self.enable_viz:
            try:
                cv2.imshow('Gesture Controller (RealSense topic)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rclpy.shutdown()
            except cv2.error:
                # 无显示环境时自动关闭可视化
                self.enable_viz = False

    def destroy_node(self):
        try:
            self.hands.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GestureController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
