#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENV:
- INFER_URL=http://<server>:5000/infer
- SCAN_TOPIC=/processed_scan
- ODOM_TOPIC=/odom
- REQUIRE_GOAL=1 | 0
- AUTO_ARM=0 | 1                 # 1=无goal也可动 armed+have_goal
- PRINT_INFER=0/1, PRINT_PREP=0/1
- CLIENT_TAG=jetson, CLIENT_LOG_DIR=./logs, CLIENT_FLUSH_EVERY=20
- TIME_TZ=Europe/Helsinki, HTTP_TIMEOUT=0.6
- FAST_BIN=1                     # 1=二进制推理，0=JSON
"""

import os, csv, copy, math, time, json, pathlib
from datetime import datetime
from typing import Tuple
from zoneinfo import ZoneInfo

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Bool, Float32
from sensor_msgs.msg import LaserScan
import requests

# ===== 常量（与你原始逻辑保持一致） =====
NUM_SCAN_SAMPLES = 580
REAL_ARENA_LENGTH = 14.2
REAL_ARENA_WIDTH  = 14.2
REAL_LIDAR_DISTANCE_CAP   = 6.0
REAL_THRESHOLD_COLLISION  = 0.3
REAL_THRESHOLD_GOAL       = 0.20

REAL_SPEED_LINEAR_MAX   = 0.40
REAL_SPEED_ANGULAR_MAX  = 1.00

LINEAR, ANGULAR = 0, 1
ENABLE_BACKWARD = False

TIME_TZ = os.getenv("TIME_TZ", "Europe/Helsinki")
def now_iso() -> str:
    return datetime.now(ZoneInfo(TIME_TZ)).isoformat()

def euler_from_quaternion(quat) -> Tuple[float, float, float]:
    x, y, z, w = quat.x, quat.y, quat.z, quat.w
    sinr_cosp = 2 * (w*x + y*z); cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w*y - z*x); sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (w*z + x*y); cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

class TD3receiveNode(Node):
    def __init__(self):
        super().__init__('td3receive')
        self.get_logger().info("Initializing TD3 Inference Node (HTTP client)…")

        # ---- URLs ----
        self.http_url     = os.getenv("INFER_URL", "http://127.0.0.1:5000/infer")
        self.http_timeout = float(os.getenv("HTTP_TIMEOUT", "0.6"))
        self.fast_bin     = (os.getenv("FAST_BIN", "1") == "1")  # 默认启用二进制；FAST_BIN=0 回退 JSON

        # ---- HTTP 会话（JSON）----
        self.sess = requests.Session()
        self.sess.headers.update({"Content-Type": "application/json"})

        # ---- QoS ----
        qos_pub  = QoSProfile(depth=10)
        qos_scan = QoSProfile(depth=10)
        qos_scan.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_scan.durability  = QoSDurabilityPolicy.VOLATILE

        # === 手势总闸（Bool）===
        self.gate_topic = os.getenv("GATE_TOPIC", "/forward_or_stop")
        self.motion_enabled = True  # 默认允许移动
        self.create_subscription(Bool, self.gate_topic, self.gate_cb, 10)
        self.get_logger().info(f"[GATE] subscribe {self.gate_topic}, default motion_enabled={self.motion_enabled}")

        # ---- 发布/订阅 ----
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_pub)
        self.create_subscription(Pose, '/goal_pose', self.goal_callback, 10)

        odom_topic = os.getenv("ODOM_TOPIC", "/odom")
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.create_subscription(Odometry, '/utlidar/robot_odom', self.odom_callback, 10)  # 可选第二源
        self.create_subscription(LaserScan, '/scan', self.scan_simu_callback, 10)          # 可选仿真激光

        self.scan_topic = os.getenv("SCAN_TOPIC", "/processed_scan")
        self.create_subscription(Float32MultiArray, self.scan_topic, self.scan_callback, qos_profile=qos_scan)

        # ---- 定时器 ----
        self.timer = self.create_timer(0.1, self.control_loop)   # 10Hz 控制
        self.idle_timer = self.create_timer(1.0, self._idle_heartbeat)  # 1Hz 门禁心跳

        # ---- 运行时状态 ----
        self.total_steps   = 0
        self.local_step    = 0

        self.scan_ranges   = [1.0] * NUM_SCAN_SAMPLES
        self.obstacle_distance = float('inf')
        self.have_scan     = False

        self.goal_pose     = Pose()
        self.goal_x        = 0.0
        self.goal_y        = 0.0
        self.have_goal     = False
        self.new_goal      = False
        self.require_goal  = (os.getenv("REQUIRE_GOAL", "1") == "1")
        self.auto_arm      = (os.getenv("AUTO_ARM", "0") == "1")
        if self.auto_arm:
            self.have_goal = True; self.new_goal = True

        self.robot_x       = 0.0
        self.robot_y       = 0.0
        self.robot_heading = 0.0
        self.have_odom     = False

        # 里程累计
        self.robot_x_prev   = 0.0
        self.robot_y_prev   = 0.0
        self.total_distance = 0.0
        self._first_odom    = True

        self.goal_distance = float('inf')
        self.goal_angle    = 0.0

        self.prev_vx, self.prev_omega = 0.0, 0.0
        self.prev_action = [0.0, 0.0]
        self._printed_result = False

        self.armed         = self.auto_arm
        self.done          = False
        self._brake_ticks       = 0
        self._brake_ticks_hold  = 5   # 触发后保持零速若干周期（不改变你的逻辑）

        self.UNKNOWN, self.SUCCESS, self.COLLISION_WALL = 0, 1, 2
        self.succeed = self.UNKNOWN

        # ---- 时间戳（用于心跳诊断）----
        self.last_scan_recv_ns = 0
        self.last_odom_recv_ns = 0
        self._last_dist_print  = 0.0

        # ---- 打印开关 ----
        self.print_infer = (os.getenv("PRINT_INFER", "0") == "1")
        self.print_prep  = (os.getenv("PRINT_PREP",  "0") == "1")

        # ---- CSV（关键字段）----
        self.client_tag     = os.getenv("CLIENT_TAG", "jetson")
        self.client_log_dir = pathlib.Path(os.getenv("CLIENT_LOG_DIR", "./logs"))
        self.client_log_dir.mkdir(parents=True, exist_ok=True)
        self.client_run_id  = datetime.now(ZoneInfo(TIME_TZ)).strftime("%Y%m%d-%H%M%S") + f"-{self.client_tag}"
        self.client_csv_path = self.client_log_dir / f"client-{self.client_run_id}.csv"

        self._client_fields = [
            "ts","run_id","step","state_dim",
            "prep_ms","scan_to_infer_ms",
            "rtt_ms","net_ms_est","up_ms","down_ms",
            "req_bytes","resp_bytes",
            "ok","error"
        ]
        self._client_csv_file = open(self.client_csv_path, "w", newline="")
        self._client_csv_writer = csv.DictWriter(self._client_csv_file, fieldnames=self._client_fields)
        self._client_csv_writer.writeheader()
        self._client_rows_since_flush = 0
        self._flush_every = int(os.getenv("CLIENT_FLUSH_EVERY", "20"))
        self.get_logger().info(f"[JETSON-CSV] {self.client_csv_path}")
        self.get_logger().info(f"[CFG] infer_url={self.http_url}  scan_topic={self.scan_topic}  odom_topic={odom_topic}  require_goal={self.require_goal}  auto_arm={self.auto_arm}")

    # ---------- 订阅回调 ----------
    def gate_cb(self, msg: Bool):
        self.motion_enabled = bool(msg.data)
        self.get_logger().info(f"[GATE] motion_enabled={self.motion_enabled}")
        if not self.motion_enabled:
            self.cmd_vel_pub.publish(Twist())   # 立即刹车

    def _idle_heartbeat(self):
        if self._brake_ticks > 0:
            return
        if self.have_odom and self.have_scan and (not self.require_goal or self.new_goal) and self.armed:
            return
        now = time.time_ns()
        scan_age = (now - self.last_scan_recv_ns)/1e6 if self.last_scan_recv_ns else None
        odom_age = (now - self.last_odom_recv_ns)/1e6 if self.last_odom_recv_ns else None
        self.get_logger().info(
            f"[IDLE] have_odom={self.have_odom} (age_ms={None if odom_age is None else round(odom_age,1)}), "
            f"have_scan={self.have_scan} (age_ms={None if scan_age is None else round(scan_age,1)}), "
            f"armed={self.armed}, new_goal={self.new_goal}, require_goal={self.require_goal}"
        )

    def scan_simu_callback(self, msg: LaserScan):
        if not msg.ranges:
            return
        n = min(NUM_SCAN_SAMPLES, len(msg.ranges))
        min_norm = 1.0
        for i in range(n):
            v = np.clip(float(msg.ranges[i]) / REAL_LIDAR_DISTANCE_CAP, 0, 1)
            if i < NUM_SCAN_SAMPLES:
                self.scan_ranges[i] = v
            if v < min_norm:
                min_norm = v
        self.obstacle_distance = min_norm * REAL_LIDAR_DISTANCE_CAP
        self.have_scan = True
        self.last_scan_recv_ns = time.time_ns()

    def scan_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        if not data:
            return
        if len(data) != NUM_SCAN_SAMPLES:
            self.get_logger().warn(f"[SCAN] got {len(data)} but expected {NUM_SCAN_SAMPLES}; will pad/clip")
        self.scan_ranges = data[:NUM_SCAN_SAMPLES]
        if len(self.scan_ranges) < NUM_SCAN_SAMPLES:
            self.scan_ranges += [1.0] * (NUM_SCAN_SAMPLES - len(self.scan_ranges))
        self.obstacle_distance = min(self.scan_ranges) * REAL_LIDAR_DISTANCE_CAP
        self.last_scan_recv_ns = time.time_ns()
        self.have_scan = True

    def goal_callback(self, msg: Pose):
        self.goal_pose = msg
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal  = True
        self.have_goal = True
        self.armed     = True
        self.done      = False
        self.succeed   = self.UNKNOWN
        self._brake_ticks = 0

        # 复位里程与一次性打印
        self.total_distance = 0.0
        self._printed_result = False
        self.local_step = 0
        if self.have_odom:
            self.robot_x_prev, self.robot_y_prev = self.robot_x, self.robot_y
            self._first_odom = False
        else:
            self._first_odom = True

        self.get_logger().info(f"[GOAL] x={self.goal_x:.3f}, y={self.goal_y:.3f}")

    def odom_callback(self, msg: Odometry):
        self.have_odom = True
        self.last_odom_recv_ns = time.time_ns()
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        _, _, self.robot_heading = euler_from_quaternion(msg.pose.pose.orientation)

        # 目标几何
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        self.goal_distance = math.hypot(dx, dy)
        heading_to_goal = math.atan2(dy, dx)
        goal_angle = heading_to_goal - self.robot_heading
        while goal_angle > math.pi:  goal_angle -= 2*math.pi
        while goal_angle < -math.pi: goal_angle += 2*math.pi
        self.goal_angle = goal_angle

        # 行驶里程累计（相邻两帧欧氏距离；过滤异常跳变）
        if self._first_odom:
            self.robot_x_prev, self.robot_y_prev = self.robot_x, self.robot_y
            self._first_odom = False
        else:
            step = math.hypot(self.robot_x - self.robot_x_prev, self.robot_y - self.robot_y_prev)
            if math.isfinite(step) and step < 2.0:  # 简单防跳变
                self.total_distance += step
            self.robot_x_prev, self.robot_y_prev = self.robot_x, self.robot_y

        # 每增加 ≥0.5m 打一行（可改成 0.2）
        if self.total_distance - self._last_dist_print >= 0.5:
            self.get_logger().info(f"[TRAVEL] {self.total_distance:.2f} m")
            self._last_dist_print = self.total_distance

            self.have_odom = True
            self.last_odom_recv_ns = time.time_ns()
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y
            _, _, self.robot_heading = euler_from_quaternion(msg.pose.pose.orientation)

            # 目标几何  
            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            self.goal_distance = math.hypot(dx, dy)
            heading_to_goal = math.atan2(dy, dx)
            goal_angle = heading_to_goal - self.robot_heading
            while goal_angle > math.pi:  goal_angle -= 2*math.pi
            while goal_angle < -math.pi: goal_angle += 2*math.pi
            self.goal_angle = goal_angle

            # 行驶里程累计（相邻两帧欧氏距离；过滤异常跳变）
            if self._first_odom:
                self.robot_x_prev, self.robot_y_prev = self.robot_x, self.robot_y
                self._first_odom = False
            else:
                step = math.hypot(self.robot_x - self.robot_x_prev, self.robot_y - self.robot_y_prev)
                if math.isfinite(step) and step < 2.0:  # 简单防跳变
                    self.total_distance += step
                self.robot_x_prev, self.robot_y_prev = self.robot_x, self.robot_y
            if self.total_distance - self._last_dist_print >= 0.5:
                self.get_logger().info(f"[TRAVEL] {self.total_distance:.2f} m")
                self._last_dist_print = self.total_distance
    #  HTTP 推理 
    def infer_http(self, state: np.ndarray, prep_ms: float = None, scan_to_infer_ms: float = None) -> np.ndarray:
        cli_send_ns = time.time_ns()
        req_bytes = 0
        headers = {}
        try:
            if self.fast_bin:
                buf = np.asarray(state, dtype=np.float32).tobytes()
                headers = {"Content-Type": "application/octet-stream"}
                r = self.sess.post(self.http_url, data=buf, headers=headers, timeout=self.http_timeout)
                cli_recv_ns = time.time_ns()
                r.raise_for_status()
                ct = (r.headers.get("Content-Type", "") or "").lower()
                if ct.startswith("application/octet-stream"):
                    action = np.frombuffer(r.content, dtype=np.float32)
                else:
                    data = r.json(); action = np.array(data.get('action', self.prev_action), dtype=np.float32)
                req_bytes = len(buf); resp_bytes = len(r.content or b"")
            else:
                payload = {'state': np.asarray(state, dtype=np.float32).tolist()}
                body = json.dumps(payload, separators=(',', ':')).encode('utf-8')
                headers = {"Content-Type": "application/json"}
                r = self.sess.post(self.http_url, data=body, headers=headers, timeout=self.http_timeout)
                cli_recv_ns = time.time_ns()
                r.raise_for_status()
                data = r.json(); action = np.array(data.get('action', self.prev_action), dtype=np.float32)
                req_bytes = len(body); resp_bytes = len(r.content or b"")

            # 统计
            rtt_ms = (cli_recv_ns - cli_send_ns) / 1e6
            net_ms_est = rtt_ms
            total_b = max(1, req_bytes + resp_bytes)
            up_ms   = rtt_ms * (req_bytes  / total_b)
            down_ms = rtt_ms * (resp_bytes / total_b)

            if self.print_infer:
                self.get_logger().info(
                    f"[INFER] rtt={rtt_ms:.2f}ms up~{up_ms:.2f} down~{down_ms:.2f} "
                    f"state_dim={len(state)} action={action[:2].tolist()}"
                )

            row = {
                "ts": now_iso(), "run_id": self.client_run_id, "step": int(self.total_steps + 1),
                "state_dim": int(len(state)), "prep_ms": (float(f"{prep_ms:.3f}") if prep_ms is not None else None),
                "scan_to_infer_ms": float(f"{scan_to_infer_ms:.3f}") if scan_to_infer_ms is not None else None,
                "rtt_ms": float(f"{rtt_ms:.3f}"), "net_ms_est": float(f"{net_ms_est:.3f}"),
                "up_ms": float(f"{up_ms:.3f}"), "down_ms": float(f"{down_ms:.3f}"),
                "req_bytes": int(req_bytes), "resp_bytes": int(resp_bytes),
                "ok": True, "error": "",
            }
            self._client_csv_writer.writerow(row)
            self._client_rows_since_flush += 1
            if self._client_rows_since_flush >= self._flush_every:
                self._client_csv_file.flush(); os.fsync(self._client_csv_file.fileno())

            self.total_steps += 1
            self.prev_action = action.tolist()
            return action

        except Exception as e:
            cli_recv_ns = time.time_ns()
            rtt_ms = (cli_recv_ns - cli_send_ns) / 1e6
            self.get_logger().warn(f"[INFER][ERR] {e}")
            row = {
                "ts": now_iso(),
                "run_id": self.client_run_id,
                "step": int(self.total_steps + 1),
                "state_dim": int(len(state)),
                "prep_ms": (float(f"{prep_ms:.3f}") if prep_ms is not None else None),
                "scan_to_infer_ms": float(f"{scan_to_infer_ms:.3f}") if scan_to_infer_ms is not None else None,
                "rtt_ms": float(f"{rtt_ms:.3f}"),
                "net_ms_est": None, "up_ms": None, "down_ms": None,
                "req_bytes": int(req_bytes), "resp_bytes": 0,
                "ok": False, "error": str(e),
            }
            self._client_csv_writer.writerow(row)
            self._client_csv_file.flush(); os.fsync(self._client_csv_file.fileno())
            return np.array(self.prev_action, dtype=np.float32)

    # 
    def _publish_stop(self, reason: str):
        # 刹停几次，避免被后续控制覆盖
        t = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(t)
        self._brake_ticks = self._brake_ticks_hold
        self.done = True
        
        # —— 一次性打印（console + ROS logger）——
        if not getattr(self, "_printed_result", False):
            human = "GOAL REACHED " if reason == "goal_reached" else \
                    ("COLLISION " if reason == "collision" else f"STOP: {reason}")
            # 控制台（可留可去）
            print(
                f"\033[1;36m[RESULT] {human} | traveled={self.total_distance:.2f} m | "
                f"d_goal={self.goal_distance:.2f} m | d_obs_min={self.obstacle_distance:.2f} m\033[0m",
                flush=True
            )
            # ★ ROS 日志里也带上“本次任务行驶距离”
            self.get_logger().info(
                f"[STOP] reason={reason}  traveled={self.total_distance:.2f} m  "
                f"goal_d={self.goal_distance:.2f}  obs_d={self.obstacle_distance:.2f}"
            )
            self._printed_result = True

    def _check_stop_conditions(self) -> bool:
        # ★ 已经 done 就别再判定（防重复触发）★
        if self.done:
            return False
        if self.goal_distance < REAL_THRESHOLD_GOAL:
            self._publish_stop("goal_reached"); return True
        if self.obstacle_distance < REAL_THRESHOLD_COLLISION:
            self._publish_stop("collision"); return True
        return False


    def control_loop(self):
        # 已触发停止：持续零速并返回（避免被后续动作覆盖）
        if self.done:
            self.cmd_vel_pub.publish(Twist())
            return

        # 刹停保持窗口
        if self._brake_ticks > 0:
            self._brake_ticks -= 1
            self.cmd_vel_pub.publish(Twist())
            return

        # 数据/armed/goal 就绪性检查
        if not self.have_odom or not self.have_scan or (self.require_goal and not self.new_goal) or not self.armed:
            self.cmd_vel_pub.publish(Twist())
            return

        # 先给模型15步热身（你的原逻辑）
        self.local_step += 1
        if self.local_step > 15 and self._check_stop_conditions():
            return

        # 门禁：禁止则直接停
        if not self.motion_enabled:
            self.cmd_vel_pub.publish(Twist())
            return

        # ---- 构造状态并推理 ----
        t_p0 = time.perf_counter_ns()
        state = copy.deepcopy(self.scan_ranges)
        max_dist = math.sqrt(REAL_ARENA_LENGTH**2 + REAL_ARENA_WIDTH**2)
        state.append(float(np.clip(self.goal_distance / max_dist, 0.0, 1.0)))
        state.append(float(self.goal_angle) / math.pi)
        state.append(float(self.prev_vx / REAL_SPEED_LINEAR_MAX))
        state.append(float(self.prev_omega / REAL_SPEED_ANGULAR_MAX))
        prep_ms = (time.perf_counter_ns() - t_p0) / 1e6
        if (os.getenv("PRINT_PREP","0") == "1"):
            self.get_logger().info(f"[PREP] lidar_preprocess_ms={prep_ms:.3f}")

        scan_to_infer_ms = None
        if self.last_scan_recv_ns:
            scan_to_infer_ms = max(0.0, (time.time_ns() - self.last_scan_recv_ns) / 1e6)

        action = self.infer_http(np.array(state, dtype=np.float32), prep_ms=prep_ms, scan_to_infer_ms=scan_to_infer_ms)

        # 动作解码（保持你的映射）
        if ENABLE_BACKWARD:
            vx = float(action[LINEAR]) * REAL_SPEED_LINEAR_MAX
        else:
            vx = float((action[LINEAR] + 1.0) * 0.5) * REAL_SPEED_LINEAR_MAX
        wz = float(action[ANGULAR]) * REAL_SPEED_ANGULAR_MAX
        self.prev_vx, self.prev_omega = vx, wz

        twist = Twist(); twist.linear.x = vx; twist.angular.z = wz
        self.cmd_vel_pub.publish(twist)

    # ---------- 退出 ----------
    def destroy_node(self):
        try:
            if hasattr(self, "_client_csv_file") and self._client_csv_file:
                self._client_csv_file.flush(); os.fsync(self._client_csv_file.fileno()); self._client_csv_file.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TD3receiveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
