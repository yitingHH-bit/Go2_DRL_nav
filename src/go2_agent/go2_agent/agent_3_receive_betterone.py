#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接收端（ROS2 节点）— 精简且与“发布端 /infer + /gpu_stats”对齐的版本
- 仅从 /infer 读取 {"action": [...]}，保持兼容
- 记录雷达预处理耗时 prep_ms 到 client-*.csv
- 周期性拉取发布端 /gpu_stats 到 gpu-*.csv（GPU-only 指标）
- 串行控制，门禁：有 odom、有 scan、(可选)有 goal、armed 才会动
- 默认 SCAN_TOPIC=/processed_scan，消息类型 Float32MultiArray，长度=580 且已归一化到 [0,1]

环境变量（常用）
- INFER_URL:        http://<server>:5000/infer
- SCAN_TOPIC:       /processed_scan
- REQUIRE_GOAL:     1（默认；设 0 则无需发 goal 也会动）
- PRINT_INFER:      1（打印每帧 RTT 等）
- PRINT_PREP:       1（打印雷达预处理耗时）
- GPU_POLL_SEC:     2.0（拉取 /gpu_stats 的周期；设为 0 或负数可禁用）
- CLIENT_TAG:       jetson 或 edge（用于区分 CSV 文件前缀）
"""

import os
import csv
import copy
import math
import time
import pathlib
from datetime import datetime
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan  
import requests
import numpy as np
# ================== 可调常量 ==================
NUM_SCAN_SAMPLES = 580                 # 与模型训练对齐的激光采样数
REAL_ARENA_LENGTH = 8.2
REAL_ARENA_WIDTH  = 8.2
REAL_LIDAR_DISTANCE_CAP = 6.5         # 真实场景上限裁剪（米），scan 已归一化则仅用于碰撞距离估计
REAL_THRESHOLD_COLLISION = 0.3        # 碰撞半径（米）
REAL_THRESHOLD_GOAL      = 0.25        # 成功阈值（米）

REAL_SPEED_LINEAR_MAX   = 0.40         # 线速度上限（m/s）
REAL_SPEED_ANGULAR_MAX  = 1.00         # 角速度上限（rad/s）

LINEAR  = 0 
ANGULAR = 1 
ENABLE_BACKWARD = False                # 与训练时一致：False 则 action[0]∈[-1,1] 映射到 [0, v_max]

# ================== 工具函数 ==================
def euler_from_quaternion(quat) -> Tuple[float, float, float]:
    x, y, z, w = quat.x, quat.y, quat.z, quat.w
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

# ================== 主节点 ==================
class TD3receiveNode(Node):
    def __init__(self):
        super().__init__('td3receive')
        self.get_logger().info("Initializing TD3 Inference Node (lean receiver)...")

        # ---- 推断服务 URL ----
        self.http_url    = os.getenv("INFER_URL", "http://127.0.0.1:5000/infer")
        self.health_url  = self.http_url.replace("/infer", "/health")
        self.gpu_url     = self.http_url.replace("/infer", "/gpu_stats")

        # ---- QoS ----
        qos_pub = QoSProfile(depth=10)
        qos_scan = QoSProfile(depth=10)
        qos_scan.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_scan.durability  = QoSDurabilityPolicy.VOLATILE

        # ---- 发布器 ----
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_pub)

        # ---- 订阅器 ----
        self.create_subscription(Pose, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(Float32MultiArray, 'cmd_pose', self.cmd_pose_callback, 10)
        #self.create_subscription(Odometry, '/utlidar/robot_odom', self.odom_callback, 10)
        
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_simu_callback, 10)
        self.scan_topic = os.getenv("SCAN_TOPIC", "/processed_scan")
        self.scan_sub = self.create_subscription(
            Float32MultiArray,
            self.scan_topic,
            self.scan_callback,
            qos_profile=qos_scan,
        )

        # ---- 定时器（10Hz）----
        self.timer = self.create_timer(0.1, self.control_loop)

        # ---- 运行时状态 ----
        self.total_steps   = 0
        self.local_step    = 0
        self.is_training   = False
        self.visualize     = False

        # 传感器/目标/位姿缓存
        self.scan_ranges   = [REAL_LIDAR_DISTANCE_CAP] * NUM_SCAN_SAMPLES
        self.obstacle_distance = float('inf')
        self.have_scan     = False

        self.goal_pose     = Pose()
        self.goal_x        = 0.0
        self.goal_y        = 0.0
        self.have_goal     = False
        self.new_goal      = False
        self.require_goal  = (os.getenv("REQUIRE_GOAL", "1") == "1")

        self.robot_x       = 0.0
        self.robot_y       = 0.0
        self.robot_heading = 0.0
        self.have_odom     = False

        self.current_pose  = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [x,y,yaw]
        self.initial_pose  = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.goal_distance = float('inf')
        self.goal_angle    = 0.0

        self.prev_vx       = 0.0
        self.prev_omega    = 0.0
        self.prev_action   = [0.0, 0.0]

        # 武装/停止
        self.armed         = False
        self.done          = False
        self._brake_ticks       = 0
        self._brake_ticks_hold  = 5  # 0.5s 刹停保持

        # 成功/失败标志（可扩展）
        self.UNKNOWN, self.SUCCESS, self.COLLISION_WALL = 0, 1, 2
        self.succeed = self.UNKNOWN

        # ---- 客户端 CSV（每帧）----
        self.last_scan_recv_ns = 0

        self.client_tag     = os.getenv("CLIENT_TAG", "jetson")
        self.client_log_dir = pathlib.Path(os.getenv("CLIENT_LOG_DIR", "./logs"))
        self.client_log_dir.mkdir(parents=True, exist_ok=True)
        self.client_run_id  = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{self.client_tag}"
        self.client_csv_path = self.client_log_dir / f"client-{self.client_run_id}.csv"

        self._client_fields = [
            "ts","run_id","step","state_dim",
            "prep_ms","scan_to_infer_ms",                    # <--- 预处理耗时
            "rtt_ms","net_ms_est","up_ms","down_ms",
            "server_total_ms","server_parse_ms","server_model_ms","server_pack_ms",
            "req_bytes","resp_bytes",
            "srv_recv_unix_ns","srv_send_unix_ns",
            "cli_send_unix_ns","cli_recv_unix_ns",
            "ok","error"
        ]
        self._client_csv_file = open(self.client_csv_path, "w", newline="")
        self._client_csv_writer = csv.DictWriter(self._client_csv_file, fieldnames=self._client_fields)
        self._client_csv_writer.writeheader()
        self.get_logger().info(f"[JETSON-CSV] {self.client_csv_path}")

        # ---- GPU CSV（定时拉取 /gpu_stats）----
        self.gpu_csv_path = self.client_log_dir / f"gpu-{self.client_run_id}.csv"
        self._gpu_fields = [
            "ts","run_id",
            "device","input_dim","model_choice",
            "load_ckpt_ms","load_state_ms","total_load_ms",
            "warmup_ms_avg","infer_p50_ms","infer_p90_ms","infer_p99_ms",
            "nvml_util","nvml_mem_mb","nvml_power_w","nvml_temp_c",
            "torch_alloc_mb","torch_reserved_mb","torch_max_alloc_mb","peak_mb_after_warmup"
        ]
        self._gpu_csv_file = open(self.gpu_csv_path, "w", newline="")
        self._gpu_csv_writer = csv.DictWriter(self._gpu_csv_file, fieldnames=self._gpu_fields)
        self._gpu_csv_writer.writeheader()
        self.get_logger().info(f"[GPU-CSV] {self.gpu_csv_path}")

        gpu_poll_sec = float(os.getenv("GPU_POLL_SEC", "2.0"))
        if gpu_poll_sec > 0:
            self.gpu_timer = self.create_timer(gpu_poll_sec, self._poll_gpu_stats)
        else:
            self.gpu_timer = None

        # ---- 健康检查 ----
        try:
            h = requests.get(self.health_url, timeout=float(os.getenv("HEALTH_TIMEOUT", "0.7"))).json()
            self.get_logger().info(f"[HEALTH] {h}")
        except Exception as e:
            self.get_logger().warn(f"[HEALTH] 推理服务不可达：{e}")

        #simulation for test 
    def scan_simu_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            pass#print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
        # normalize laser values
        

        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
                self.scan_ranges[i] = np.clip(float(msg.ranges[i]) / REAL_LIDAR_DISTANCE_CAP, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= REAL_LIDAR_DISTANCE_CAP
        self.have_scan = True  # <--- 新增
        self.last_scan_recv_ns = time.time_ns()

        if self.obstacle_distance < REAL_THRESHOLD_COLLISION:
            print(f"  Collision is detected!: {self.obstacle_distance}")

    # --------------- 订阅回调 ---------------
    def scan_callback(self, msg: Float32MultiArray):
        if len(msg.data) != NUM_SCAN_SAMPLES:
            self.get_logger().warn(
                f"[SCAN] got {len(msg.data)} but expected {NUM_SCAN_SAMPLES}; will pad/clip later")
        # 归一化数组（0..1），转缓存
       
        self.scan_ranges = list(msg.data)[:NUM_SCAN_SAMPLES]
        if len(self.scan_ranges) < NUM_SCAN_SAMPLES:
            self.scan_ranges += [1.0] * (NUM_SCAN_SAMPLES - len(self.scan_ranges))
        # 最小距离估计（用于碰撞判定）
        self.obstacle_distance = min(self.scan_ranges) * REAL_LIDAR_DISTANCE_CAP
        self.last_scan_recv_ns = time.time_ns() # for measure lidar infer time 
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
        self.get_logger().info(f"[GOAL] x={self.goal_x:.3f}, y={self.goal_y:.3f}")

    def cmd_pose_callback(self, msg: Float32MultiArray):
        # 可选：相对命令 [dx, dy, dtheta]
        cmd = np.array(msg.data, dtype=np.float32)
        if cmd.size < 3:
            return
        # 用当前里程估计的位姿作为 current_pose
        self.current_pose = np.array([self.robot_x, self.robot_y, self.robot_heading], dtype=np.float32)
        if not hasattr(self, "initial_cmd_pose"):
            self.initial_cmd_pose = cmd - self.initial_pose
        rem = self.initial_cmd_pose - self.current_pose  # [dx, dy, dth]
        dx, dy = float(rem[0]), float(rem[1])
        yaw = self.robot_heading
        gx = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        gy = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        d  = max(1e-6, math.sqrt(gx*gx + gy*gy))
        self.goal_distance = d
        self.goal_angle = math.atan2(gy, gx)  # 机器人坐标系下
        self.have_goal = True
        self.new_goal  = True
        self.armed     = True
        self.done      = False
        self.succeed   = self.UNKNOWN
        self._brake_ticks = 0

    def odom_callback(self, msg: Odometry):
        self.have_odom = True
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        _, _, self.robot_heading = euler_from_quaternion(msg.pose.pose.orientation)

        # 更新全局误差
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        self.goal_distance = math.sqrt(dx*dx + dy*dy)
        heading_to_goal = math.atan2(dy, dx)
        goal_angle = heading_to_goal - self.robot_heading
        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi
        self.goal_angle = goal_angle

    # --------------- HTTP 推理 ---------------
    def infer_http(self, state: np.ndarray, prep_ms: float = None,scan_to_infer_ms: float = None) -> np.ndarray:
        payload = {'state': state.tolist()}
        cli_send_ns = time.time_ns()
        try:
            r = requests.post(self.http_url, json=payload, timeout=float(os.getenv("HTTP_TIMEOUT", "0.6")))
            cli_recv_ns = time.time_ns()
            r.raise_for_status()
            data = r.json()

            action = np.array(data.get('action', self.prev_action), dtype=np.float32)

            # 发布端“精简 /infer”通常不返回下面这些字段，保持兼容，取不到则 0
            parse_ms    = float(data.get('parse_ms', 0.0))
            model_ms    = float(data.get('model_ms', 0.0))
            total_ms    = float(data.get('server_total_ms', 0.0))
            pack_ms_est = float(data.get('pack_ms_est', 0.0))
            srv_recv_ns = int(data.get('srv_recv_unix_ns', 0) or 0)
            srv_send_ns = int(data.get('srv_send_unix_ns', 0) or 0)
            state_dim   = int(data.get('state_dim', len(state)))
            
            # RTT & 上下行估计
            rtt_ms = (cli_recv_ns - cli_send_ns) / 1e6
            net_ms_est = max(0.0, rtt_ms - total_ms)
            if srv_recv_ns and srv_send_ns:
                up_ms   = max(0.0, (srv_recv_ns - cli_send_ns) / 1e6)
                down_ms = max(0.0, (cli_recv_ns - srv_send_ns) / 1e6)
            else:
                half = net_ms_est / 2.0
                up_ms, down_ms = half, half

            if os.getenv("PRINT_INFER", "1") == "1":
                self.get_logger().info(
                    f"[INFER] rtt={rtt_ms:.2f}ms total={total_ms:.2f}ms up~{up_ms:.2f} down~{down_ms:.2f} "
                    f"state_dim={state_dim} action={action.tolist()}")

            # CSV 一帧
            row = {
                "ts": datetime.utcnow().isoformat(),
                
                "run_id": self.client_run_id,
                "step": int(self.total_steps + 1),
                "state_dim": int(state_dim),
                "prep_ms": (float(f"{prep_ms:.3f}") if prep_ms is not None else None),
                "scan_to_infer_ms": float(f"{scan_to_infer_ms:.3f}") if scan_to_infer_ms is not None else None,  # <--- 新增
                "rtt_ms": float(f"{rtt_ms:.3f}"),
                "net_ms_est": float(f"{net_ms_est:.3f}"),
                "up_ms": float(f"{up_ms:.3f}"),
                "down_ms": float(f"{down_ms:.3f}"),
                "server_total_ms": float(f"{total_ms:.3f}"),
                "server_parse_ms": float(f"{parse_ms:.3f}"),
                "server_model_ms": float(f"{model_ms:.3f}"),
                "server_pack_ms": float(f"{pack_ms_est:.3f}"),
                "req_bytes": len(r.request.body) if r.request.body else len(r.request.headers.get('Content-Length', '0')),
                "resp_bytes": len(r.content or b""),
                "srv_recv_unix_ns": int(srv_recv_ns),
                "srv_send_unix_ns": int(srv_send_ns),
                "cli_send_unix_ns": int(cli_send_ns),
                "cli_recv_unix_ns": int(cli_recv_ns),
                "ok": True,
                "error": "",
            }
            self._client_csv_writer.writerow(row)
            self._client_csv_file.flush()
            os.fsync(self._client_csv_file.fileno())

            self.total_steps += 1
            self.prev_action = action.tolist()
            return action

        except Exception as e:
            cli_recv_ns = time.time_ns()
            self.get_logger().warn(f"[INFER][ERR] {e}")
            row = {
                "ts": datetime.utcnow().isoformat(),
                "run_id": self.client_run_id,
                "step": int(self.total_steps + 1),
                "state_dim": int(len(state)),
                "prep_ms": (float(f"{prep_ms:.3f}") if prep_ms is not None else None),
                "rtt_ms": float((cli_recv_ns - cli_send_ns) / 1e6),
                "net_ms_est": None, "up_ms": None, "down_ms": None,
                "server_total_ms": None, "server_parse_ms": None, "server_model_ms": None, "server_pack_ms": None,
                "req_bytes": 0, "resp_bytes": 0,
                "srv_recv_unix_ns": 0, "srv_send_unix_ns": 0,
                "cli_send_unix_ns": int(cli_send_ns), "cli_recv_unix_ns": int(cli_recv_ns),
                "ok": False, "error": str(e),
            }
            self._client_csv_writer.writerow(row)
            self._client_csv_file.flush()
            os.fsync(self._client_csv_file.fileno())
            return np.array(self.prev_action, dtype=np.float32)

    # --------------- GPU 指标拉取 ---------------
    def _poll_gpu_stats(self):
        try:
            g = requests.get(self.gpu_url, timeout=0.5).json()
            load = g.get("load_ms", {}) or {}
            q    = g.get("infer_ms_quantiles", {}) or {}
            nv   = g.get("gpu_now", {}) or {}
            tm   = g.get("torch_mem", {}) or {}
            row = {
                "ts": datetime.utcnow().isoformat(),
                "run_id": self.client_run_id,
                "device": g.get("device"),
                "input_dim": g.get("input_dim"),
                "model_choice": g.get("model_choice"),
                "load_ckpt_ms": load.get("load_ckpt_ms"),
                "load_state_ms": load.get("load_state_ms"),
                "total_load_ms": load.get("total_load_ms"),
                "warmup_ms_avg": g.get("warmup_ms_avg"),
                "infer_p50_ms": q.get("p50"),
                "infer_p90_ms": q.get("p90"),
                "infer_p99_ms": q.get("p99"),
                "nvml_util": nv.get("util"),
                "nvml_mem_mb": nv.get("mem_mb"),
                "nvml_power_w": nv.get("power_w"),
                "nvml_temp_c": nv.get("temp_c"),
                "torch_alloc_mb": tm.get("alloc_mb"),
                "torch_reserved_mb": tm.get("reserved_mb"),
                "torch_max_alloc_mb": tm.get("max_alloc_mb") if tm else None,
                "peak_mb_after_warmup": tm.get("peak_mb_after_warmup"),
            }
            self._gpu_csv_writer.writerow(row)
            self._gpu_csv_file.flush()
            os.fsync(self._gpu_csv_file.fileno())
        except Exception as e:
            self.get_logger().warn(f"[GPU] pull fail: {e}")

    # --------------- 终止/刹停 ---------------
    def _check_stop_conditions(self) -> bool:
        if self.goal_distance < REAL_THRESHOLD_GOAL:
            self._publish_stop("goal_reached")
            return True
        if self.obstacle_distance < REAL_THRESHOLD_COLLISION:
            self._publish_stop("collision")
            return True
        return False

    def _publish_stop(self, reason: str):
        t = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(t)
        self._brake_ticks = self._brake_ticks_hold
        self.done = True
        self.get_logger().info(f"[STOP] reason={reason}  goal_d={self.goal_distance:.2f}  obs_d={self.obstacle_distance:.2f}")

    def stop_reset_robot(self, success: bool):
        self.cmd_vel_pub.publish(Twist())
        self.done = True
        self.armed = False
        self.have_goal = False
        self._publish_stop("success" if success else "collision")

    # --------------- 控制主循环 ---------------
    def control_loop(self):
        # 刹停保持
        if self._brake_ticks > 0:
            self._brake_ticks -= 1
            self.cmd_vel_pub.publish(Twist())
            return

        # 门禁：缺任一即静止
        if not self.have_odom or not self.have_scan or (self.require_goal and not self.new_goal) or not self.armed:
            self.cmd_vel_pub.publish(Twist())
            return

        # 到达/碰撞（15 帧宽限）
        self.local_step += 1
        if self.local_step > 15 and self._check_stop_conditions():
            return

        # —— 构造 state（与训练对齐：580 + 4）并计时 ——
        t_p0 = time.perf_counter_ns()
        state = copy.deepcopy(self.scan_ranges)  # 长度 580
        # 1) 归一化距离：以对角线作为最大尺度
        max_dist = math.sqrt(REAL_ARENA_LENGTH**2 + REAL_ARENA_WIDTH**2)
        state.append(float(np.clip(self.goal_distance / max_dist, 0.0, 1.0)))
        # 2) 角度 / π 到 [-1,1]
        state.append(float(self.goal_angle) / math.pi)
        # 3) 上一帧动作（近似归一化）
        state.append(float(self.prev_vx / REAL_SPEED_LINEAR_MAX))
        state.append(float(self.prev_omega / REAL_SPEED_ANGULAR_MAX))
        prep_ms = (time.perf_counter_ns() - t_p0) / 1e6
        if os.getenv("PRINT_PREP", "1") == "1":
            self.get_logger().info(f"[PREP] lidar_preprocess_ms={prep_ms:.3f}")

        # http 推理 
        scan_to_infer_ms = None
        if self.last_scan_recv_ns:
            scan_to_infer_ms = max(0.0, (time.time_ns() - self.last_scan_recv_ns) / 1e6)
        action = self.infer_http(np.array(state, dtype=np.float32), prep_ms=prep_ms,scan_to_infer_ms=scan_to_infer_ms)

        # 反归一化
        if ENABLE_BACKWARD:
            action_linear = float(action[LINEAR]) * REAL_SPEED_LINEAR_MAX
        else:
            action_linear = float((action[LINEAR] + 1.0) * 0.5) * REAL_SPEED_LINEAR_MAX
        action_angular = float(action[ANGULAR]) * REAL_SPEED_ANGULAR_MAX

        self.prev_vx, self.prev_omega = action_linear, action_angular

        # 发布速度
        twist = Twist()
        twist.linear.x  = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

    # --------------- 退出清理 ---------------
    def destroy_node(self):
        try:
            if hasattr(self, "_client_csv_file") and self._client_csv_file:
                self._client_csv_file.flush(); os.fsync(self._client_csv_file.fileno()); self._client_csv_file.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_gpu_csv_file") and self._gpu_csv_file:
                self._gpu_csv_file.flush(); os.fsync(self._gpu_csv_file.fileno()); self._gpu_csv_file.close()
        except Exception:
            pass
        super().destroy_node()

# ================== main ==================
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
