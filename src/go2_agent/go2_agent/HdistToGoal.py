# 现在的逻辑问题是，我们只发布了前方位置是 距离机器人，2 米，所以数据转换的时候，只要x有值就可以，
# 因为我会保持人距离机器人1.5～2米的位置，，超过或者小于这个数字都过滤调

#!/usr/bin/env python3  
# hdist_to_goal.py
import math, time   
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose

def yaw_from_quat(qz: float, qw: float) -> float:
    # 平面 yaw = atan2(2*w*z, 1 - 2*z^2)
    return math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))

def quat_from_yaw(yaw: float):
    s, c = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
    return (0.0, 0.0, s, c)  # x, y, z, w

class HdistToGoal(Node):
    def __init__(self):
        super().__init__('hdist_to_goal')

        # === 参数 ===
        self.declare_parameter('publish_to', 'goal_pose')   # 'goal_pose' | 'cmd_pose'
        self.declare_parameter('min_d', 1.5)                # 只接受的最小前向距离
        self.declare_parameter('max_d', 2.0)                # 只接受的最大前向距离
        self.declare_parameter('min_publish_interval_ms', 300)  # 最小发布间隔
        self.declare_parameter('min_goal_move', 0.10)       # 新旧目标欧氏距离小于该值则不重复发

        # === 状态 ===
        self.last_odom         = None
        self.last_odom_ns      = 0
        self.last_d            = None
        self.last_d_ns         = 0
        self.last_goal_xy      = None
        self.last_publish_ns   = 0

        # === 发布器 ===
        self.pub_cmd_pose  = self.create_publisher(Float32MultiArray, '/cmd_pose', 10)
        self.pub_goal_pose = self.create_publisher(Pose, '/goal_pose', 10)

        # === 订阅器：两路里程计、两路距离 ===
        self.create_subscription(Odometry, '/odom',                 self.odom_cb, 10)
        self.create_subscription(Odometry, '/utlidar/robot_odom',   self.odom_cb, 10)
        self.create_subscription(Float32, '/human_distance',        self.hdist_cb, 10)
        self.create_subscription(Float32, '/hdist',                 self.hdist_cb, 10)

        # 心跳
        self.create_timer(1.0, self.heartbeat)

        to = self.get_parameter('publish_to').get_parameter_value().string_value
        self.get_logger().info(f"[init] listen [/human_distance,/hdist] + [/odom,/utlidar/robot_odom] -> publish '{to}'")

    # 
    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = yaw_from_quat(qz, qw)
        self.last_odom = (x, y, yaw)
        self.last_odom_ns = time.time_ns()

    def hdist_cb(self, msg: Float32):
        d = float(msg.data)
        self.last_d = d
        self.last_d_ns = time.time_ns()

        # 前置条件：必须有里程计
        if self.last_odom is None:
            self.get_logger().warn("No odom yet; ignore distance.")
            return

        # 只接受 1.5~2.0 m
        min_d = float(self.get_parameter('min_d').value)
        max_d = float(self.get_parameter('max_d').value)
        if not math.isfinite(d) or d < min_d or d > max_d:
            self.get_logger().info(f"distance {d:.2f} m out of [{min_d:.2f},{max_d:.2f}]; skip.")
            return

        # 计算世界系目标（沿当前 yaw 前进 d 米）
        x, y, yaw = self.last_odom
        goal_x = x + d * math.cos(yaw)
        goal_y = y + d * math.sin(yaw)

        # 防抖 + 去重
        now_ns = time.time_ns()
        min_int_ms = float(self.get_parameter('min_publish_interval_ms').value)
        if self.last_publish_ns:
            age_ms = (now_ns - self.last_publish_ns) / 1e6
            if age_ms < min_int_ms:
                # 发布太频繁，丢弃
                return
        min_move = float(self.get_parameter('min_goal_move').value)
        if self.last_goal_xy is not None:
            dx = goal_x - self.last_goal_xy[0]
            dy = goal_y - self.last_goal_xy[1]
            if math.hypot(dx, dy) < min_move:
                # 目标变化太小，丢弃
                return

        # 发布
        publish_to = self.get_parameter('publish_to').get_parameter_value().string_value
        if publish_to == 'cmd_pose':
            m = Float32MultiArray()
            m.data = [goal_x, goal_y, math.cos(yaw), math.sin(yaw)]
            self.pub_cmd_pose.publish(m)
            self.get_logger().info(f"Published /cmd_pose -> [{goal_x:.3f}, {goal_y:.3f}, cos={math.cos(yaw):.3f}, sin={math.sin(yaw):.3f}]")
        else:
            _, _, pz, pw = quat_from_yaw(yaw)
            p = Pose()
            p.position.x = goal_x
            p.position.y = goal_y
            p.position.z = 0.0
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = pz
            p.orientation.w = pw
            self.pub_goal_pose.publish(p)
            self.get_logger().info(f"Published /goal_pose -> x={goal_x:.3f}, y={goal_y:.3f}, yaw={yaw:.3f} rad")

        self.last_goal_xy = (goal_x, goal_y)
        self.last_publish_ns = now_ns

    # 心跳：方便你看有没有拿到数据 & 新鲜度
    def heartbeat(self):
        now = time.time_ns()
        odom_age = (now - self.last_odom_ns) / 1e6 if self.last_odom_ns else None
        dist_age = (now - self.last_d_ns) / 1e6 if self.last_d_ns else None
        self.get_logger().info(
            f"[HB] have_odom={self.last_odom is not None} (age_ms={None if odom_age is None else round(odom_age,1)}), "
            f"have_dist={self.last_d is not None} (age_ms={None if dist_age is None else round(dist_age,1)}), "
            f"last_d={None if self.last_d is None else round(self.last_d,3)}"
        )

def main():
    rclpy.init()
    node = HdistToGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
