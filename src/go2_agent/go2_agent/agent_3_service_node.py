#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray
from go2_agent.td3_algrithm import Customed_TD3

# ===== 模型编号对应路径 =====
from ament_index_python.packages import get_package_share_directory
PKG = "go2_agent"
SHARE_DIR = get_package_share_directory(PKG)

MODEL_PATHS = {
    "0": "model/td3_21/stage7_agent_ep14800.pt",
    "1": "model/td3_21/stage7_agent_ep21000.pt",
    "2": "model/td3_21/stage7_agent_ep22800.pt",
    "3": "model/td3_22/stage7_agent_ep24100.pt",
    "4": "model/td3_23/stage7_agent_ep24100.pt", #19200
    "5": "model/td3_24/stage7_agent_ep29700.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DRLInferenceNode(Node):
    def __init__(self, model_path):
        super().__init__('drl_inference_node')

        # 加载模型  
        self.get_logger().info(f"加载模型: {model_path}")
        self.model = Customed_TD3(device=DEVICE, sim_speed=1)
        ckpt = torch.load(model_path, map_location=DEVICE)
        self.model.actor.load_state_dict(ckpt['actor'])
        self.model.actor_target.load_state_dict(ckpt['actor_target'])
        self.model.critic.load_state_dict(ckpt['critic'])
        self.model.critic_target.load_state_dict(ckpt['critic_target'])
        for net in (self.model.actor, self.model.actor_target,
                    self.model.critic, self.model.critic_target):
            net.eval()

        self.step_counter = 0

        # ROS2 订阅和发布
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/drl_state',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(
            Float32MultiArray,
            '/drl_action',
            10
        )

    def listener_callback(self, msg):
        # 转成 numpy
        state = np.array(msg.data, dtype=np.float32)
        if state.ndim > 1:
            state = state.flatten()

        # 获取动作
        action = self.model.get_action(
            state,
            is_training=False,
            step=self.step_counter,
            visualize=False
        )
        self.step_counter += 1

        # 发布动作
        action_msg = Float32MultiArray()
        action_msg.data = list(map(float, action))
        self.publisher.publish(action_msg)
        self.get_logger().info(f"Action: {action_msg.data}")

def main(args=None):
    print("可用模型：")
    for k, v in MODEL_PATHS.items():
        print(f"{k}: {v}")
    model_choice = input("请输入模型编号: ").strip()
    if model_choice not in MODEL_PATHS:
        raise ValueError(f"模型编号 {model_choice} 不存在！可选: {list(MODEL_PATHS.keys())}")
    model_path = MODEL_PATHS[model_choice]

    rclpy.init(args=args)
    node = DRLInferenceNode(model_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
