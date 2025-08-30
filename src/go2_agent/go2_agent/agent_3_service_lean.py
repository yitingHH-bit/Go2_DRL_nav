#!/usr/bin/env python3
from flask import Flask, request, jsonify
import torch
import numpy as np
from go2_agent.td3_algrithm import Customed_TD3

app = Flask(__name__)
from ament_index_python.packages import get_package_share_directory
PKG = "go2_agent"
SHARE_DIR = get_package_share_directory(PKG)

MODEL_PATHS = {
    "0": "model/td3_21_stage_7/stage7_agent_ep14800.pt",
    "1": "model/td3_21_stage_7/stage7_agent_ep21000.pt",
    "2": "td3_21_stage_7/stage7_agent_ep22800.pt",
    "3": "td3_22_stage_7/stage7_agent_ep24100.pt",
    "4": "td3_23_stage_7/stage7_agent_ep24100.pt", #19200
    "5": "td3_24_stage_7/stage7_agent_ep29700.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 启动时选择模型 ======
print("可用模型：")
for k, v in MODEL_PATHS.items():
    print(f"{k}: {v}")
model_choice = input("请输入模型编号: ").strip()
if model_choice not in MODEL_PATHS:
    raise ValueError(f"模型编号 {model_choice} 不存在！可选: {list(MODEL_PATHS.keys())}")
MODEL_PATH = MODEL_PATHS[model_choice]

print(f"[INFO] 加载模型 {model_choice} -> {MODEL_PATH}")
model = Customed_TD3(device=DEVICE, sim_speed=1)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.actor.load_state_dict(ckpt['actor'])
model.actor_target.load_state_dict(ckpt['actor_target'])
model.critic.load_state_dict(ckpt['critic'])
model.critic_target.load_state_dict(ckpt['critic_target'])
for net in (model.actor, model.actor_target, model.critic, model.critic_target):
    net.eval()

step_counter = 0

@app.route('/infer', methods=['POST'])
def infer():
    global step_counter
    data = request.get_json()
    state = np.array(data['state'], dtype=np.float32)

    if state.ndim > 1:
        state = state.flatten()

    action = model.get_action(
        state,
        is_training=False,
        step=step_counter,
        visualize=False
    )
    step_counter += 1

    return jsonify({'action': action})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
