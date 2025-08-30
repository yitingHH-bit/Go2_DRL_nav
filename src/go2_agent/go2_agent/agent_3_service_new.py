#!/usr/bin/env python3
from flask import Flask, request, jsonify
import torch
import numpy as np
import time
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

# ====== 简单统计器（EMA）======
ema_model_ms = None
ema_total_ms = None
EMA_ALPHA = 0.1
_warmed_up = False
_warmup_dim = None

def _maybe_sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _ema_update(prev, val):
    return val if prev is None else (1.0 - EMA_ALPHA) * prev + EMA_ALPHA * val

@app.route('/infer', methods=['POST'])
def infer():
    global step_counter, ema_model_ms, ema_total_ms, _warmed_up, _warmup_dim

    t_total0 = time.perf_counter()

    data = request.get_json()
    state = np.array(data['state'], dtype=np.float32)
    if state.ndim > 1:
        state = state.flatten()

    # 第一次收到请求时，按实际输入维度做预热，避免首批抖动
    if not _warmed_up:
        _warmup_dim = int(state.size)
        dummy = np.zeros((_warmup_dim,), dtype=np.float32)
        with torch.no_grad():
            for _ in range(20):
                _maybe_sync_cuda()
                _ = model.get_action(dummy, is_training=False, step=0, visualize=False)
                _maybe_sync_cuda()
        _warmed_up = True
        print(f"[WARMUP] Done with dim={_warmup_dim}")

    # 纯模型前向计时（含 GPU 同步）
    _maybe_sync_cuda()
    t_model0 = time.perf_counter()
    with torch.no_grad():
        action = model.get_action(
            state,
            is_training=False,
            step=step_counter,
            visualize=False
        )
    _maybe_sync_cuda()
    t_model1 = time.perf_counter()

    step_counter += 1

    # 统计
    t_total1 = time.perf_counter()
    model_ms = (t_model1 - t_model0) * 1000.0
    total_ms = (t_total1 - t_total0) * 1000.0

    ema_model_ms = _ema_update(ema_model_ms, model_ms)
    ema_total_ms = _ema_update(ema_total_ms, total_ms)

    if step_counter % 50 == 0:
        print(f"[infer] model_ms: last={model_ms:.3f}  EMA={ema_model_ms:.3f} | total_ms: last={total_ms:.3f}  EMA={ema_total_ms:.3f}")

    # 确保可 JSON 化
    if isinstance(action, np.ndarray):
        action = action.tolist()
    elif torch.is_tensor(action):
        action = action.detach().cpu().numpy().tolist()

    return jsonify({
        'action': action,
        'model_ms': float(model_ms),
        'total_ms': float(total_ms),
        'ema_model_ms': float(ema_model_ms),
        'ema_total_ms': float(ema_total_ms),
    })


if __name__ == '__main__':
    # 生产环境建议用 gunicorn/uvicorn；这里先保持简单
    app.run(host='0.0.0.0', port=5000)
