#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading, numpy as np, torch
from flask import Flask, request, jsonify, Response
from ament_index_python.packages import get_package_share_directory
# ===== you can customed your own algri. =====

from .td3_algrithm import Customed_TD3


PKG = "go2_agent"

def _find_model_root() -> str:
   
    p = os.getenv("MODEL_ROOT")
    if p and os.path.isdir(p):
        return os.path.abspath(p)
    here = os.path.abspath(os.path.dirname(__file__))
    cand = os.path.join(here, "model")
    return cand if os.path.isdir(cand) else here

MODEL_ROOT = _find_model_root()
print(f"[BOOT] MODEL_ROOT={MODEL_ROOT}", flush=True)

MODEL_PATHS = {
    "1": os.path.join(MODEL_ROOT, "td3_21_stage_7", "stage7_agent_ep21000.pt"),
    "0": os.path.join(MODEL_ROOT, "td3_21_stage_7", "stage7_agent_ep14800.pt"),
    "2": os.path.join(MODEL_ROOT, "td3_21_stage_7", "stage7_agent_ep22800.pt"),
    "3": os.path.join(MODEL_ROOT, "td3_22_stage_7", "stage7_agent_ep24100.pt"),
    "4": os.path.join(MODEL_ROOT, "td3_23_stage_7", "stage7_agent_ep24100.pt"), #19200
    "5": os.path.join(MODEL_ROOT, "td3_24_stage_7", "stage7_agent_ep29700.pt"),
}

MODEL_CHOICE = os.getenv("MODEL_CHOICE", "1")
CPU_THREADS  = int(os.getenv("CPU_THREADS", "1"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "8"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(max(1, CPU_THREADS))
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if MODEL_CHOICE not in MODEL_PATHS:
    raise ValueError(f"MODEL_CHOICE={MODEL_CHOICE} no this modele , can choose : {list(MODEL_PATHS.keys())}")
MODEL_PATH = MODEL_PATHS[MODEL_CHOICE]
print(f"[LOAD] {MODEL_CHOICE} -> {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model = Customed_TD3(device=DEVICE, sim_speed=1)
model.actor.load_state_dict(ckpt["actor"])
model.actor_target.load_state_dict(ckpt["actor_target"])
model.critic.load_state_dict(ckpt["critic"])
model.critic_target.load_state_dict(ckpt["critic_target"])
for net in (model.actor, model.actor_target, model.critic, model.critic_target):
    net.eval()

# ===== dims =====
SCAN_DIM  = int(getattr(model.actor, "num_scan", 580))
EXTRA_DIM = int(getattr(model.actor, "extra_inputs", 4))
INPUT_DIM = SCAN_DIM + EXTRA_DIM
# output action
with torch.inference_mode():
    _probe = model.get_action(np.zeros(INPUT_DIM, np.float32), False, 0, False)
if isinstance(_probe, np.ndarray):
    ACTION_DIM = int(_probe.size)
elif torch.is_tensor(_probe):
    ACTION_DIM = int(_probe.numel())
else:
    ACTION_DIM = int(np.asarray(_probe, dtype=np.float32).size)
print(f"[INPUT] num_scan={SCAN_DIM}, extra_inputs={EXTRA_DIM} -> INPUT_DIM={INPUT_DIM}, ACTION_DIM={ACTION_DIM}")


MODEL_LOCK = threading.Lock()

ALLOW_PAD = bool(int(os.getenv("ALLOW_PAD", "0"))) 
def _align_state(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size != INPUT_DIM and not ALLOW_PAD:
        raise ValueError(f"obs dim={arr.size}, expected {INPUT_DIM}")
    if arr.size > INPUT_DIM:
        arr = arr[:INPUT_DIM]   
    elif arr.size < INPUT_DIM:  
        tmp = np.zeros((INPUT_DIM,), np.float32)
        tmp[:arr.size] = arr
        arr = tmp
    return np.ascontiguousarray(arr, dtype=np.float32)

def _infer_core(state: np.ndarray) -> np.ndarray:
    with MODEL_LOCK, torch.inference_mode():
        act = model.get_action(state, False, 0, False)
    if isinstance(act, np.ndarray):
        out = act.astype(np.float32, copy=False)
    elif torch.is_tensor(act):
        out = act.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        out = np.asarray(act, dtype=np.float32)
    if out.ndim > 1: out = out.reshape(-1)
    if out.size != ACTION_DIM:
        out2 = np.zeros((ACTION_DIM,), np.float32)
        out2[:min(ACTION_DIM, out.size)] = out[:min(ACTION_DIM, out.size)]
        out = out2
    return out

if WARMUP_STEPS > 0:
    dummy = np.zeros((INPUT_DIM,), np.float32)
    with MODEL_LOCK, torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            _ = model.get_action(dummy, False, 0, False)
    print(f"[WARMUP] steps={WARMUP_STEPS} done.")

# ===== Flask =====
app = Flask(__name__)

def _read_state() -> np.ndarray:
    ct = (request.content_type or "").lower()
    if "application/octet-stream" in ct:
        buf = request.get_data(cache=False)
        arr = np.frombuffer(buf, dtype=np.float32, count=-1)
    else:
        data = request.get_json(force=True, silent=False)  # {"state":[...]}
        arr = np.array(data["state"], dtype=np.float32, copy=False).reshape(-1)
    return _align_state(arr)

@app.post("/infer")
def infer_mix():
    """同一路由，二进制优先；否则 JSON 兼容"""
    state = _read_state()
    action = _infer_core(state)
    ct = (request.content_type or "").lower()
    if "application/octet-stream" in ct:
        return Response(action.tobytes(), mimetype="application/octet-stream")
    else:
        return jsonify({"action": action.tolist()})

@app.post("/infer_json")
def infer_json():
    """强制 JSON（调试/旧客户端显式使用）"""
    data = request.get_json(force=True, silent=False)
    state = _align_state(np.array(data["state"], dtype=np.float32, copy=False))
    action = _infer_core(state)
    return jsonify({"action": action.tolist()})

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "input_dim": INPUT_DIM,
        "action_dim": ACTION_DIM
    })
