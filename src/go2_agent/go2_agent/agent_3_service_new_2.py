#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, make_response
import os, time, threading, logging, json, csv, queue, pathlib
import numpy as np
import torch
from collections import deque
from datetime import datetime

from go2_agent.td3_algrithm import Customed_TD3
 # 你的 TD3（Actor 里有 num_scan=580, extra_inputs=4）

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

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


WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "20"))
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.1"))
HIST_LEN = int(os.getenv("HIST_LEN", "2000"))
RUN_TAG = os.getenv("RUN_TAG", "cloud-5g")
LOG_DIR = pathlib.Path(os.getenv("LOG_DIR", "./logs")); LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{RUN_TAG}"
CSV_PATH = LOG_DIR / f"metrics-{RUN_ID}.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ===== 模型加载 =====
print("可用模型："); [print(f"{k}: {v}") for k, v in MODEL_PATHS.items()]
model_choice = os.getenv("MODEL_CHOICE") or input("请输入模型编号: ").strip()
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

# ===== 关键：从 Actor 读取“外部输入维度” =====
SCAN_DIM  = int(getattr(model.actor, "num_scan", 580))         # 训练里就是 580
EXTRA_DIM = int(getattr(model.actor, "extra_inputs", 4))       # 训练里就是 4
INPUT_DIM = SCAN_DIM + EXTRA_DIM                               # 580 + 4 = 584
print(f"[INPUT] actor.num_scan={SCAN_DIM}, actor.extra_inputs={EXTRA_DIM} -> INPUT_DIM={INPUT_DIM} (外部输入维度)")
print("      注意：模型内部卷积展平后会有更大的特征数（比如 2304），那是内部维度，和外部输入维度不同。")

# ===== 指标缓存 =====
lock = threading.Lock()
step_counter = 0
ema = {"parse": None, "model": None, "pack": None, "total": None}
hist_model = deque(maxlen=HIST_LEN)
hist_total = deque(maxlen=HIST_LEN)
_warmed_up = False

def _ema(prev, val): return val if prev is None else (1-EMA_ALPHA)*prev + EMA_ALPHA*val
def _pct(arr_like):
    if not arr_like: return {"p50": None, "p90": None, "p99": None}
    a = np.fromiter(arr_like, dtype=np.float64, count=len(arr_like))
    return {"p50": float(np.percentile(a,50)), "p90": float(np.percentile(a,90)), "p99": float(np.percentile(a,99))}
def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

# ===== GPU 指标（NVML）=====
gpu = {"util":None,"mem_mb":None,"power_w":None,"temp":None}
try:
    import pynvml
    pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    def _gpu_loop():
        while True:
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(_h)
                m = pynvml.nvmlDeviceGetMemoryInfo(_h)
                p = pynvml.nvmlDeviceGetPowerUsage(_h)/1000.0
                t = pynvml.nvmlDeviceGetTemperature(_h, pynvml.NVML_TEMPERATURE_GPU)
                gpu.update({
                    "util": int(u.gpu),
                    "mem_mb": int(m.used//(1024*1024)),
                    "power_w": float(p),
                    "temp": int(t)
                })
            except Exception:
                pass
            time.sleep(0.5)
    threading.Thread(target=_gpu_loop, daemon=True).start()
except Exception:
    print("[WARN] 未检测到 NVML，GPU 指标不可用（pip install nvidia-ml-py3）")

# ===== CSV 异步写入 =====
import queue as _q
csv_q:_q.Queue = _q.Queue()
FIELDS = ["ts","run_id","tag","step","state_dim",
          "parse_ms","model_ms","pack_ms_est","server_total_ms",
          "gpu_util","gpu_mem_mb","gpu_power_w","gpu_temp_c",
          "srv_recv_unix_ns","srv_send_unix_ns","req_bytes"]
def _csv_worker():
    with open(CSV_PATH,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        while True:
            row = csv_q.get()
            if row is None: break
            w.writerow(row); f.flush()
threading.Thread(target=_csv_worker, daemon=True).start()
print(f"[METRICS] CSV -> {CSV_PATH}")

# ===== Prometheus（可选）=====
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    REQ = Counter("infer_requests_total","Total inference requests",["tag"])
    ERR = Counter("infer_errors_total","Total inference errors",["tag"])
    H_PARSE = Histogram("infer_parse_ms","Parse time (ms)",["tag"],buckets=(.1,.2,.5,1,2,5,10,20,50,100,200))
    H_MODEL = Histogram("infer_model_ms","Model time (ms)",["tag"],buckets=(.1,.2,.5,1,2,5,10,20,50,100,200))
    H_TOTAL = Histogram("infer_total_ms","Server total (ms)",["tag"],buckets=(.5,1,2,5,10,20,50,100,200,500))
    G_U = Gauge("gpu_util_percent","GPU util %",["tag"])
    G_M = Gauge("gpu_mem_used_mb","GPU mem MB",["tag"])
    G_P = Gauge("gpu_power_w","GPU power W",["tag"])
    G_T = Gauge("gpu_temp_c","GPU temp C",["tag"])
    PROM = True
except Exception:
    PROM = False
    print("[WARN] prometheus_client 未安装，/metrics 会返回 501")

def _prom_gpu():
    if not PROM: return
    if gpu["util"] is not None:
        G_U.labels(RUN_TAG).set(gpu["util"]); G_M.labels(RUN_TAG).set(gpu["mem_mb"] or 0)
        G_P.labels(RUN_TAG).set(gpu["power_w"] or 0.0); G_T.labels(RUN_TAG).set(gpu["temp"] or 0)

# ===== 输入读取：对齐到 584（或 actor.num_scan+extra_inputs）=====
def read_state():
    ct = (request.content_type or "").lower()
    if "application/octet-stream" in ct:
        arr = np.frombuffer(request.data, dtype=np.float32)
    else:
        data = request.get_json(force=True, silent=False)
        arr = np.array(data["state"], dtype=np.float32)

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.size != INPUT_DIM:
        # 允许容错：若只发了 580，则用 4 个 0 补足；若多发则截断
        msg = f"[WARN] 收到 state.size={arr.size}，期望 {INPUT_DIM}（{SCAN_DIM}+{EXTRA_DIM}）。已自动对齐。"
        try: app.logger.warning(msg)
        except Exception: print(msg)
        if arr.size > INPUT_DIM:
            arr = arr[:INPUT_DIM]
        else:
            tmp = np.zeros((INPUT_DIM,), dtype=np.float32)
            tmp[:arr.size] = arr
            arr = tmp
    return arr

# ===== 推理实现 =====
def _infer_impl():
    global step_counter, ema, _warmed_up

    srv_recv_unix_ns = time.time_ns()
    try:
        req_bytes = int(request.headers.get('Content-Length', 0))
    except Exception:
        req_bytes = len(request.data or b"")

    t_total0 = time.perf_counter_ns()
    try:
        t_p0 = time.perf_counter_ns(); state = read_state(); t_p1 = time.perf_counter_ns()
    except Exception as e:
        if PROM: ERR.labels(RUN_TAG).inc()
        return jsonify({"error":str(e)}), 400

    # 预热：按外部输入维度（584）构造 dummy
    if not _warmed_up:
        dummy = np.zeros((INPUT_DIM,), dtype=np.float32)
        with torch.no_grad():
            for _ in range(WARMUP_STEPS):
                _sync(); _ = model.get_action(dummy, False, 0, False); _sync()
        _warmed_up=True; app.logger.info(f"[WARMUP] dim={INPUT_DIM} steps={WARMUP_STEPS}")

    # 纯前向计时
    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.no_grad():
            _sync(); starter.record()
            action = model.get_action(state, False, step_counter, False)
            ender.record(); torch.cuda.synchronize()
        model_ms = float(starter.elapsed_time(ender))
    else:
        t_m0=time.perf_counter()
        with torch.no_grad(): action = model.get_action(state, False, step_counter, False)
        t_m1=time.perf_counter(); model_ms=(t_m1-t_m0)*1000.0

    step_counter += 1
    if isinstance(action, np.ndarray): action = action.tolist()
    elif torch.is_tensor(action): action = action.detach().cpu().numpy().tolist()

    parse_ms = (t_p1 - t_p0)/1e6
    t_total1 = time.perf_counter_ns()
    total_ms = (t_total1 - t_total0)/1e6
    pack_ms  = max(total_ms - parse_ms - model_ms, 0.0)

    with lock:
        ema["parse"]=_ema(ema["parse"],parse_ms); ema["model"]=_ema(ema["model"],model_ms)
        ema["pack"]=_ema(ema["pack"],pack_ms);    ema["total"]=_ema(ema["total"],total_ms)
        hist_model.append(model_ms); hist_total.append(total_ms)

    if PROM:
        REQ.labels(RUN_TAG).inc()
        H_PARSE.labels(RUN_TAG).observe(parse_ms); H_MODEL.labels(RUN_TAG).observe(model_ms)
        H_TOTAL.labels(RUN_TAG).observe(total_ms); _prom_gpu()

    srv_send_unix_ns = time.time_ns()

    # CSV
    csv_q.put({
        "ts": datetime.utcnow().isoformat(), "run_id": RUN_ID, "tag": RUN_TAG, "step": step_counter,
        "state_dim": INPUT_DIM, "parse_ms": round(parse_ms,3), "model_ms": round(model_ms,3),
        "pack_ms_est": round(pack_ms,3), "server_total_ms": round(total_ms,3),
        "gpu_util": gpu["util"], "gpu_mem_mb": gpu["mem_mb"], "gpu_power_w": gpu["power_w"], "gpu_temp_c": gpu["temp"],
        "srv_recv_unix_ns": srv_recv_unix_ns, "srv_send_unix_ns": srv_send_unix_ns, "req_bytes": req_bytes,
    })

    payload = {
        "action": action,
        "parse_ms": round(parse_ms,3), "model_ms": round(model_ms,3), "server_total_ms": round(total_ms,3),
        "ema": {k:(None if v is None else round(v,3)) for k,v in ema.items()},
        "quantiles": {"model_ms": _pct(hist_model), "total_ms": _pct(hist_total)},
        "gpu": gpu, "device": str(DEVICE), "run_id": RUN_ID, "tag": RUN_TAG,
        "state_dim": INPUT_DIM, "step": int(step_counter),
        "srv_recv_unix_ns": int(srv_recv_unix_ns),
        "srv_send_unix_ns": int(srv_send_unix_ns),
        "req_bytes": int(req_bytes),
    }

    resp = make_response(json.dumps(payload),200)
    resp.headers["Content-Type"]="application/json"
    resp.headers["Server-Timing"]=f"parse;dur={payload['parse_ms']}, model;dur={payload['model_ms']}, total;dur={payload['server_total_ms']}"
    resp.headers["X-Srv-Recv-UNIX-NS"] = str(srv_recv_unix_ns)
    resp.headers["X-Srv-Send-UNIX-NS"] = str(srv_send_unix_ns)
    return resp

# ===== 路由 =====
@app.post("/infer")
def infer_json(): return _infer_impl()

@app.post("/infer_bin")
def infer_bin():  return _infer_impl()

@app.get("/stats")
def stats():
    with lock:
        out = {"ema":{k:(None if v is None else round(v,3)) for k,v in ema.items()},
               "q_model":_pct(hist_model),"q_total":_pct(hist_total),
               "gpu":gpu,"csv_path":str(CSV_PATH),"run_id":RUN_ID,"tag":RUN_TAG,
               "expected_state_dim": INPUT_DIM, "scan_dim": SCAN_DIM, "extra_dim": EXTRA_DIM}
    return jsonify(out)

@app.get("/metrics")
def prom_metrics():
    if not PROM: return "prometheus_client not installed", 501
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.get("/health")
def health(): return jsonify({"status":"ok","device":str(DEVICE),"gpu":gpu,"csv":str(CSV_PATH),
                             "expected_state_dim": INPUT_DIM})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
