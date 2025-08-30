import socket
import json
import time
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# 初始化 SDK
network_interface = "eth0"
ChannelFactoryInitialize(0, network_interface)
client = SportClient()
client.SetTimeout(10.0)
client.Init()
print("[SDK] SportClient initialized")

# UDP 接收
UDP_IP = "127.0.0.2"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"[SDK] Listening on UDP {UDP_IP}:{UDP_PORT}")

while True:
    data, addr = sock.recvfrom(1024)
    try:
        cmd = json.loads(data.decode())
        vx = cmd.get("vx", 0.0)
        vy = cmd.get("vy", 0.0)
        wz = cmd.get("wz", 0.0)
        print(f"[SDK] Received cmd: vx={vx}, vy={vy}, wz={wz}")
        client.Move(vx, vy, wz)
    except Exception as e:
        print(f"[SDK] Error: {e}")
