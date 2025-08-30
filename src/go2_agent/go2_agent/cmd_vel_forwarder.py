import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket
import json

UDP_IP = "127.0.0.2"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class CmdVelForwarder(Node):
    def __init__(self):
        super().__init__("cmd_vel_forwarder")
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.get_logger().info("CmdVelForwarder ready, forwarding to SDK")

    def cmd_callback(self, msg):
        cmd = {"vx": msg.linear.x, "vy": msg.linear.y, "wz": msg.angular.z}
        sock.sendto(json.dumps(cmd).encode(), (UDP_IP, UDP_PORT))
        self.get_logger().info(f"Forwarded cmd: {cmd}")

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelForwarder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
