import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty


class Go2ServiceClient(Node):
    def __init__(self):
        super().__init__('go2_service_client')

        # 改名为非保留名
        self.service_clients = {
            "1": ("stand_up", self.create_client(Empty, 'stand_up')),
            "2": ("lay_down", self.create_client(Empty, 'lay_down')),
            "3": ("recover_stand", self.create_client(Empty, 'recover_stand')),
            "4": ("damping", self.create_client(Empty, 'damping'))
        }

        for key, (name, client) in self.service_clients.items():
            self.get_logger().info(f"Waiting for '{name}' service...")
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(f"'{name}' service not available yet...")

        self.get_logger().info("All services available.")
        self.prompt_user()

    def prompt_user(self):
        print("\n选择一个动作控制机器人：")
        print("1 - 站起来 (stand_up)")
        print("2 - 躺下 (lay_down)")
        print("3 - 恢复站立 (recover_stand)")
        print("4 - 阻尼模式 (damping)")
        print("q - 退出")
        while rclpy.ok():
            choice = input("请输入编号（1/2/3/4/q）：").strip()
            if choice == 'q':
                print("退出控制程序。")
                rclpy.shutdown()
                break
            elif choice in self.service_clients:
                self.call_service(choice)
            else:
                print("无效输入，请输入 1、2、3、4 或 q。")

    def call_service(self, choice):
        name, client = self.service_clients[choice]
        request = Empty.Request()
        self.get_logger().info(f"Calling service: {name}")
        future = client.call_async(request)
        future.add_done_callback(lambda fut: self.handle_response(fut, name))

    def handle_response(self, future, name):
        try:
            future.result()
            self.get_logger().info(f"{name} service call successful.")
        except Exception as e:
            self.get_logger().error(f"{name} service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = Go2ServiceClient()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
