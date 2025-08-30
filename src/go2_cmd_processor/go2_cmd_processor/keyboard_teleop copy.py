import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame
import sys

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # 设置速度参数
        self.linear_speed = 0.3  # 米/秒
        self.angular_speed = 1.0  # 弧度/秒

        # 初始化 pygame
        pygame.init()
        self.screen = pygame.display.set_mode((400, 100))
        pygame.display.set_caption('Go2 键盘遥控')

        self.get_logger().info("启动键盘控制节点，使用 W A S D 控制，空格停止，ESC 退出")

        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        twist = Twist()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            twist.linear.x = self.linear_speed
        elif keys[pygame.K_s]:
            twist.linear.x = -self.linear_speed

        if keys[pygame.K_a]:
            twist.angular.z = self.angular_speed
        elif keys[pygame.K_d]:
            twist.angular.z = -self.angular_speed

        if keys[pygame.K_SPACE]:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.publisher.publish(twist)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                rclpy.shutdown()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.get_logger().info("退出控制")
                    pygame.quit()
                    rclpy.shutdown()
                    sys.exit()

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
