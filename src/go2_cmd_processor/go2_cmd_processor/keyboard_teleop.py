import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import pygame
import sys

class Button:
    def __init__(self, rect, color, text, action):
        self.rect = pygame.Rect(rect)
        self.color = color
        self.original_color = color
        self.text = text
        self.action = action

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color, self.rect)
        label = font.render(self.text, True, (255, 255, 255))
        label_rect = label.get_rect(center=self.rect.center)
        screen.blit(label, label_rect)

    def handle_event(self, event, publish_fn):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.color = (100, 100, 100)
                publish_fn(self.action)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.color = self.original_color

class GuiTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.linear_speed = 0.44
        self.angular_speed = 2.0

        # 初始化服务客户端
        self.stand_up_client = self.create_client(Empty, 'stand_up')
        self.lay_down_client = self.create_client(Empty, 'lay_down')

        # 初始化 pygame
        pygame.init()
        self.screen = pygame.display.set_mode((600, 450))
        pygame.display.set_caption("Go2 GUI Teleop")
        self.font = pygame.font.SysFont(None, 36)

        # 图形按钮
        self.buttons = [
            Button((250, 30, 100, 50), (0, 138, 0), "forward", "forward"),
            Button((150, 100, 100, 50), (0, 138, 0), "left", "left"),
            Button((350, 100, 100, 50), (0, 138, 0), "right", "right"),
            Button((250, 170, 100, 50), (0, 138, 0), "backward", "backward"),
            Button((50, 240, 100, 50), (0, 138, 100), "← move left", "move_left"),
            Button((450, 240, 100, 50), (0, 138, 100), "→ move right", "move_right"),
            Button((100, 330, 130, 50), (0, 100, 200), "stand_up", "stand_up"),
            Button((370, 330, 130, 50), (200, 50, 50), "lay_down", "lay_down"),
        ]

        self.get_logger().info("GUI supports keyboard and GUI control")
        self.timer = self.create_timer(0.1, self.loop)

    def publish_twist(self, action):
        twist = Twist()
        if action == "forward":
            twist.linear.x = self.linear_speed
        elif action == "backward":
            twist.linear.x = -self.linear_speed
        elif action == "left":
            twist.angular.z = self.angular_speed
        elif action == "right":
            twist.angular.z = -self.angular_speed
        elif action == "move_left":
            twist.linear.y = -self.linear_speed
        elif action == "move_right":
            twist.linear.y = self.linear_speed
        else:
            return
        self.publisher.publish(twist)
        self.get_logger().info(f"Published motion: {action}")

    def call_posture_service(self, action):
        client = None
        if action == "stand_up":
            client = self.stand_up_client
        elif action == "lay_down":
            client = self.lay_down_client
        if client is None:
            return

        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"{action} service not available")
            return
        request = Empty.Request()
        future = client.call_async(request)
        future.add_done_callback(lambda f: self.get_logger().info(f"{action} command sent"))

    def handle_action(self, action):
        if action in ["forward", "backward", "left", "right", "move_left", "move_right"]:
            self.publish_twist(action)
        elif action in ["stand_up", "lay_down"]:
            self.call_posture_service(action)

    def stop_robot(self):
        twist = Twist()
        self.publisher.publish(twist)

    def loop(self):
        self.screen.fill((30, 30, 30))

        for button in self.buttons:
            button.draw(self.screen, self.font)

        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.publish_twist("forward")
        elif keys[pygame.K_s]:
            self.publish_twist("backward")
        elif keys[pygame.K_a]:
            self.publish_twist("left")
        elif keys[pygame.K_d]:
            self.publish_twist("right")
        elif keys[pygame.K_q]:
            self.publish_twist("move_left")
        elif keys[pygame.K_e]:
            self.publish_twist("move_right")
        elif keys[pygame.K_SPACE]:
            self.stop_robot()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                rclpy.shutdown()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                rclpy.shutdown()
                sys.exit()
            else:
                for button in self.buttons:
                    button.handle_event(event, self.handle_action)

def main(args=None):
    rclpy.init(args=args)
    node = GuiTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        rclpy.shutdown()

if __name__ == '__main__':
    main()