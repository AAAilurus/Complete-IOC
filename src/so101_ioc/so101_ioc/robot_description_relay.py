#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotDescriptionRelay(Node):
    def __init__(self):
        super().__init__('robot_description_relay')
        self.declare_parameter('in_topic',  '/so100/robot_description')
        self.declare_parameter('out_topic', '/robot_description')

        self.in_topic  = self.get_parameter('in_topic').value
        self.out_topic = self.get_parameter('out_topic').value

        self.pub = self.create_publisher(String, self.out_topic, 10)
        self.sub = self.create_subscription(String, self.in_topic, self.cb, 10)

        self.get_logger().info(f"Relaying {self.in_topic}  ->  {self.out_topic}")

    def cb(self, msg: String):
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = RobotDescriptionRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
