#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float64MultiArray

class TrajToPos(Node):
    def __init__(self):
        super().__init__('traj_to_pos_bridge')
        
        # Subscribe to trajectory
        self.sub = self.create_subscription(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            self.callback,
            10
        )
        
        # Publish positions
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/so100/arm_position_controller/commands',
            10
        )
        
        self.get_logger().info('Bridge: JointTrajectory -> Float64MultiArray')
    
    def callback(self, msg):
        if len(msg.points) > 0:
            # Extract first point's positions
            positions = msg.points[0].positions
            
            # Create Float64MultiArray
            out = Float64MultiArray()
            out.data = list(positions)
            
            self.pub.publish(out)

def main():
    rclpy.init()
    node = TrajToPos()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
