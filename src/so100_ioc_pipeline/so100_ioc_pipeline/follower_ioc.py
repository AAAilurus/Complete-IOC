#!/usr/bin/env python3
"""
Follower: Uses learned Q from offline IOC
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy import linalg


class FollowerIOC(Node):
    def __init__(self):
        super().__init__('follower_ioc')
        
        self.ns = self.declare_parameter('ns', '/so101').value
        self.j1 = self.declare_parameter('j1', 'Shoulder_Pitch').value
        self.j2 = self.declare_parameter('j2', 'Elbow').value
        self.rate = float(self.declare_parameter('rate_hz', 50.0).value)
        self.q_des_1 = float(self.declare_parameter('q_des_1', -0.4).value)
        self.q_des_2 = float(self.declare_parameter('q_des_2', 0.6).value)
        self.q_file = self.declare_parameter('q_file', '/tmp/learned_Q.npz').value
        
        # Load learned Q and K
        data = np.load(self.q_file)
        self.Q = data['Q']
        self.K = data['K']
        
        self.js_topic = f'{self.ns}/joint_states'
        self.cmd_topic = f'{self.ns}/arm_position_controller/commands'
        
        self.q = None
        self.dq = None
        
        self.create_subscription(JointState, self.js_topic, self.cb_js, 50)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        
        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self.tick)
        
        self.get_logger().info(f'Follower with learned Q')
        self.get_logger().info(f'Q file: {self.q_file}')
        self.get_logger().info(f'Q_learned:\n{self.Q}')
        self.get_logger().info(f'K_learned:\n{self.K}')
        self.get_logger().info(f'Desired: [{self.q_des_1}, {self.q_des_2}]')
    
    def cb_js(self, msg: JointState):
        try:
            i1 = msg.name.index(self.j1)
            i2 = msg.name.index(self.j2)
        except ValueError:
            return
        
        q1, q2 = msg.position[i1], msg.position[i2]
        if len(msg.velocity) == len(msg.name):
            dq1, dq2 = msg.velocity[i1], msg.velocity[i2]
        else:
            dq1, dq2 = 0.0, 0.0
        
        self.q = np.array([q1, q2])
        self.dq = np.array([dq1, dq2])
    
    def tick(self):
        if self.q is None:
            return
        
        q_des = np.array([self.q_des_1, self.q_des_2])
        dq_des = np.array([0.0, 0.0])
        
        x = np.hstack([self.q - q_des, self.dq - dq_des])
        u = -self.K @ x
        q_cmd = self.q + u
        
        msg = Float64MultiArray()
        msg.data = [float(q_cmd[0]), float(q_cmd[1])]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FollowerIOC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
