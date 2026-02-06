#!/usr/bin/env python3
"""
Follower: Uses learned Q from offline IOC
Records state, control, and error data during execution
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy import linalg
import pandas as pd
import time


class FollowerIOCWithLogging(Node):
    def __init__(self):
        super().__init__('follower_ioc_with_logging')
        
        self.ns = self.declare_parameter('ns', '/so101').value
        self.j1 = self.declare_parameter('j1', 'Shoulder_Pitch').value
        self.j2 = self.declare_parameter('j2', 'Elbow').value
        self.rate = float(self.declare_parameter('rate_hz', 50.0).value)
        self.q_des_1 = float(self.declare_parameter('q_des_1', -0.4).value)
        self.q_des_2 = float(self.declare_parameter('q_des_2', 0.6).value)
        self.q_file = self.declare_parameter('q_file', '/root/so100_ws/ioc_results/learned_Q.npz').value
        self.log_every = int(self.declare_parameter('log_every', 200).value)
        self.log_file = self.declare_parameter('log_file', '/root/so100_ws/ioc_results/follower_data.csv').value
        
        # Load learned Q and K
        data = np.load(self.q_file)
        self.Q = data['Q']
        self.K = data['K']
        
        self.js_topic = f'{self.ns}/joint_states'
        self.cmd_topic = f'{self.ns}/arm_position_controller/commands'
        
        self.q = None
        self.dq = None
        self.step = 0
        self.start_time = time.time()
        
        # Data logging
        self.log_data = {
            'time': [],
            'step': [],
            'q1': [], 'q2': [],
            'dq1': [], 'dq2': [],
            'q_err1': [], 'q_err2': [],
            'dq_err1': [], 'dq_err2': [],
            'u1': [], 'u2': [],
            'qcmd1': [], 'qcmd2': []
        }
        
        self.create_subscription(JointState, self.js_topic, self.cb_js, 50)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        
        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self.tick)
        
        self.get_logger().info(f'Follower with learned Q + Data Logging')
        self.get_logger().info(f'Q file: {self.q_file}')
        self.get_logger().info(f'Logging every {self.log_every} steps to: {self.log_file}')
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
        
        # State error
        q_err = self.q - q_des
        dq_err = self.dq - dq_des
        x = np.hstack([q_err, dq_err])
        
        # Control
        u = -self.K @ x
        q_cmd = self.q + u
        
        # Publish command
        msg = Float64MultiArray()
        msg.data = [float(q_cmd[0]), float(q_cmd[1])]
        self.pub.publish(msg)
        
        # Log data every N steps
        if self.step % self.log_every == 0:
            elapsed = time.time() - self.start_time
            
            self.log_data['time'].append(elapsed)
            self.log_data['step'].append(self.step)
            self.log_data['q1'].append(self.q[0])
            self.log_data['q2'].append(self.q[1])
            self.log_data['dq1'].append(self.dq[0])
            self.log_data['dq2'].append(self.dq[1])
            self.log_data['q_err1'].append(q_err[0])
            self.log_data['q_err2'].append(q_err[1])
            self.log_data['dq_err1'].append(dq_err[0])
            self.log_data['dq_err2'].append(dq_err[1])
            self.log_data['u1'].append(u[0])
            self.log_data['u2'].append(u[1])
            self.log_data['qcmd1'].append(q_cmd[0])
            self.log_data['qcmd2'].append(q_cmd[1])
            
            self.get_logger().info(
                f'Step {self.step}: q=[{self.q[0]:.3f}, {self.q[1]:.3f}], '
                f'err=[{q_err[0]:.4f}, {q_err[1]:.4f}]'
            )
        
        self.step += 1
    
    def __del__(self):
        """Save data on exit"""
        if len(self.log_data['time']) > 0:
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_file, index=False)
            print(f'\n✓ Saved {len(df)} data points to: {self.log_file}')


def main():
    rclpy.init()
    node = FollowerIOCWithLogging()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save data before shutdown
        if len(node.log_data['time']) > 0:
            df = pd.DataFrame(node.log_data)
            df.to_csv(node.log_file, index=False)
            print(f'\n✓ Saved {len(df)} data points to: {node.log_file}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
