#!/usr/bin/env python3
"""
Leader LQR Controller
- Uses forward LQR with designed Q matrix
- Computes K* and controls the leader arm
- Publishes K* to follower
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import numpy as np
from scipy import linalg
import control
import json


class LeaderLQR(Node):
    def __init__(self):
        super().__init__('leader_lqr')
        
        # Parameters
        self.ns = self.declare_parameter('ns', '/so100').value
        self.j1 = self.declare_parameter('j1', 'Shoulder_Pitch').value
        self.j2 = self.declare_parameter('j2', 'Elbow').value
        self.rate = float(self.declare_parameter('rate_hz', 50.0).value)
        self.q_des_1 = float(self.declare_parameter('q_des_1', -0.4).value)
        self.q_des_2 = float(self.declare_parameter('q_des_2', 0.6).value)
        
        # Q matrix diagonal values
        self.q_weights = [
            float(self.declare_parameter('q1', 10.0).value),
            float(self.declare_parameter('q2', 10.0).value),
            float(self.declare_parameter('q3', 1.0).value),
            float(self.declare_parameter('q4', 1.0).value)
        ]
        
        # Topics
        self.js_topic = f'{self.ns}/joint_states'
        self.cmd_topic = f'{self.ns}/arm_position_controller/commands'
        self.gain_topic = '/leader_gain'  # Publish K* here
        
        # State
        self.q = None
        self.dq = None
        
        # Build LQR controller
        self.dt = 1.0 / self.rate
        self.A, self.B = self.build_discrete_dynamics(self.dt)
        self.Q = np.diag(self.q_weights)
        self.R = np.eye(2)
        
        # Compute optimal gain K*
        self.K_star = self.compute_lqr_gain(self.A, self.B, self.Q, self.R)
        
        self.get_logger().info(f'Leader LQR Controller')
        self.get_logger().info(f'Q weights: {self.q_weights}')
        self.get_logger().info(f'K* (optimal gain):\n{self.K_star}')
        self.get_logger().info(f'Desired state: [{self.q_des_1}, {self.q_des_2}]')
        
        # Publishers and subscribers
        self.create_subscription(JointState, self.js_topic, self.cb_js, 50)
        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.pub_gain = self.create_publisher(String, self.gain_topic, 10)
        
        # Control timer
        self.timer = self.create_timer(self.dt, self.tick)
        
        # Publish K* at 1 Hz
        self.gain_timer = self.create_timer(1.0, self.publish_gain)
        
    def build_discrete_dynamics(self, dt, n=2, m=2):
        """Build discrete-time double integrator dynamics."""
        Ac = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [np.zeros((n, n)), np.zeros((n, n))]
        ])
        Bc = np.block([
            [np.zeros((n, m))],
            [np.eye(m)]
        ])
        
        sys_c = control.StateSpace(Ac, Bc, np.eye(4), np.zeros((4, 2)))
        sys_d = control.c2d(sys_c, dt, method='zoh')
        
        return sys_d.A, sys_d.B
    
    def compute_lqr_gain(self, A, B, Q, R):
        """Compute LQR optimal gain."""
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return K
    
    def cb_js(self, msg: JointState):
        """Joint state callback."""
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
        """LQR control loop."""
        if self.q is None:
            return
        
        # State deviation from desired
        q_des = np.array([self.q_des_1, self.q_des_2])
        dq_des = np.array([0.0, 0.0])
        
        x = np.hstack([self.q - q_des, self.dq - dq_des])  # 4D state
        
        # LQR control: u = -K*x
        u = -self.K_star @ x  # 2D control (acceleration)
        
        # Integrate to get position command
        # u is acceleration, integrate twice: dq_new = dq + u*dt, q_new = q + dq*dt
        # But for position controller, we want position command
        # Simple approach: q_cmd = q_des (let the position controller handle it)
        # OR: q_cmd = q + velocity_correction
        
        # For now, use the same approach as follower:
        q_cmd = self.q + u  # Treating u as position correction
        
        msg = Float64MultiArray()
        msg.data = [float(q_cmd[0]), float(q_cmd[1])]
        self.pub_cmd.publish(msg)
    
    def publish_gain(self):
        """Publish K* matrix as JSON."""
        data = {
            'K': self.K_star.tolist(),
            'Q': self.Q.tolist(),
            'R': self.R.tolist(),
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'dt': self.dt
        }
        msg = String()
        msg.data = json.dumps(data)
        self.pub_gain.publish(msg)


def main():
    rclpy.init()
    node = LeaderLQR()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
