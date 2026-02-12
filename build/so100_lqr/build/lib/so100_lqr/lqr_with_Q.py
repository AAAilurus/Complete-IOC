#!/usr/bin/env python3
"""
LQR Controller that computes K from Q (discrete-time)
"""
from typing import Dict, List, Optional
import numpy as np
from scipy.linalg import solve_discrete_are

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class LqrWithQ(Node):
    def __init__(self):
        super().__init__("lqr_with_Q")

        # Parameters
        self.declare_parameter("joints", ["Shoulder_Pitch", "Elbow"])
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("cmd_topic", "/arm_position_controller/commands")
        self.declare_parameter("rate_hz", 50.0)
        
        # Target state
        self.declare_parameter("q_des", [0.5, -0.6])
        self.declare_parameter("dq_des", [0.0, 0.0])
        
        # Cost matrices Q and R (NOT K!)
        self.declare_parameter("Q_diag", [100.0, 100.0, 10.0, 10.0])
        self.declare_parameter("R_diag", [0.5, 0.5])
        
        # Control parameters
        self.declare_parameter("u_max", 2.0)
        self.declare_parameter("control_scale", 0.01)

        # Get parameters
        self.joints: List[str] = list(self.get_parameter("joints").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        
        self.q_des = [float(v) for v in self.get_parameter("q_des").value]
        self.dq_des = [float(v) for v in self.get_parameter("dq_des").value]
        
        Q_diag = list(self.get_parameter("Q_diag").value)
        R_diag = list(self.get_parameter("R_diag").value)
        
        self.u_max = float(self.get_parameter("u_max").value)
        self.control_scale = float(self.get_parameter("control_scale").value)

        # Build discrete-time system
        self.dt = 1.0 / self.rate_hz
        Ts = self.dt
        
        # Discrete double integrator (ZOH)
        self.Ad = np.array([[1, 0, Ts, 0],
                            [0, 1, 0, Ts],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        self.Bd = np.array([[0.5*Ts**2, 0],
                            [0, 0.5*Ts**2],
                            [Ts, 0],
                            [0, Ts]])
        
        # Cost matrices
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        
        # Compute K from Q using discrete-time Riccati equation
        self.P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.Bd.T @ self.P @ self.Bd) @ (self.Bd.T @ self.P @ self.Ad)
        
        # State
        self.q: Optional[List[float]] = None
        self.dq: Optional[List[float]] = None
        self.last_cmd = None

        # ROS interfaces
        self.sub = self.create_subscription(JointState, self.state_topic, self._on_joint_state, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.timer = self.create_timer(self.dt, self._step)

        self.get_logger().info(f"[LQR with Q] joints={self.joints}")
        self.get_logger().info(f"[LQR with Q] sub={self.state_topic}")
        self.get_logger().info(f"[LQR with Q] pub={self.cmd_topic}")
        self.get_logger().info(f"[LQR with Q] rate={self.rate_hz:.1f}Hz dt={self.dt:.4f}s")
        self.get_logger().info(f"[LQR with Q] Q_diag={Q_diag}")
        self.get_logger().info(f"[LQR with Q] R_diag={R_diag}")
        self.get_logger().info(f"[LQR with Q] Computed K:\n{self.K}")

    def _on_joint_state(self, msg: JointState):
        name_to_i: Dict[str, int] = {n: i for i, n in enumerate(msg.name)}
        if any(j not in name_to_i for j in self.joints):
            return

        q = []
        dq = []
        for j in self.joints:
            i = name_to_i[j]
            q.append(float(msg.position[i]))
            dq.append(float(msg.velocity[i]) if len(msg.velocity) > i else 0.0)

        self.q = q
        self.dq = dq

    def _step(self):
        if self.q is None or self.dq is None:
            return

        # State error
        e = [
            self.q[0] - self.q_des[0],
            self.q[1] - self.q_des[1],
            self.dq[0] - self.dq_des[0],
            self.dq[1] - self.dq_des[1],
        ]
        e_vec = np.array(e).reshape(4, 1)

        # LQR control: u = -K*e
        u = -self.K @ e_vec
        u = u.flatten()
        
        # Clamp
        u[0] = clamp(u[0], -self.u_max, self.u_max)
        u[1] = clamp(u[1], -self.u_max, self.u_max)

        # Scale and compute position command
        u_scaled = u * self.control_scale
        q_cmd = [self.q[0] + u_scaled[0], self.q[1] + u_scaled[1]]
        
        # Rate limiting
        if self.last_cmd is not None:
            max_change = 0.05
            for i in range(2):
                delta = q_cmd[i] - self.last_cmd[i]
                delta = clamp(delta, -max_change, max_change)
                q_cmd[i] = self.last_cmd[i] + delta
        
        self.last_cmd = q_cmd[:]

        # Publish
        msg = Float64MultiArray()
        msg.data = q_cmd
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LqrWithQ()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
