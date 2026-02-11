#!/usr/bin/env python3
"""
Leader LQR Controller (SO-100)
- Builds discrete double-integrator dynamics (Ad, Bd) using closed-form ZOH
- Designs expert cost Q* (diagonal weights) and R
- Computes K* via discrete Riccati (DARE)
- Runs control loop:
    x = [q-q_des, dq-dq_des]
    u = -K* x            (u is "acceleration-like" command)
    Integrate u -> dq -> q_cmd (position command for position controller)
    Rate-limit q_cmd for stability
- Publishes K* (and Q,R,A,B,dt) as JSON on /leader_gain at 1 Hz
"""

import json
import numpy as np
from scipy import linalg

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String


class LeaderLQR(Node):
    def __init__(self):
        super().__init__('leader_lqr')

        # ----------------------------
        # Parameters (ROS2)
        # ----------------------------
        self.ns = self.declare_parameter('ns', '/so100').value
        self.j1 = self.declare_parameter('j1', 'Shoulder_Pitch').value
        self.j2 = self.declare_parameter('j2', 'Elbow').value

        self.rate = float(self.declare_parameter('rate_hz', 50.0).value)
        self.dt = 1.0 / self.rate

        self.q_des_1 = float(self.declare_parameter('q_des_1', -0.4).value)
        self.q_des_2 = float(self.declare_parameter('q_des_2', 0.6).value)

        # Q diagonal weights (state cost)
        self.q_weights = [
            float(self.declare_parameter('q1', 10.0).value),
            float(self.declare_parameter('q2', 10.0).value),
            float(self.declare_parameter('q3', 1.0).value),
            float(self.declare_parameter('q4', 1.0).value),
        ]

        # Stability knobs (VERY IMPORTANT for position controller)
        # u is "acceleration-like"; we integrate it to get position commands.
        self.max_pos_step = float(self.declare_parameter('max_pos_step', 0.05).value)  # rad per tick
        self.max_vel = float(self.declare_parameter('max_vel', 2.0).value)             # rad/s clamp (optional)
        self.use_vel_clamp = bool(self.declare_parameter('use_vel_clamp', True).value)

        # Optional joint limits safety clamp (rad)
        self.use_joint_limits = bool(self.declare_parameter('use_joint_limits', True).value)
        self.j1_min = float(self.declare_parameter('j1_min', -3.14).value)
        self.j1_max = float(self.declare_parameter('j1_max',  3.14).value)
        self.j2_min = float(self.declare_parameter('j2_min', -3.14).value)
        self.j2_max = float(self.declare_parameter('j2_max',  3.14).value)

        # Topics
        self.js_topic = f'{self.ns}/joint_states'
        self.cmd_topic = f'{self.ns}/arm_position_controller/commands'
        self.gain_topic = '/leader_gain'

        # ----------------------------
        # State from /joint_states
        # ----------------------------
        self.q = None     # np.array([q1,q2])
        self.dq = None    # np.array([dq1,dq2])
        self.last_q_cmd = None

        # ----------------------------
        # Build expert LQR (K*)
        # ----------------------------
        self.A, self.B = self.build_discrete_dynamics(self.dt, n=2)
        self.Q = np.diag(self.q_weights)
        self.R = np.eye(2)

        self.K_star, self.P_star = self.compute_dlqr(self.A, self.B, self.Q, self.R)

        # Log
        self.get_logger().info("Leader LQR Controller READY")
        self.get_logger().info(f"dt={self.dt:.4f}s, rate={self.rate:.1f}Hz")
        self.get_logger().info(f"Q diag={self.q_weights}, R=I")
        self.get_logger().info(f"K* =\n{self.K_star}")
        self.get_logger().info(f"Target q_des=[{self.q_des_1}, {self.q_des_2}]")
        self.get_logger().info(f"max_pos_step={self.max_pos_step}, use_joint_limits={self.use_joint_limits}")

        # ROS IO
        self.create_subscription(JointState, self.js_topic, self.cb_js, 50)
        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.pub_gain = self.create_publisher(String, self.gain_topic, 10)

        # Timers
        self.timer = self.create_timer(self.dt, self.tick)
        self.gain_timer = self.create_timer(1.0, self.publish_gain)

    # ---------------------------------------------------------
    # Dynamics: discrete double integrator (closed-form ZOH)
    # State x = [q1,q2,dq1,dq2], control u = [ddq1, ddq2]
    # ---------------------------------------------------------
    def build_discrete_dynamics(self, dt: float, n: int = 2):
        """
        Closed-form ZOH discretization for double integrator per joint.

        For each joint:
            q_{k+1}  = q_k + dt*dq_k + 0.5*dt^2*u_k
            dq_{k+1} = dq_k + dt*u_k

        Stack n joints => A is 2n x 2n, B is 2n x n.
        """
        I = np.eye(n)
        Z = np.zeros((n, n))

        A = np.block([
            [I, dt * I],
            [Z, I]
        ])

        B = np.block([
            [0.5 * (dt ** 2) * I],
            [dt * I]
        ])

        return A, B

    # ---------------------------------------------------------
    # DLQR: solve DARE -> K
    # ---------------------------------------------------------
    def compute_dlqr(self, A, B, Q, R):
        """
        Solve discrete-time LQR:
            P = solve_discrete_are(A,B,Q,R)
            K = (R + B'PB)^-1 (B'PA)
        """
        P = linalg.solve_discrete_are(A, B, Q, R)
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, (B.T @ P @ A))
        return K, P

    # ---------------------------------------------------------
    # JointState callback
    # ---------------------------------------------------------
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

        self.q = np.array([q1, q2], dtype=float)
        self.dq = np.array([dq1, dq2], dtype=float)

    # ---------------------------------------------------------
    # Control tick
    # ---------------------------------------------------------
    def tick(self):
        if self.q is None:
            return

        q_des = np.array([self.q_des_1, self.q_des_2], dtype=float)
        dq_des = np.array([0.0, 0.0], dtype=float)

        # x = [q-qdes, dq-dqdes]
        x = np.hstack([self.q - q_des, self.dq - dq_des])  # shape (4,)

        # LQR acceleration-like command
        u = -self.K_star @ x  # shape (2,)

        # Integrate u -> dq_next -> q_cmd
        dq_next = self.dq + u * self.dt

        if self.use_vel_clamp:
            dq_next = np.clip(dq_next, -self.max_vel, self.max_vel)

        q_cmd = self.q + dq_next * self.dt

        # Rate-limit position command per tick (stability)
        if self.last_q_cmd is None:
            self.last_q_cmd = self.q.copy()

        delta = q_cmd - self.last_q_cmd
        delta = np.clip(delta, -self.max_pos_step, self.max_pos_step)
        q_cmd = self.last_q_cmd + delta
        self.last_q_cmd = q_cmd.copy()

        # Joint limit clamp (safety)
        if self.use_joint_limits:
            q_cmd[0] = float(np.clip(q_cmd[0], self.j1_min, self.j1_max))
            q_cmd[1] = float(np.clip(q_cmd[1], self.j2_min, self.j2_max))

        # Publish
        msg = Float64MultiArray()
        msg.data = [float(q_cmd[0]), float(q_cmd[1])]
        self.pub_cmd.publish(msg)

    # ---------------------------------------------------------
    # Publish gain JSON
    # ---------------------------------------------------------
    def publish_gain(self):
        data = {
            'K': self.K_star.tolist(),
            'Q': self.Q.tolist(),
            'R': self.R.tolist(),
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'dt': float(self.dt),
            'joints': [self.j1, self.j2],
            'q_des': [float(self.q_des_1), float(self.q_des_2)],
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
