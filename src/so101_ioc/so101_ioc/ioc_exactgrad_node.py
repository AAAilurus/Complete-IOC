#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray

from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def proj_psd_floor_diagQ(Q: np.ndarray, floor_eig: float) -> np.ndarray:
    # since we keep Q diagonal, just clamp diag
    d = np.diag(Q).copy()
    d = np.maximum(d, floor_eig)
    return np.diag(d)


class IOCExactGrad(Node):
    """
    Online IOC (continuous-time) on a double integrator x=[q1,q2,dq1,dq2], u=[ddq1,ddq2].
    Learns diagonal Q using exact gradient (Lyapunov) to match leader behavior.

    Leader data:
      - from JointState: q, dq
      - ddq via finite difference on dq

    Policy:
      u_hat = -K(Q) (x - x_eq)    where K(Q)= R^{-1} B^T P, P solves continuous ARE

    Loss:
      J = 0.5 * || u_hat - u_leader ||^2   (per-sample)
    Gradient:
      For each q_ii:
        Acl = A - B K
        Solve: Acl^T dP + dP Acl + E_i = 0
        dK = R^{-1} B^T dP
        du = -(dK) (x-x_eq)
        dJ/dq_i = (u_hat - u_leader)^T du
    """
    def __init__(self):
        super().__init__('ioc_exactgrad')

        # ---------- ROS params ----------
        self.declare_parameter('leader_state_topic', '/joint_states')
        self.declare_parameter('leader_joints', ['Shoulder_Pitch', 'Elbow'])

        self.declare_parameter('follower_cmd_topic', '/joint_trajectory_controller/joint_trajectory')
        self.declare_parameter('follower_joints', ['Shoulder_Pitch', 'Elbow'])

        self.declare_parameter('rate_hz', 100.0)

        # equilibrium / target
        self.declare_parameter('q_des', [0.5, -0.6])
        self.declare_parameter('dq_des', [0.0, 0.0])

        # LQR / IOC
        self.declare_parameter('R_diag', [0.5, 0.5])
        self.declare_parameter('Q0_diag', [10.0, 10.0, 1.0, 1.0])
        self.declare_parameter('alphaQ', 5e-2)        # start smaller than MATLAB 5e-1 for stability
        self.declare_parameter('pd_floor', 1e-6)
        self.declare_parameter('print_every', 100)    # iterations
        self.declare_parameter('warmup', 5)           # skip first ddq estimates

        # follower saturation
        self.declare_parameter('u_max', 2.0)          # max ddq
        self.declare_parameter('dq_max', 2.0)         # max dq_cmd
        self.declare_parameter('q_min', [-math.pi, -math.pi])
        self.declare_parameter('q_max', [ math.pi,  math.pi])

        # ---------- read params ----------
        self.leader_topic = self.get_parameter('leader_state_topic').value
        self.leader_joints = list(self.get_parameter('leader_joints').value)

        self.cmd_topic = self.get_parameter('follower_cmd_topic').value
        self.follower_joints = list(self.get_parameter('follower_joints').value)

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.dt = 1.0 / self.rate_hz

        q_des = np.array(self.get_parameter('q_des').value, dtype=float).reshape(2, )
        dq_des = np.array(self.get_parameter('dq_des').value, dtype=float).reshape(2, )
        self.x_eq = np.hstack([q_des, dq_des]).reshape(4, 1)

        R_diag = np.array(self.get_parameter('R_diag').value, dtype=float).reshape(2, )
        self.R = np.diag(R_diag)

        Q0_diag = np.array(self.get_parameter('Q0_diag').value, dtype=float).reshape(4, )
        self.Q = np.diag(Q0_diag)

        self.alphaQ = float(self.get_parameter('alphaQ').value)
        self.pd_floor = float(self.get_parameter('pd_floor').value)
        self.print_every = int(self.get_parameter('print_every').value)
        self.warmup = int(self.get_parameter('warmup').value)

        self.u_max = float(self.get_parameter('u_max').value)
        self.dq_max = float(self.get_parameter('dq_max').value)
        self.q_min = np.array(self.get_parameter('q_min').value, dtype=float).reshape(2, )
        self.q_max = np.array(self.get_parameter('q_max').value, dtype=float).reshape(2, )

        # ---------- system matrices (double integrator) ----------
        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=float)

        self.B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1],
        ], dtype=float)

        # ---------- leader state buffers ----------
        self.prev_t = None
        self.prev_dq = None
        self.last_x = None
        self.last_u_leader = None
        self.sample_count = 0

        # ---------- follower internal command integrator ----------
        # start follower at equilibrium
        self.q_cmd = q_des.copy()
        self.dq_cmd = dq_des.copy()

        # ---------- ROS interfaces ----------
        self.sub = self.create_subscription(JointState, self.leader_topic, self.cb_leader, 50)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self.timer = self.create_timer(self.dt, self.step)

        self.iter = 0

        self.get_logger().info(
            f"IOCExactGrad started.\n"
            f"  leader_topic={self.leader_topic} joints={self.leader_joints}\n"
            f"  cmd_topic={self.cmd_topic} follower_joints={self.follower_joints}\n"
            f"  dt={self.dt:.4f}s rate={self.rate_hz:.1f}Hz\n"
            f"  x_eq={self.x_eq.ravel().tolist()}\n"
            f"  Q0_diag={np.diag(self.Q).tolist()} R_diag={np.diag(self.R).tolist()}"
        )

    def now_sec(self, msg: JointState) -> float:
        # Use stamp if present, else node clock
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            return float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
        return self.get_clock().now().nanoseconds * 1e-9

    def cb_leader(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        try:
            i1 = name_to_i[self.leader_joints[0]]
            i2 = name_to_i[self.leader_joints[1]]
        except KeyError:
            return

        q = np.array([
            msg.position[i1] if len(msg.position) > i1 else 0.0,
            msg.position[i2] if len(msg.position) > i2 else 0.0
        ], dtype=float)

        dq = np.array([
            msg.velocity[i1] if len(msg.velocity) > i1 else 0.0,
            msg.velocity[i2] if len(msg.velocity) > i2 else 0.0
        ], dtype=float)

        t = self.now_sec(msg)

        ddq = np.zeros(2, dtype=float)
        if self.prev_t is not None and self.prev_dq is not None:
            dt = t - self.prev_t
            if dt > 1e-6:
                ddq = (dq - self.prev_dq) / dt

        self.prev_t = t
        self.prev_dq = dq.copy()

        x = np.hstack([q, dq]).reshape(4, 1)
        u = ddq.reshape(2, 1)

        self.last_x = x
        self.last_u_leader = u
        self.sample_count += 1

    def lqr_gain(self, Q: np.ndarray) -> np.ndarray:
        # continuous ARE: A^T P + P A - P B R^-1 B^T P + Q = 0
        P = solve_continuous_are(self.A, self.B, Q, self.R)
        P = sym(P)
        Rinv = np.linalg.inv(self.R)
        K = Rinv @ (self.B.T @ P)   # 2x4
        return K, P

    def exact_grad_diagQ(self, K: np.ndarray, P: np.ndarray, x_err: np.ndarray, u_leader: np.ndarray) -> np.ndarray:
        # J = 0.5 ||u_hat - u_leader||^2
        # u_hat = -K x_err
        u_hat = -K @ x_err
        e = (u_hat - u_leader)  # 2x1

        Acl = self.A - self.B @ K
        Rinv = np.linalg.inv(self.R)

        g = np.zeros(4, dtype=float)
        for i in range(4):
            Ei = np.zeros((4, 4), dtype=float)
            Ei[i, i] = 1.0

            # Solve: Acl^T dP + dP Acl + Ei = 0
            # scipy.solve_continuous_lyapunov(A, Q) solves: A X + X A^T = -Q
            # we want (Acl^T) dP + dP (Acl) = -Ei  => A=Acl^T, Q=Ei
            dP = solve_continuous_lyapunov(Acl.T, Ei)
            dP = sym(dP)

            dK = Rinv @ (self.B.T @ dP)  # 2x4
            du = -(dK @ x_err)           # 2x1

            g[i] = float(e.T @ du)       # scalar
        return g  # length 4
    def publish_follower_cmd(self, q_cmd: np.ndarray, dt_cmd: float):
        msg = Float64MultiArray()
        msg.data = [float(q_cmd[0]), float(q_cmd[1])]
        self.pub.publish(msg)

    def step(self):
        # Need leader data first
        if self.last_x is None or self.last_u_leader is None:
            return
        if self.sample_count <= self.warmup:
            return

        self.iter += 1

        x = self.last_x
        uL = self.last_u_leader

        x_err = x - self.x_eq

        # 1) current K(Q)
        try:
            K, P = self.lqr_gain(self.Q)
        except Exception as ex:
            self.get_logger().error(f"LQR solve failed: {ex}")
            return

        # 2) exact gradient update on diag(Q)
        g = self.exact_grad_diagQ(K, P, x_err, uL)

        # gradient step on diagonal entries only
        qdiag = np.diag(self.Q).copy()
        qdiag_new = qdiag - self.alphaQ * g
        self.Q = proj_psd_floor_diagQ(np.diag(qdiag_new), self.pd_floor)

        # 3) generate follower command using u_hat from current K
        # recompute K after update (optional; small difference)
        K2, _ = self.lqr_gain(self.Q)
        u_hat = -K2 @ x_err  # 2x1
        u_hat = np.clip(u_hat, -self.u_max, self.u_max)

        # integrate to position command
        self.dq_cmd = np.clip(self.dq_cmd + (u_hat.ravel() * self.dt), -self.dq_max, self.dq_max)
        self.q_cmd = self.q_cmd + self.dq_cmd * self.dt
        self.q_cmd = np.clip(self.q_cmd, self.q_min, self.q_max)

        self.publish_follower_cmd(self.q_cmd, dt_cmd=0.2)

        # 4) prints
        if (self.iter % self.print_every) == 0:
            self.get_logger().info(
                f"[IOC] iter={self.iter} "
                f"Qdiag={np.round(np.diag(self.Q), 4).tolist()}\n"
                f"K=\n{np.array2string(K2, precision=4, suppress_small=True)}"
            )


def main():
    rclpy.init()
    node = IOCExactGrad()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            if rclpy.ok():
                rclpy.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
