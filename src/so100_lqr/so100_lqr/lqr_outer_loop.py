from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class LqrOuterLoop(Node):
    def __init__(self):
        super().__init__("so100_lqr_outer_loop")

        self.declare_parameter("joints", ["Shoulder_Pitch", "Elbow"])
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("cmd_topic", "/joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("rate_hz", 100.0)

        self.declare_parameter("q_des", [0.5, -0.6])
        self.declare_parameter("dq_des", [0.0, 0.0])

        # Flattened K: [k11,k12,k13,k14,k21,k22,k23,k24]
        self.declare_parameter("K", [
            20.0, 0.0, 6.0, 0.0,
            0.0, 20.0, 0.0, 6.0
        ])

        self.declare_parameter("u_max", 2.0)
        self.declare_parameter("dq_max", 1.0)
        self.declare_parameter("point_dt", 0.05)

        self.joints: List[str] = list(self.get_parameter("joints").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.q_des = [float(v) for v in self.get_parameter("q_des").value]
        self.dq_des = [float(v) for v in self.get_parameter("dq_des").value]

        flatK = list(self.get_parameter("K").value)
        if len(flatK) != 8:
            raise RuntimeError("K must have length 8 (2x4 flattened).")
        self.K = [flatK[0:4], flatK[4:8]]

        self.u_max = float(self.get_parameter("u_max").value)
        self.dq_max = float(self.get_parameter("dq_max").value)
        self.point_dt = float(self.get_parameter("point_dt").value)

        self.q: Optional[List[float]] = None
        self.dq: Optional[List[float]] = None
        self.dq_cmd = [0.0, 0.0]

        self.sub = self.create_subscription(JointState, self.state_topic, self._on_joint_state, 10)
        self.pub = self.create_publisher(JointTrajectory, self.cmd_topic, 10)

        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self._step)

        self.get_logger().info(f"[LQR] joints={self.joints}")
        self.get_logger().info(f"[LQR] sub={self.state_topic}")
        self.get_logger().info(f"[LQR] pub={self.cmd_topic}")
        self.get_logger().info(f"[LQR] rate={self.rate_hz:.1f}Hz dt={self.dt:.4f}s")

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

        e = [
            self.q[0] - self.q_des[0],
            self.q[1] - self.q_des[1],
            self.dq[0] - self.dq_des[0],
            self.dq[1] - self.dq_des[1],
        ]

        u = [0.0, 0.0]
        for r in range(2):
            u[r] = -(self.K[r][0]*e[0] + self.K[r][1]*e[1] + self.K[r][2]*e[2] + self.K[r][3]*e[3])
            u[r] = clamp(u[r], -self.u_max, self.u_max)

        self.dq_cmd[0] = clamp(self.dq_cmd[0] + u[0]*self.dt, -self.dq_max, self.dq_max)
        self.dq_cmd[1] = clamp(self.dq_cmd[1] + u[1]*self.dt, -self.dq_max, self.dq_max)

        q_cmd = [
            self.q[0] + self.dq_cmd[0]*self.dt,
            self.q[1] + self.dq_cmd[1]*self.dt,
        ]

        traj = JointTrajectory()
        traj.joint_names = self.joints

        pt = JointTrajectoryPoint()
        pt.positions = q_cmd
        pt.time_from_start = Duration(
            sec=int(self.point_dt),
            nanosec=int((self.point_dt % 1.0) * 1e9),
        )
        traj.points = [pt]
        self.pub.publish(traj)


def main():
    rclpy.init()
    node = LqrOuterLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
