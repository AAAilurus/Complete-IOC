#!/usr/bin/env python3
import csv
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class StateLoggerCSV(Node):
    """
    Logs joint position/velocity from JointState and computes acceleration by finite difference.
    CSV columns:
      t, q1, q2, dq1, dq2, ddq1, ddq2
    """
    def __init__(self):
        super().__init__('state_logger_csv')

        self.declare_parameter('state_topic', '/so100/joint_states')
        self.declare_parameter('joints', ['Shoulder_Pitch', 'Elbow'])
        self.declare_parameter('dt_out', 0.01)
        self.declare_parameter('csv_path', '/root/so100_ws/leader_state.csv')

        self.state_topic = self.get_parameter('state_topic').value
        self.joints = list(self.get_parameter('joints').value)
        self.dt_out = float(self.get_parameter('dt_out').value)
        self.csv_path = self.get_parameter('csv_path').value

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        self.idx = None
        self.last_t = None
        self.last_dq = None
        self.latest_row = None

        self.sub = self.create_subscription(JointState, self.state_topic, self.cb, 50)
        self.timer = self.create_timer(self.dt_out, self.on_timer)

        self.f = open(self.csv_path, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(['t', 'q1', 'q2', 'dq1', 'dq2', 'ddq1', 'ddq2'])
        self.f.flush()

        self.get_logger().info(f"Logging from {self.state_topic} joints={self.joints} -> {self.csv_path}")

    def cb(self, msg: JointState):
        if self.idx is None:
            name_to_i = {n: i for i, n in enumerate(msg.name)}
            try:
                self.idx = [name_to_i[j] for j in self.joints]
            except KeyError:
                self.get_logger().warn(f"Joint names not found yet. msg.name={msg.name}")
                return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = [msg.position[i] if i < len(msg.position) else float('nan') for i in self.idx]
        dq = [msg.velocity[i] if i < len(msg.velocity) else float('nan') for i in self.idx]

        ddq = [0.0, 0.0]
        if self.last_t is not None and self.last_dq is not None:
            dt = t - self.last_t
            if dt > 1e-6:
                ddq = [(dq[k] - self.last_dq[k]) / dt for k in range(2)]

        self.last_t = t
        self.last_dq = dq
        self.latest_row = [t, q[0], q[1], dq[0], dq[1], ddq[0], ddq[1]]

    def on_timer(self):
        if self.latest_row is None:
            return
        self.w.writerow(self.latest_row)
        self.f.flush()

    def destroy_node(self):
        try:
            self.f.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = StateLoggerCSV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
