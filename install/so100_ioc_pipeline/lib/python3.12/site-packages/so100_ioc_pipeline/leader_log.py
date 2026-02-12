#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time, math, csv

def now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

class LeaderLog(Node):
    def __init__(self):
        super().__init__('leader_log')

        self.ns = self.declare_parameter('ns', '/so100').value
        self.j1 = self.declare_parameter('j1', 'Shoulder_Pitch').value
        self.j2 = self.declare_parameter('j2', 'Elbow').value
        self.rate = float(self.declare_parameter('rate_hz', 50.0).value)
        self.duration = float(self.declare_parameter('duration_s', 10.0).value)
        self.amp = float(self.declare_parameter('amp', 0.35).value)
        self.freq = float(self.declare_parameter('freq_hz', 0.12).value)

        self.cmd_topic = f"{self.ns}/arm_position_controller/commands"
        self.js_topic  = f"{self.ns}/joint_states"

        # BEST_EFFORT subscription to match joint_states QoS
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.last_js = None
        self.create_subscription(JointState, self.js_topic, self._js_cb, qos)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self.rows = []
        self.t0 = time.time()

        self.timer = self.create_timer(1.0/self.rate, self._tick)

        self.csv_path = f"/tmp/leader_{self.ns.strip('/').replace('/','_')}_{now_str()}.csv"
        self.get_logger().info(f"Leader logging: {self.csv_path}")
        self.get_logger().info(f"Pub: {self.cmd_topic} | Sub: {self.js_topic}")

    def _js_cb(self, msg: JointState):
        self.last_js = msg

    def _get_joint(self, msg: JointState, name: str):
        if name not in msg.name:
            return None, None
        i = msg.name.index(name)
        q = msg.position[i] if i < len(msg.position) else None
        dq = msg.velocity[i] if i < len(msg.velocity) else None
        return q, dq

    def _tick(self):
        t = time.time() - self.t0
        if t > self.duration:
            self._save_and_exit()
            return

        # command (position) sine
        w = 2.0 * math.pi * self.freq
        qcmd1 = self.amp * math.sin(w * t)
        qcmd2 = -self.amp * math.sin(w * t)

        # treat "acceleration input" as commanded ddq from the sine
        u1 = -self.amp * (w**2) * math.sin(w * t)
        u2 = +self.amp * (w**2) * math.sin(w * t)

        # publish command
        msg = Float64MultiArray()
        msg.data = [float(qcmd1), float(qcmd2)]
        self.pub.publish(msg)

        # log only if we actually received joint_states
        if self.last_js is None:
            return

        q1, dq1 = self._get_joint(self.last_js, self.j1)
        q2, dq2 = self._get_joint(self.last_js, self.j2)
        if q1 is None or q2 is None:
            return
        if dq1 is None: dq1 = 0.0
        if dq2 is None: dq2 = 0.0

        self.rows.append([t, q1, q2, dq1, dq2, qcmd1, qcmd2, u1, u2])

    def _save_and_exit(self):
        # write CSV
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t','q1','q2','dq1','dq2','qcmd1','qcmd2','u1','u2'])
            w.writerows(self.rows)

        self.get_logger().info(f"Done. rows={len(self.rows)} CSV saved: {self.csv_path}")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = LeaderLog()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
