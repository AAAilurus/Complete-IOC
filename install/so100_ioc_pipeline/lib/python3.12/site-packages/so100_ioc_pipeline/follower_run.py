#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class FollowerRun(Node):
    def __init__(self):
        super().__init__('follower_run')

        self.ns = self.declare_parameter('ns', '/so101').value
        self.j1 = self.declare_parameter('j1', 'Shoulder_Pitch').value
        self.j2 = self.declare_parameter('j2', 'Elbow').value
        self.rate = float(self.declare_parameter('rate_hz', 50.0).value)
        self.duration = float(self.declare_parameter('duration_s', 25.0).value)
        self.kfile = self.declare_parameter('kfile', '/tmp/ioc_result.npz').value
        self.scale = float(self.declare_parameter('scale', 1.0).value)

        self.K = np.load(self.kfile)['K']  # 2x4

        self.js_topic = f'{self.ns}/joint_states'
        self.cmd_topic = f'{self.ns}/arm_position_controller/commands'

        self.q = None
        self.dq = None

        self.create_subscription(JointState, self.js_topic, self.cb_js, 50)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self.dt = 1.0/self.rate
        self.steps = int(self.duration*self.rate)
        self.k = 0
        self.timer = self.create_timer(self.dt, self.tick)

        self.get_logger().info(f'Follower K file: {self.kfile}')
        self.get_logger().info(f'Pub: {self.cmd_topic} | Sub: {self.js_topic}')

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
        self.q = (q1, q2)
        self.dq = (dq1, dq2)

    def tick(self):
        if self.k >= self.steps:
            self.get_logger().info('Follower done.')
            rclpy.shutdown()
            return

        if self.q is None:
            return

        q1, q2 = self.q
        dq1, dq2 = self.dq
        x = np.array([q1, q2, dq1, dq2]).reshape(4, 1)

        u = -self.K @ x
        u = self.scale * u

        # qcmd = q + u  (because your "u" in the log is qcmd-q)
        qcmd = np.array([q1, q2]).reshape(2, 1) + u

        msg = Float64MultiArray()
        msg.data = [float(qcmd[0]), float(qcmd[1])]
        self.pub.publish(msg)

        self.k += 1

def main():
    rclpy.init()
    node = FollowerRun()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
