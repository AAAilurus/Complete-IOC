#!/usr/bin/env python3
"""
Unified IOC Data Logger (No pandas dependency)
Records both leader and follower in ONE CSV file
"""
import numpy as np
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class UnifiedLogger(Node):
    def __init__(self):
        super().__init__("unified_ioc_logger")
        
        self.declare_parameter("output_file", "/root/so100_ws/ioc_results/unified_data.csv")
        self.declare_parameter("rate_hz", 50.0)
        
        self.output_file = self.get_parameter("output_file").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz
        
        # Subscribe to both robots
        self.sub_leader = self.create_subscription(
            JointState, '/so100/joint_states', self.on_leader_state, 10)
        self.sub_follower = self.create_subscription(
            JointState, '/so101/joint_states', self.on_follower_state, 10)
        
        # Data storage
        self.data_lines = []
        self.start_time = time.time()
        self.step_count = 0
        
        # Load learned Q info if available
        try:
            data = np.load('/root/so100_ws/ioc_results/learned_Q_discrete.npz')
            self.Q_star = data['Q_star']
            self.Q_learned = data['Q']
            self.K_star = data['K_star']
            self.K_learned = data['K']
            self.R = data['R']
            self.has_ioc_data = True
            self.get_logger().info("✓ Loaded IOC learning results")
        except:
            self.has_ioc_data = False
            self.get_logger().warn("⚠ No IOC data found, will use default header")
        
        # Timer for logging
        self.timer = self.create_timer(self.dt, self.log_step)
        
        # Latest states
        self.leader_state = None
        self.follower_state = None
        
        # CSV header written flag
        self.header_written = False
        
        self.get_logger().info("="*60)
        self.get_logger().info("UNIFIED IOC DATA LOGGER")
        self.get_logger().info(f"Output: {self.output_file}")
        self.get_logger().info(f"Rate: {self.rate_hz} Hz")
        self.get_logger().info("Recording both leader and follower...")
        self.get_logger().info("="*60)
    
    def on_leader_state(self, msg):
        try:
            idx1 = msg.name.index('Shoulder_Pitch')
            idx2 = msg.name.index('Elbow')
            self.leader_state = {
                'q1': msg.position[idx1],
                'q2': msg.position[idx2],
                'dq1': msg.velocity[idx1] if len(msg.velocity) > idx1 else 0.0,
                'dq2': msg.velocity[idx2] if len(msg.velocity) > idx2 else 0.0,
            }
        except:
            pass
    
    def on_follower_state(self, msg):
        try:
            idx1 = msg.name.index('Shoulder_Pitch')
            idx2 = msg.name.index('Elbow')
            self.follower_state = {
                'q1': msg.position[idx1],
                'q2': msg.position[idx2],
                'dq1': msg.velocity[idx1] if len(msg.velocity) > idx1 else 0.0,
                'dq2': msg.velocity[idx2] if len(msg.velocity) > idx2 else 0.0,
            }
        except:
            pass
    
    def log_step(self):
        current_time = time.time() - self.start_time
        
        # Log leader
        if self.leader_state is not None:
            line = f"{current_time:.4f},{self.step_count},leader,{self.leader_state['q1']:.10f},{self.leader_state['q2']:.10f},{self.leader_state['dq1']:.10f},{self.leader_state['dq2']:.10f}"
            self.data_lines.append(line)
        
        # Log follower
        if self.follower_state is not None:
            line = f"{current_time:.4f},{self.step_count},follower,{self.follower_state['q1']:.10f},{self.follower_state['q2']:.10f},{self.follower_state['dq1']:.10f},{self.follower_state['dq2']:.10f}"
            self.data_lines.append(line)
        
        self.step_count += 1
        
        if self.step_count % 100 == 0:
            self.get_logger().info(f"Logged {self.step_count} steps ({len(self.data_lines)} records)")
    
    def save_data(self):
        with open(self.output_file, 'w') as f:
            # Write header
            f.write("# UNIFIED IOC DATA LOG\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Sample rate: {self.rate_hz} Hz\n")
            f.write("#\n")
            
            if self.has_ioc_data:
                f.write("# ========== LEADER (Q*, R*, K*) ==========\n")
                f.write(f"# Q_star_diag: {np.diag(self.Q_star).tolist()}\n")
                f.write(f"# R_diag: {np.diag(self.R).tolist()}\n")
                f.write(f"# K_star:\n")
                f.write(f"#   [{self.K_star[0,0]:.10f}, {self.K_star[0,1]:.10f}, {self.K_star[0,2]:.10f}, {self.K_star[0,3]:.10f}]\n")
                f.write(f"#   [{self.K_star[1,0]:.10f}, {self.K_star[1,1]:.10f}, {self.K_star[1,2]:.10f}, {self.K_star[1,3]:.10f}]\n")
                f.write("#\n")
                f.write("# ========== FOLLOWER (Q_learned, K_learned) ==========\n")
                f.write(f"# Q_learned_diag: {np.diag(self.Q_learned).tolist()}\n")
                f.write(f"# K_learned:\n")
                f.write(f"#   [{self.K_learned[0,0]:.10f}, {self.K_learned[0,1]:.10f}, {self.K_learned[0,2]:.10f}, {self.K_learned[0,3]:.10f}]\n")
                f.write(f"#   [{self.K_learned[1,0]:.10f}, {self.K_learned[1,1]:.10f}, {self.K_learned[1,2]:.10f}, {self.K_learned[1,3]:.10f}]\n")
                f.write("#\n")
                f.write(f"# ========== IOC RESULTS ==========\n")
                f.write(f"# K_error: {np.linalg.norm(self.K_learned - self.K_star, 'fro'):.10f}\n")
                f.write(f"# Q_error: {np.linalg.norm(self.Q_learned - self.Q_star, 'fro'):.10f}\n")
                f.write("#\n")
            
            f.write("# ========== DATA ==========\n")
            f.write("time,step,robot,q1,q2,dq1,dq2\n")
            
            # Write data
            for line in self.data_lines:
                f.write(line + "\n")
        
        # Count leader and follower
        leader_count = sum(1 for line in self.data_lines if ',leader,' in line)
        follower_count = sum(1 for line in self.data_lines if ',follower,' in line)
        
        self.get_logger().info(f"\n✓ Saved {len(self.data_lines)} records to {self.output_file}")
        self.get_logger().info(f"  Leader records: {leader_count}")
        self.get_logger().info(f"  Follower records: {follower_count}")

def main():
    rclpy.init()
    node = UnifiedLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
