import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class PeakMonitor(Node):
    def __init__(self, joint, window):
        super().__init__('joint_peak_monitor')
        self.joint, self.window = joint, window
        self.peak, self.first, self.last = None, None, None
        self.t0 = time.time()
        self.create_subscription(JointState, '/joint_states', self.cb, 10)
        self.create_timer(0.5, self.check)
        self.get_logger().info(f'monitoring {joint} for {window:.0f}s ... send the gesture command now')

    def cb(self, msg):
        if self.joint not in msg.name:
            return
        v = msg.position[list(msg.name).index(self.joint)]
        self.first = v if self.first is None else self.first
        self.last = v
        self.peak = v if self.peak is None else max(self.peak, v, key=abs)

    def check(self):
        if time.time() - self.t0 > self.window:
            print(f'RESULT joint={self.joint} start={self.first:.3f} peak={self.peak:.3f} end={self.last:.3f}', flush=True)
            raise SystemExit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--joint', default='crane_plus_joint3')
    p.add_argument('--window', type=float, default=6.0)
    a, _ = p.parse_known_args()
    rclpy.init()
    node = PeakMonitor(a.joint, a.window)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
