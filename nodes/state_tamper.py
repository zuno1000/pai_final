#!/usr/bin/env python3
import argparse
import collections

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class StateTamper(Node):
    def __init__(self, mode, joint, bias, sigma, delay_s, in_topic, out_topic):
        super().__init__('state_tamper')
        self.mode, self.joint, self.bias, self.sigma = mode, joint, bias, sigma
        self.rng = np.random.default_rng(0)
        self.buf = collections.deque(maxlen=max(1, int(delay_s * 100)))
        self.pub = self.create_publisher(JointState, out_topic, 10)
        self.create_subscription(JointState, in_topic, self.callback, 10)
        self.count = 0
        self.get_logger().info(f'mode={mode} joint={joint} bias={bias} sigma={sigma} delay={delay_s}s  {in_topic} -> {out_topic}')

    def callback(self, msg):
        pos = np.array(msg.position, dtype=float)
        if self.mode == 'bias':
            if self.joint in msg.name:
                pos[list(msg.name).index(self.joint)] += self.bias
        elif self.mode == 'noise':
            pos = pos + self.rng.normal(0.0, self.sigma, pos.shape)
        elif self.mode == 'delay':
            self.buf.append(pos.copy())
            pos = self.buf[0]

        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = pos.tolist()
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        self.pub.publish(out)
        self.count += 1
        if self.count % 500 == 0:
            self.get_logger().info(f'relayed {self.count} messages')


def main():
    parser = argparse.ArgumentParser(description='Tamper /joint_states and republish')
    parser.add_argument('--mode', choices=['none', 'bias', 'noise', 'delay'], default='none')
    parser.add_argument('--joint', default='crane_plus_joint3')
    parser.add_argument('--bias', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--in-topic', default='/joint_states')
    parser.add_argument('--out-topic', default='/joint_states_tampered')
    args, _unknown = parser.parse_known_args()

    rclpy.init()
    node = StateTamper(args.mode, args.joint, args.bias, args.sigma, args.delay, args.in_topic, args.out_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
