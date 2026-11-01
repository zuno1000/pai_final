#!/usr/bin/env python3
import argparse
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.common.policies.act.modeling_act import ACTPolicy

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'
GRIPPER_HOME = -0.5236
DEFAULT_DURATIONS = {'bow': 6.0, 'wave': 16.0, 'nod': 9.0}


class GestureDispatcher(Node):
    def __init__(self, model_paths, state_topic, durations, n_action_steps=8, rate_hz=10.0, obs_noise=0.0,
                 lag_s=0.0, lag_gestures=('wave',)):
        super().__init__('gesture_dispatcher')
        self.lag_s = lag_s
        self.lag_gestures = set(lag_gestures)
        self.history = [] 
        self.policies = {}
        for name, path in model_paths.items():
            self.get_logger().info(f'loading policy "{name}" from {path}')
            policy = ACTPolicy.from_pretrained(path)
            policy.eval()
            policy.to(torch.device('cpu'))
            policy.config.n_action_steps = max(1, min(n_action_steps, policy.config.chunk_size))
            self.policies[name] = policy
        self.durations = durations
        self.obs_noise = obs_noise
        self.rng = np.random.default_rng(0)
        self.period = 1.0 / rate_hz

        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, state_topic, self.on_joint_state, 10)
        self.create_subscription(String, '/gesture_cmd', self.on_command, 10)

        self.latest_obs = None
        self.state = 'IDLE' 
        self.active = None
        self.t_start = 0.0
        self.step_count = 0
        self.create_timer(self.period, self.tick)
        self.get_logger().info(f'ready. gestures: {list(self.policies)}  state topic: {state_topic}  '
                               f'n_action_steps={n_action_steps} rate={rate_hz}Hz obs_noise={obs_noise} '
                               f'lag={lag_s}s for {sorted(self.lag_gestures)}')

    # ---- subscriptions -------------------------------------------------
    def on_joint_state(self, msg):
        pos = dict(zip(msg.name, msg.position))
        arm = [pos.get(j, 0.0) for j in ARM_JOINT_NAMES]
        grip = pos.get(GRIPPER_JOINT_NAME, 0.0)
        self.latest_obs = np.array(arm + [grip], dtype=np.float32)
        if self.lag_s > 0:
            now = time.time()
            self.history.append((now, self.latest_obs))
            while self.history and now - self.history[0][0] > self.lag_s + 0.2:
                self.history.pop(0)

    def lagged_obs(self):
        """joint positions observed lag_s seconds ago (oldest sample within the window)."""
        if self.lag_s <= 0 or not self.history:
            return self.latest_obs
        now = time.time()
        for t, o in self.history:
            if now - t <= self.lag_s:
                return o
        return self.history[-1][1]

    def on_command(self, msg):
        name = msg.data.strip().lower()
        if name not in self.policies:
            self.get_logger().warn(f'unknown gesture "{name}" (available: {list(self.policies)})')
            return
        if self.state != 'IDLE':
            self.get_logger().warn(f'busy ({self.state}); ignored "{name}"')
            return
        self.policies[name].reset()
        self.active = name
        self.state = 'RUNNING'
        self.t_start = time.time()
        self.step_count = 0
        self.get_logger().info(f'>>> start gesture: {name}')

    # ---- publishing ----------------------------------------------------
    def send(self, arm_positions, gripper_position, sec=0.1):
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in arm_positions]
        pt.time_from_start.sec = int(sec)
        pt.time_from_start.nanosec = int((sec - int(sec)) * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)

        jg = JointTrajectory()
        jg.joint_names = [GRIPPER_JOINT_NAME]
        pg = JointTrajectoryPoint()
        pg.positions = [float(gripper_position)]
        pg.time_from_start.sec = int(sec)
        pg.time_from_start.nanosec = int((sec - int(sec)) * 1e9)
        jg.points.append(pg)
        self.gripper_pub.publish(jg)

    # ---- main loop -----------------------------------------------------
    def tick(self):
        if self.state == 'IDLE' or self.latest_obs is None:
            return
        elapsed = time.time() - self.t_start

        if self.state == 'RUNNING':
            if elapsed > self.durations.get(self.active, 4.0):
                self.get_logger().info(f'<<< gesture "{self.active}" done ({self.step_count} steps); returning upright')
                self.state = 'RESETTING'
                self.t_start = time.time()
                self.send([0.0, 0.0, 0.0, 0.0], GRIPPER_HOME, sec=1.0)
                return
            obs_np = self.latest_obs.copy()
            env_np = self.lagged_obs().copy() if self.active in self.lag_gestures else obs_np.copy()
            if self.obs_noise > 0:
                obs_np += self.rng.normal(0.0, self.obs_noise, obs_np.shape).astype(np.float32)
                env_np += self.rng.normal(0.0, self.obs_noise, env_np.shape).astype(np.float32)
            obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            env = torch.tensor(env_np, dtype=torch.float32).unsqueeze(0)
            batch = {'observation.state': obs, 'observation.environment_state': env}
            with torch.no_grad():
                action = self.policies[self.active].select_action(batch).cpu().numpy().flatten()
            self.send(action[:4], action[4], sec=self.period)
            self.step_count += 1
            if self.step_count % 5 == 0:
                print(f'[{self.active}] t={elapsed:4.1f}s action={np.round(action, 3)}', flush=True)

        elif self.state == 'RESETTING':
            if elapsed > 1.5:
                self.state = 'IDLE'
                self.active = None
                self.get_logger().info('ready for next command')


def main():
    parser = argparse.ArgumentParser(description='Gesture dispatcher (ACT policies) for CRANE+ V2')
    parser.add_argument('--models-root', default='/home/ubuntu/checkpoints/train')
    parser.add_argument('--bow', default='crane_plus_pekori_act_500', help='sub-directory of the bow policy')
    parser.add_argument('--wave', default='crane_plus_wave_lag_act_2000')
    parser.add_argument('--nod', default='crane_plus_nod_act')
    parser.add_argument('--state-topic', default='/joint_states',
                        help='observation topic (use /joint_states_tampered with state_tamper.py)')
    parser.add_argument('--only', default='', help='comma separated subset of gestures to load, e.g. "bow,wave"')
    parser.add_argument('--n-action-steps', type=int, default=8,
                        help='how many actions of each predicted chunk to execute before re-planning (1 = replan every step)')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='control rate [Hz]; the dataset was converted at 20 fps and delta_timestamps step is 0.1 s')
    parser.add_argument('--obs-noise', type=float, default=0.02,
                        help='std [rad] of Gaussian noise added to the observation. With state-only observation the '
                             'upright pose (state 0 -> action 0) is a fixed point of the learned policy and action '
                             'chunking alone did not escape it; dithering with 0.02 rad does. Use 0 to disable.')
    parser.add_argument('--lag', type=float, default=0.5,
                        help='seconds; feed the joint positions this long ago as observation.environment_state '
                             '(for policies trained on datasets made with scripts/rosbag2lerobot_lag.py, use 0.5)')
    parser.add_argument('--lag-gestures', default='wave',
                        help='comma separated gestures whose policy was trained with the lagged environment_state')
    args, _unknown = parser.parse_known_args()

    model_paths = {'bow': f'{args.models_root}/{args.bow}',
                   'wave': f'{args.models_root}/{args.wave}',
                   'nod': f'{args.models_root}/{args.nod}'}
    if args.only:
        keep = {s.strip() for s in args.only.split(',')}
        model_paths = {k: v for k, v in model_paths.items() if k in keep}

    rclpy.init()
    node = GestureDispatcher(model_paths, args.state_topic, DEFAULT_DURATIONS,
                             n_action_steps=args.n_action_steps, rate_hz=args.rate, obs_noise=args.obs_noise,
                             lag_s=args.lag,
                             lag_gestures=[s.strip() for s in args.lag_gestures.split(',') if s.strip()])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
