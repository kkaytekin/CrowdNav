import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                # actor_weights = os.path.join(args.model_dir, 'actor_model.pth')
                # critic_weights = os.path.join(args.model_dir, 'critic_model.pth')
                actor_critic_weights = os.path.join(args.model_dir, 'actor_critic_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        # policy.get_model().load_state_dict(torch.load(model_weights))
        # policy.actor.load_state_dict(torch.load(actor_weights))
        policy.actor_critic.load_state_dict(torch.load(actor_critic_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    # explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.actor_critic.to(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    obs = env.reset(args.phase, args.test_case)
    done = False
   
    while not done:
        state = JointState(robot.get_full_state(), obs)
        self_state = state.self_state
        theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
        vx = np.cos(theta) * self_state.v_pref * 0.5
        vy = np.sin(theta) * self_state.v_pref * 0.5

        batch_states = torch.cat([torch.Tensor([state.self_state + human_state]).to(device)
                                        for human_state in state.human_states], dim=0)
        rotated_batch_input = policy.rotate(batch_states).unsqueeze(0)
        # print(rotated_batch_input)
        output = policy.actor_critic(rotated_batch_input)
        
        print(output)
        action_probs = output[:, 1:]
        action_id = action_probs.argmax(dim = 1)
        print("Best action: {} with probability: {}".format(action_id, action_probs[0, action_id]))
        action = policy.action_space[action_id]
        print(action)
        action = ActionXY(action.vx + vx, 
                action.vy + vy)
        obs, reward, done, info = env.step(action)
        print(reward)
    if args.traj:
        env.render('traj', args.video_file)
    else:
        env.render('video', args.video_file)

    logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    if robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))


if __name__ == '__main__':
    main()
