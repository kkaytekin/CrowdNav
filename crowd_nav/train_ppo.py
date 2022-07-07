import sys
import logging
import argparse
import configparser
import os
import shutil
from crowd_nav.policy.sarl import SARL_PPO
import torch
import gym
import git
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.ppo import PPO
import re

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--timesteps', type=int, default=1000000)
    
    args = parser.parse_args()

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')
    

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    # policy.actor.to(device)
    # policy.critic.to(device)
    policy.actor_critic.to(device)
    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    # actor_weight_file = os.path.join(args.output_dir, 'actor_model.pth')
    # critic_weight_file = os.path.join(args.output_dir, 'critic_model.pth')
    actor_critic_weight_file = os.path.join(args.output_dir, 'actor_critic_model.pth')
    output_log_file = os.path.join(args.output_dir, 'output.log')
    # reinforcement learning
    if args.resume:
        if not os.path.exists(actor_critic_weight_file) or not os.path.exists(output_log_file):
            logging.error('actor/critic weights and previous output log does not exist')
            # policy.actor.load_state_dict(torch.load(actor_weight_file))
            # policy.critic.load_state_dict(torch.load(critic_weight_file))
        else:
            policy.actor_critic.load_state_dict(torch.load(actor_critic_weight_file))
            logging.info('Load actor/critic learning trained weights. Resume training')
            print(output_log_file)
            with open(output_log_file, 'r') as file:
                log = file.read()
            
            train_pattern = r"TRAIN in epoch (?P<epoch>\d+) has avg. loss: (?P<loss>\d+.\d+), " \
                            r"avg. episodic return: (?P<reward>[-+]?\d+.\d+), timesteps accumulated: (?P<timesteps>\d+)"
            train_epoch = []
            for r in re.findall(train_pattern, log):
                train_epoch.append(int(r[0]))
            num_policy_itr = train_epoch[-1]



    ppo = PPO(train_config, robot, env, policy, args.output_dir, device)
    ppo.learn(args.timesteps, num_policy_itr)

if __name__ == '__main__':
    main()
