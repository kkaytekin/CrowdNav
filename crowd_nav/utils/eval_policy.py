from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
import numpy as np
import torch
from crowd_sim.envs.utils.info import *
import logging

def run_k_episodes(gamma, allow_backward, with_constant_speed, k, policy, env, robot, phase, device, episode=None, print_failure=False):
    robot.policy.set_phase(phase)
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    gamma = 0.95
    for i in range(k):
        ob = env.reset(phase)
        done = False
        states = []
        actions = []
        rewards = []
        while not done:
            state = JointState(robot.get_full_state(), ob)
            batch_states = torch.cat([torch.Tensor([state.self_state + human_state]).to(device)
                                            for human_state in state.human_states], dim=0)
            rotated_batch_input = policy.rotate(batch_states).unsqueeze(0)
            output = policy.actor_critic(rotated_batch_input)
            action_probs = output[:, 1:]
            action_id = action_probs.argmax(dim = 1)
            action = policy.action_space[action_id]
            if not with_constant_speed:
                pass
            else:
                self_state = state.self_state
                theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
                if allow_backward:
                    vx = np.cos(theta) * self_state.v_pref * 0.5
                    vy = np.sin(theta) * self_state.v_pref * 0.5
                else:
                    vx = np.cos(theta) * self_state.v_pref
                    vy = np.sin(theta) * self_state.v_pref
                finalized_action = np.array([action.vx + vx, action.vy + vy])
                v_norm = np.linalg.norm(finalized_action)
                finalized_action = finalized_action / v_norm * self_state.v_pref
                action = ActionXY(finalized_action[0], 
                        finalized_action[1])
            ob, reward, done, info = env.step(action)

            states.append(robot.policy.last_state)
            actions.append(action)
            rewards.append(reward)

            if isinstance(info, Danger):
                too_close += 1
                min_dist.append(info.min_dist)

        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env.time_limit)
        else:
            raise ValueError('Invalid end signal from environment')

        cumulative_rewards.append(sum([pow(gamma, t) * reward for t, reward in enumerate(rewards)]))

    success_rate = success / k
    collision_rate = collision / k
    assert success + collision + timeout == k
    avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit

    extra_info = '' if episode is None else 'in episode {} '.format(episode)
    logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                    format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                        average(cumulative_rewards)))
    if phase in ['val', 'test']:
        num_step = sum(success_times + collision_times + timeout_times) / robot.time_step
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                        too_close / num_step, average(min_dist))

    if print_failure:
        logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
        logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0