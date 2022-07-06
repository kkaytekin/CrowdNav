from crowd_sim.envs.utils.info import ReachGoal
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
from torch import nn, tensor
import numpy as np
import time
import os
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
import logging
from crowd_sim.envs.utils.info import *
from torch.distributions import Categorical

class PPO:
    def __init__(self, config, robot, env, model, output_dir, device):
        # First, we need to know in which environemnt we are working on
        self.env = env
        # self.act_dim = 2
        self.configure(config)
        self.model = model
        self.robot = robot
        self.device = device
        ## Algorithm Step 1 
        ## Initialize actor and critic network
        self.actor_critic = model.actor_critic
        # Create our variable for the matrix.
        # self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Create the covariance matrix
        # self.cov_mat = torch.diag(self.cov_var)

        # ## Initialize the actor/critic optimizer
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.lr)
        # This logger will help us with printing out summaries of each iteration

        self.logger = {
			't': 0,                 # timesteps so far
			'epoch': 0,               # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}
        
        ## For debugging
        self.render = False
        self.render_every_i = 50

        self.actor_critic_weight_file = os.path.join(output_dir, 'actor_critic_model.pth')
    
    def configure(self, config):
        self.with_constant_speed = config.getboolean('ppo', 'with_constant_speed')
        self.allow_backward = config.getboolean('ppo', 'allow_backward')
        self.use_gae = config.getboolean('ppo', 'use_gae')
        self.time_steps_per_batch = config.getint('ppo', 'time_steps_per_batch')
        self.max_time_steps_per_episode = config.getint('ppo', 'max_time_steps_per_episode')
        self.gamma = config.getfloat('ppo', 'gamma')
        self.n_updates_per_iteration = config.getint('ppo', 'n_updates_per_iteration')
        self.clip = config.getfloat('ppo', 'clip')
        self.lr = config.getfloat('ppo', 'lr')
        self.lmbda = config.getfloat('ppo', 'lmbda')
        self.save_freq = config.getint('ppo', 'save_freq')
        logging.info('PPO training {} constant speed to goal'.format('w/' if self.with_constant_speed else 'w/o'))
        logging.info('{} GAE'.format('Activate' if self.use_gae else 'Deactivate'))
    
    def compute_advantages(self, batch_rtgs, V, batch_rewards, masks):
        if not self.use_gae:
            A_k = batch_rtgs - V
        else:
            returns = []
            gae = 0
            for i in reversed(range(len(batch_rewards) - 1)):
                delta = batch_rewards[i] + self.gamma * V[i + 1] * masks[i] - V[i]
                gae = delta + self.gamma * self.lmbda * masks[i] * gae
                returns.insert(0, gae + V[i])
            A_k = torch.FloatTensor(returns).to(self.device) - V[:-1]
        # Normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        return A_k
    
    def learn(self, total_time_steps):
        t = 0 # Time steps which we have generated so far
        epoch = 0 
        while t < total_time_steps:
            ## Algorithm Step 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rewards, batch_masks = self.rollout()
            
            V, _, _ = self.evaluate(batch_obs, batch_acts)
            ## Algorithm Step 5
            ## Calculate advantage
            A_k = self.compute_advantages(batch_rtgs, V.detach(), batch_rewards, batch_masks)
            if self.use_gae:
                batch_log_probs = batch_log_probs[:-1]
                batch_rtgs = batch_rtgs[:-1]
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs, dist_entropy = self.evaluate(batch_obs, batch_acts)
                if self.use_gae:
                    V = V[:-1]
                    curr_log_probs = curr_log_probs[:-1]
                    dist_entropy = dist_entropy[:-1]
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                ## The actor loss according to the pseudocode is
                actor_loss = (-torch.min(surr1, surr2))
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                self.logger['actor_losses'].append(loss.detach().cpu().numpy())

            # Calculate how many timesteps we collected this batch   
            t += np.sum(batch_lens)
            epoch += 1
            self.logger['t'] = t
            self.logger['epoch'] = epoch
        
            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if epoch % self.save_freq == 0:
                torch.save(self.actor_critic.state_dict(), self.actor_critic_weight_file)


    def rollout(self):
        ## Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rewards = []            # batch rewards
        batch_rewards_to_go = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch
        
        batch_masks = []

        batch_obs = torch.Tensor(batch_obs).to(self.device)     
        batch_acts = torch.Tensor(batch_acts).to(self.device) 
        batch_log_probs = torch.Tensor(batch_log_probs).to(self.device) 

        t = 0

        while t < self.time_steps_per_batch: ## Algorithm Step 2

            ## Reward sequence in "THIS" episode
            ep_rewards = []

            obs = self.env.reset()
            done = False

            for ep in range(self.max_time_steps_per_episode):
                
                ## Increment timesteps so far
                t += 1

                ## Collect observation
                state = JointState(self.robot.get_full_state(), obs)
                
                batch_states = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.model.device)
                                              for human_state in state.human_states], dim=0)
                # print(batch_states)
                rotated_batch_input = self.model.rotate(batch_states).unsqueeze(0)
                rotated_batch_input = rotated_batch_input.to(self.device)

                batch_obs = torch.cat((batch_obs, rotated_batch_input), dim = 0)
                action_id, log_prob = self.get_action(rotated_batch_input)
                action = self.model.action_space[action_id[0].item()]
                if not self.with_constant_speed:
                    pass
                    # Do nothing, just output the action from the actor
                else:
                    self_state = state.self_state
                    theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
                    if self.allow_backward:
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

                obs, reward, done, info = self.env.step(action)
                
                ## Collect reward, action, and log probability
                ep_rewards.append(reward)
                # ep_rewards = torch.cat((ep_rewards, reward), dim = 0)
                batch_acts = torch.cat((batch_acts, torch.Tensor(action_id).to(self.device)), dim = 0)
                batch_log_probs = torch.cat((batch_log_probs, log_prob), dim = 0)
                if not done:
                    mask = 1
                else:
                    mask = 0

                batch_masks.append(mask)

                if done: ## Either the episode ends or finishes, we break the for loop
                    break
            ## For debugging
            if self.render and (self.logger['epoch'] % self.render_every_i == 0) and len(batch_lens) == 0:
                self.env.render('video')

            ## Collect episodic length and rewards
            batch_lens.append(ep + 1) ## ep start from 0
            batch_rewards.append(ep_rewards)
        
        # Algorithm Step 4
        self.logger['batch_rews'] = batch_rewards
        batch_rewards_to_go, batch_rewards = self.compute_rewards_to_go(batch_rewards)      
        self.logger['batch_lens'] = batch_lens
        batch_masks = torch.IntTensor(batch_masks)
        return batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens, batch_rewards, batch_masks

    def get_action(self, obs, continuous = False):
        ## First, query the actor network for a mean action
        if continuous:
            raise NotImplementedError
            # mean = self.actor(obs)
            # mean_map = mean.clone()
            # mean_map[:, 1] = mean_map[:, 1] * torch.pi

            # ## Create the Multivariate Normal Distribution
            # dist = MultivariateNormal(mean_map, self.cov_mat) ## consider it as a normal distribution in high dimensional space

            # ## Sample an action from the distribition and get its log-probability
            # action = dist.sample()
            # v = action[0, 0]
            # theta = action[0, 1]

            # action[0, 0] = v * np.cos(theta)
            # action[0, 1] = v * np.sin(theta)
            # log_prob = dist.log_prob(action)
        else:
            action_probs = self.actor_critic(obs)
            dist = Categorical(action_probs[:, 1:])
            action = dist.sample()
            log_prob = dist.log_prob(action)


        return action.detach().cpu().numpy(), log_prob.detach()

    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []
        batch_rewards_flatten = []
        ## Note that we calculate the reward-to-go typically from the last state
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0 ## This accumulative discounted reward

            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0,  discounted_reward) ## make sure the order is still from 1 to k not k to 1, so we always "INSERT" new discounted reward in the front
                batch_rewards_flatten.insert(0, reward)
        ## Convert rewards-to-go into tensor
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype = torch.float).to(self.device)
        batch_rewards_flatten = torch.tensor(batch_rewards_flatten, dtype = torch.float).to(self.device)

        return batch_rewards_to_go, batch_rewards_flatten

    def evaluate(self, batch_obs, batch_acts, continuous = False):
        ## Query the critic network for a value V for each observation in batch_obs
        V = self.actor_critic(batch_obs)[:, 0].squeeze()
        if continuous:
            raise NotImplementedError
            # Calculate the log probabilities of batch actions using most 
            # recent actor network.
            # This segment of code is similar to that in get_action()
            # mean = self.actor(batch_obs)
            # mean_map = mean.clone()
            # mean_map[:, 1] = mean_map[:, 1] * torch.pi

            # dist = MultivariateNormal(mean_map, self.cov_mat)
            # log_probs = dist.log_prob(batch_acts) ## Note that we don't sample action here because we already did it in rollout()
        else:
            action_probs = self.actor_critic(batch_obs)
            dist = Categorical(action_probs[:, 1:])
            log_probs = dist.log_prob(batch_acts)
            dist_entropy = dist.entropy()
        # Return predicted values V and log probs log_probs
        return V, log_probs, dist_entropy

    def _log_summary(self):
        """
			Print to stdout what we've logged so far in the most recent batch.
			Parameters:
				None
			Return:
				None
		"""
        t_so_far = self.logger['t']
        epoch = self.logger['epoch']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_critic_loss = np.mean([losses.mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 4))
        avg_actor_critic_loss = str(round(avg_actor_critic_loss, 4))

        # Print logging statements
        logging.info('TRAIN in epoch {} has avg. loss: {}, avg. episodic return: {}, timesteps accumulated: {}'.format(epoch, avg_actor_critic_loss, avg_ep_rews, t_so_far))
        # logging.info("-------------------- Iteration #{} --------------------".format(epoch))
        # logging.info("Average Episodic Length: {}".format(avg_ep_lens))
        # logging.info("Average Episodic Return: {}".format(avg_ep_rews))
        # logging.info("Average Loss: {}".format(avg_actor_critic_loss))
        # logging.info("Timesteps So Far: {}".format(t_so_far))
        # logging.info("Iteration took: {} secs".format(delta_t))
        # logging.info("------------------------------------------------------")
        # print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        # print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        # print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        # print(f"Average Loss: {avg_actor_loss}", flush=True)
        # print(f"Timesteps So Far: {t_so_far}", flush=True)
        # print(f"Iteration took: {delta_t} secs", flush=True)
        # print(f"------------------------------------------------------", flush=True)
        # print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
