from crowd_sim.envs.utils.info import ReachGoal
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
import time
import os
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
import logging
from crowd_sim.envs.utils.info import *
from torch.distributions import Categorical

class PPO:
    def __init__(self, robot, env, model, output_dir):
        # First, we need to know in which environemnt we are working on
        self.env = env
        self.act_dim = 2
        
        self._init_hyperparameters()
        self.model = model
        self.robot = robot
        ## Algorithm Step 1 
        ## Initialize actor and critic network
        # self.actor = model.actor
        # self.critic = model.critic
        self.actor_critic = model.actor_critic
        # Create our variable for the matrix.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        # ## Initialize the actor optimizer
        # self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        
        # ## Initialize the critic optimizer
        # self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam([
        #                 {'params': self.actor.parameters(), 'lr': self.lr},
        #                 {'params': self.critic.parameters(), 'lr': self.lr}
        #             ])
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.lr)
        # This logger will help us with printing out summaries of each iteration

        self.logger = {
			'delta_t': time.time_ns(), 
			't': 0,                 # timesteps so far
			'itr': 0,               # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}
        
        self.render = False
        self.render_every_i = 50
        self.save_freq = 10

        # self.actor_weight_file = os.path.join(output_dir, 'actor_model.pth')
        # self.critic_weight_file = os.path.join(output_dir, 'critic_model.pth')
        self.actor_critic_weight_file = os.path.join(output_dir, 'actor_critic_model.pth')
    def learn(self, total_time_steps):
        print(f"Learning... Running {self.max_time_steps_per_episode} timesteps per episode, ", end='')
        print(f"{self.time_steps_per_batch} timesteps per batch for a total of {total_time_steps} timesteps")

        t = 0 # Time steps which we have generated so far
        itr = 0 
        while t < total_time_steps:
            ## Algorithm Step 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            V, _, _ = self.evaluate(batch_obs, batch_acts)

            ## Algorithm Step 5
            ## Calculate advantage
            # print(batch_rtgs.shape)
            # print(V.shape)
            A_k = batch_rtgs - V.detach()
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs, dist_entropy = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                ## The actor loss according to the pseudocode is
                # actor_loss = (-torch.min(surr1, surr2)).mean() ## We take mean() here because we have summation and 1/N in the equation
                # # Calculate gradients and perform backward propagation for actor 
                # # network
                # self.actor_optim.zero_grad()
                # actor_loss.backward(retain_graph=True)
                # self.actor_optim.step()
                actor_loss = (-torch.min(surr1, surr2))
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # critic_loss = nn.MSELoss()(V, batch_rtgs)
                # # Calculate gradients and perform backward propagation for critic network    
                # self.critic_optim.zero_grad()    
                # critic_loss.backward()    
                # self.critic_optim.step()

                self.logger['actor_losses'].append(loss.detach())

            # Calculate how many timesteps we collected this batch   
            t += np.sum(batch_lens)
            itr += 1
            self.logger['t'] = t
            self.logger['itr'] = itr
        
            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if itr % self.save_freq == 0:
                # torch.save(self.actor.state_dict(), self.actor_weight_file)
                # torch.save(self.critic.state_dict(), self.critic_weight_file)
                torch.save(self.actor_critic.state_dict(), self.actor_critic_weight_file)


    def _init_hyperparameters(self):
        self.time_steps_per_batch = 5000
        self.max_time_steps_per_episode = 110
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 4e-5

    def rollout(self):
        ## Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rewards = []            # batch rewards
        batch_rewards_to_go = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch
        
        batch_obs = torch.Tensor(batch_obs)            
        batch_acts = torch.Tensor(batch_acts)  
        batch_log_probs = torch.Tensor(batch_log_probs)  
        # batch_rewards = torch(batch_rewards)  
        # batch_rewards_to_go = torch.Tensor(batch_rewards_to_go)  
        # batch_lens = torch(batch_lens)  
        t = 0

        # success = 0
        # collision = 0
        # timeout = 0

        # success_times = []
        # collision_times = []
        # timeout_times = []
        # k = 0
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
                self_state = state.self_state
                theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
                vx = np.cos(theta) * self_state.v_pref * 0.5
                vy = np.sin(theta) * self_state.v_pref * 0.5
                batch_states = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.model.device)
                                              for human_state in state.human_states], dim=0)
                rotated_batch_input = self.model.rotate(batch_states).unsqueeze(0)
                batch_obs = torch.cat((torch.Tensor(batch_obs), rotated_batch_input), dim = 0)
                # print(batch_obs.shape)
                action_id, log_prob = self.get_action(rotated_batch_input)
                # print(self.model.action_space[action[0].item()])
                action = ActionXY(self.model.action_space[action_id[0].item()].vx + vx, 
                self.model.action_space[action_id[0].item()].vy + vy)
                
                obs, reward, done, info = self.env.step(action)
                # self.env.render()
                ## Collect reward, action, and log probability
                ep_rewards.append(reward)
                # ep_rewards = torch.cat((ep_rewards, reward), dim = 0)
                batch_acts = torch.cat((batch_acts, torch.Tensor(action_id)), dim = 0)
                batch_log_probs = torch.cat((batch_log_probs, log_prob), dim = 0)
                # batch_acts.append(action)
                # batch_log_probs.append(log_prob)

                if done: ## Either the episode ends or finishes, we break the for loop
                    # k += 1
                    # if isinstance(info, ReachGoal):
                    #     success += 1
                    #     success_times.append(self.env.global_time)
                    # elif isinstance(info, Collision):
                    #     collision += 1
                    #     collision_times.append(self.env.global_time)
                    # elif isinstance(info, Timeout):
                    #     timeout += 1
                    #     timeout_times.append(self.env.time_limit)
                    # else:
                    #     raise ValueError('Invalid end signal from environment')
                    break
            if self.render and (self.logger['itr'] % self.render_every_i == 0) and len(batch_lens) == 0:
                self.env.render('video')
            ## Collect episodic length and rewards
            batch_lens.append(ep + 1) ## ep start from 0
            batch_rewards.append(ep_rewards)
        
        # success_rate = success / k
        # collision_rate = collision / k
        # assert success + collision + timeout == k
        # avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        # Reshape data as tensors in the shape specified before returning
        # batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        # batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        # batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # Algorithm Step 4
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        # Return the batch data
        # logging.info('success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
        #              format(success_rate, collision_rate, avg_nav_time, average(batch_rewards_to_go)))
        self.logger['batch_rews'] = batch_rewards
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens

    def get_action(self, obs, continuous = False):
        ## First, query the actor network for a mean action
        if continuous:
            mean = self.actor(obs)
            mean_map = mean.clone()
            mean_map[:, 1] = mean_map[:, 1] * torch.pi

            ## Create the Multivariate Normal Distribution
            dist = MultivariateNormal(mean_map, self.cov_mat) ## consider it as a normal distribution in high dimensional space

            ## Sample an action from the distribition and get its log-probability
            action = dist.sample()
            v = action[0, 0]
            theta = action[0, 1]

            action[0, 0] = v * np.cos(theta)
            action[0, 1] = v * np.sin(theta)
            log_prob = dist.log_prob(action)
        else:
            action_probs = self.actor_critic(obs)
            dist = Categorical(action_probs[:, 1:])
            action = dist.sample()
            log_prob = dist.log_prob(action)


        return action.detach().numpy(), log_prob.detach()

    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []

        ## Note that we calculate the reward-to-go typically from the last state
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0 ## This accumulative discounted reward

            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0,  discounted_reward) ## make sure the order is still from 1 to k not k to 1, so we always "INSERT" new discounted reward in the front

        ## Convert rewards-to-go into tensor
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype = torch.float)

        return batch_rewards_to_go

    def evaluate(self, batch_obs, batch_acts, continuous = False):
        ## Query the critic network for a value V for each observation in batch_obs
        # V = self.critic(batch_obs).squeeze()
        V = self.actor_critic(batch_obs)[:, 0].squeeze()
        if continuous:
            # Calculate the log probabilities of batch actions using most 
            # recent actor network.
            # This segment of code is similar to that in get_action()
            mean = self.actor(batch_obs)
            mean_map = mean.clone()
            mean_map[:, 1] = mean_map[:, 1] * torch.pi

            dist = MultivariateNormal(mean_map, self.cov_mat)
            log_probs = dist.log_prob(batch_acts) ## Note that we don't sample action here because we already did it in rollout()
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
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t']
        i_so_far = self.logger['itr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        # print(flush=True)
        logging.info("-------------------- Iteration #{} --------------------".format(i_so_far))
        logging.info("Average Episodic Length: {}".format(avg_ep_lens))
        logging.info("Average Episodic Return: {}".format(avg_ep_rews))
        logging.info("Average Loss: {}".format(avg_actor_loss))
        logging.info("Timesteps So Far: {}".format(t_so_far))
        logging.info("Iteration took: {} secs".format(delta_t))
        logging.info("------------------------------------------------------")
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

if __name__ == '__main__':
    import gym 
    env = gym.make('Pendulum-v1')
    model = PPO(env)
    model.learn(5000000)