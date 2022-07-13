import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value

class DQN(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        action_value = self.mlp3(joint_state)
        return action_value

class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights

class SARL_DDQN(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL_DDQN'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl_ddqn', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl_ddqn', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl_ddqn', 'mlp3_dims').split(', ')]
        mlp3_dims[-1] = self.speed_samples * self.rotation_samples + 1
        attention_dims = [int(x) for x in config.get('sarl_ddqn', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl_ddqn', 'with_om')
        with_global_state = config.getboolean('sarl_ddqn', 'with_global_state')
        self.model = DQN(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl_ddqn', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL_DDQN'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        # occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            self.max_action_id = np.random.choice(len(self.action_space))
            max_action = self.action_space[self.max_action_id]
        else:
            rotated_state_input = self.transform(state).unsqueeze(0)
            q_values = self.model(rotated_state_input).squeeze()
            self.action_values = q_values.detach()
            self.max_action_id = np.argmax(q_values.cpu().detach().numpy())
            max_action = self.action_space[self.max_action_id]

            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action
    def get_max_action_id(self):
        return self.max_action_id
        
