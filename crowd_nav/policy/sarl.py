from xml import dom
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.gnn import GNN

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
        print(state.shape)
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

class ActorNetwork(nn.Module):
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
        # self.tanh = nn.Tanh()
        # print(mlp3_dims[-1])
        self.softmax = nn.Softmax()

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
        output = self.mlp3(joint_state)
        # print(output.shape)
        # print(output)
        output = self.softmax(output)
        # print(output)
        # output = self.tanh(output)
        # print(output.shape)
        # print(output)
        return output

class ActorCriticNetwork(nn.Module):
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
        # self.tanh = nn.Tanh()
        # print(mlp3_dims[-1])
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        # print(self_state.shape)
        human_states = state[:, :, self.self_state_dim:]
        # print(human_states.shape)
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
        # print(torch.mul(weights, features).shape)
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)

        # print(joint_state.shape)
        output = self.mlp3(joint_state)
        # print(output.shape)
        # print(output)
        value = output[:, 0]
        value = value.reshape((size[0], 1))
        # print(value)
        # print(value.shape)
        action_probs = output[:, 1:]
        action_probs = self.softmax(action_probs)
        # print(action_probs)
        # print(action_probs.shape)
        output = torch.cat([value, action_probs], dim = 1)
        # self.softmax(output[:, 1:])
        # print(output)
        # output = self.softmax(output)
        # print(output)
        # output = self.tanh(output)
        # print(output.shape)
        # print(output)
        return output

class ActorCriticGnnNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        ## GNN layers
        self.node_a_embedding_mlp = mlp(input_dim - self.self_state_dim, [32, 16])
        self.node_b_embedding_mlp = mlp(self.self_state_dim, [32, 16])
        self.edge_embedding_mlp = mlp(input_dim, [32, 16])
        ## 
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim + 16
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None
        self.softmax = nn.Softmax(dim = 1)
        self.gnn = GNN(16, 16, 6, 1)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim] ## shape (batch_size, self_state_dim)
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
        # weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        weighted_feature = torch.mul(weights, features)

        human_states = state[:, :, self.self_state_dim:].reshape((size[0] * size[1], -1))
        
        human_node_embedding = self.node_a_embedding_mlp(human_states)
        human_node_embedding = human_node_embedding.reshape((size[0], size[1], -1))
        robot_node_embedding = self.node_b_embedding_mlp(self_state)
        robot_node_embedding = robot_node_embedding.unsqueeze(dim = 1)

        edge_embedding = self.edge_embedding_mlp(state.reshape((size[0] * size[1], -1)))
        edge_embedding = edge_embedding.reshape((size[0], size[1], -1))
        edge_embedding = edge_embedding.unsqueeze(dim = 2)
        # print(edge_embedding.shape)
        edge_embeds_latent, _, _ = self.gnn(edge_embedding, human_node_embedding, robot_node_embedding)
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state.unsqueeze(dim = 1).expand((size[0], size[1], self.self_state_dim)), 
        weighted_feature, 
        edge_embeds_latent.reshape((size[0], size[1], -1))
        ], dim = -1)
        
        joint_state = torch.sum(joint_state, dim = 1)

        output = self.mlp3(joint_state)
        value = output[:, 0]
        value = value.reshape((size[0], 1))
        action_probs = output[:, 1:]
        action_probs = self.softmax(action_probs)
        output = torch.cat([value, action_probs], dim = 1)
        return output

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
        value = self.mlp3(joint_state)
        return value

class SARL_DQN(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL_DQN'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl_dqn', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl_dqn', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl_dqn', 'mlp3_dims').split(', ')]
        mlp3_actions_dims = mlp3_dims.copy()
        if self.has_zero_speed:
            mlp3_actions_dims[-1] = (self.speed_samples * self.rotation_samples + 1 ) + 1
        else: 
            mlp3_actions_dims[-1] = self.speed_samples * self.rotation_samples + 1
        
        attention_dims = [int(x) for x in config.get('sarl_dqn', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl_dqn', 'with_om')
        with_global_state = config.getboolean('sarl_dqn', 'with_global_state')
        self.model = DQN(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_actions_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl_dqn', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
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
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            action_values = self.model(self.transform(state))
            max_action_id = torch.argmax(action_values)
            max_action = self.action_space[max_action_id]

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

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

class SARL_PPO(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL_PPO'

    def configure(self, config):
        self.set_common_parameters(config)
        self.build_action_space(v_pref = 1)
        mlp1_dims = [int(x) for x in config.get('sarl_ppo', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl_ppo', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl_ppo', 'mlp3_dims').split(', ')]
        mlp3_actor_dims = mlp3_dims.copy()
        if self.has_zero_speed:
            mlp3_actor_dims[-1] = (self.speed_samples * self.rotation_samples + 1 ) + 1
        else: 
            mlp3_actor_dims[-1] = self.speed_samples * self.rotation_samples + 1
        attention_dims = [int(x) for x in config.get('sarl_ppo', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl_ppo', 'with_om')
        with_global_state = config.getboolean('sarl_ppo', 'with_global_state')
        # self.critic = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
        #                           attention_dims, with_global_state, self.cell_size, self.cell_num)
        # self.actor = ActorNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_actor_dims,
        #                           attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.actor_critic = ActorCriticNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_actor_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl_ppo', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.actor_critic.attention_weights

    def predict(self, state):
        batch_states = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                              for human_state in state.human_states], dim=0)
        rotated_batch_input = self.rotate(batch_states).unsqueeze(0)

        logits = self.actor(rotated_batch_input)

        action_ids = torch.argmax(logits, dim = 1)



        return self.actor(rotated_batch_input)


class GNN_SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'GNN_SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        self.build_action_space(v_pref = 1)
        mlp1_dims = [int(x) for x in config.get('gnn_sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('gnn_sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('gnn_sarl', 'mlp3_dims').split(', ')]
        mlp3_actor_dims = mlp3_dims.copy()
        if self.has_zero_speed:
            mlp3_actor_dims[-1] = (self.speed_samples * self.rotation_samples + 1 ) + 1
        else: 
            mlp3_actor_dims[-1] = self.speed_samples * self.rotation_samples + 1
        attention_dims = [int(x) for x in config.get('gnn_sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('gnn_sarl', 'with_om')
        with_global_state = config.getboolean('gnn_sarl', 'with_global_state')
        
        self.actor_critic = ActorCriticGnnNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_actor_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('gnn_sarl', 'multiagent_training')
        # if self.with_om:
        #     self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.actor_critic.attention_weights


