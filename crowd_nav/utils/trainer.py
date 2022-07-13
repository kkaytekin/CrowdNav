import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import copy


class Trainer(object):
    def __init__(self, model, memory, device, batch_size, dqn = False, gamma = 0.9, tau = 0.01):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.dqn = dqn
        self.gamma = gamma
        self.tau = tau
    
    def init_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        if not self.dqn:
            losses = 0
            for _ in range(num_batches):
                inputs, values = next(iter(self.data_loader))
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                losses += loss.data.item()
        else:
            losses = 0
            for _ in range(num_batches):
                data = next(iter(self.data_loader))

                states, actions, rewards, next_states, dones = next(iter(self.data_loader))
            
                ## Print the above value to debug
                logging.debug('Action : {}', actions)
                logging.debug('dones : {}', dones)
                
                # resize tensors
                actions = actions.view(actions.size(0), 1)
                dones = dones.view(dones.size(0), 1)

                # compute loss
                Q_t = self.model.forward(states).gather(1, actions) ## Compute the Q value of the selected action (N x 1)
                Q_t_next = self.target_model.forward(next_states) ## N x action_space
                max_next_Q = torch.max(Q_t_next, 1)[0]
                max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
                Q_t_star = rewards + (1 - dones) * self.gamma * max_next_Q
                


                self.optimizer.zero_grad()

                loss = self.criterion(Q_t.float(), Q_t_star.detach().float())
                loss.backward()
                self.optimizer.step()

                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


                losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss


