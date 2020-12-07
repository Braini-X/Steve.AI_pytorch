import sys
import pylab
import random
import numpy as np
import configparser
from collections import deque
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

config = configparser.ConfigParser()
config.read('config.ini')
epsilon = float(config.get('DEFAULT', 'EPSILON'))
epsilon_min = float(config.get('DEFAULT', 'EPSILON_MIN'))
epsilon_decay = float(config.get('DEFAULT', 'EPSILON_DECAY'))
batch_size = int(config.get('DEFAULT', 'BATCH_SIZE'))
gamma = float(config.get('DEFAULT', 'GAMMA'))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f1=nn.Linear(7,24)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2=nn.Linear(24,32)
        self.f2.weight.data.normal_(0, 0.1)
        self.f3=nn.Linear(32,24)
        self.f3.weight.data.normal_(0, 0.1)
        self.f4=nn.Linear(24,7)
        self.f4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x=self.f1(x)
        x=F.relu(x)
        x = x.view(x.size(0),-1)
        x=self.f2(x)
        x=F.relu(x)
        x = x.view(x.size(0),-1)
        x=self.f3(x)
        x=F.relu(x)
        action=self.f4(x)
        return action

class DQNAgent(object):
    """docstring for DQNAgent"""
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = 0.1
        self.memory = deque(maxlen=2000)
        self.model = Net()
        self.target_model = Net()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = self.lr)
        self.loss_func = torch.nn.SmoothL1Loss()

        
    def load(self, saved_model):
        self.model.load_state_dict(torch.load(saved_model))

    def save(self, nn_save):
        torch.save(self.model.state_dict(), nn_save)

    def act(self, state):
        if np.random.uniform() < self.epsilon:
            act = random.randrange(self.action_size)
#             print(act)
            return act
        else:
            actions_value = self.model.forward(state)
#             print(torch.max(actions_value, 1)[1])
            action = torch.max(actions_value, 1)[1].data.numpy()
#             print(action)
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, memory):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.Tensor(state)
#             action = torch.Tensor(action)
#             reward = torch.Tensor(reward)
            next_state = torch.Tensor(next_state)
            q_eval = self.model(state)[0][action].unsqueeze(-1)#.gather(1, action)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
#             print(q_eval)
            q_next = self.target_model(next_state).detach()     # q_next 不进行反向传递误差, 所以 detach Q现实
#             print(q_eval.detach())
            q_target = reward + self.gamma * q_next.max(1)[0]
#             print(q_eval.shape, q_target.shape)
            loss = self.loss_func(q_eval, q_target)
#             print(loss.detach().numpy())
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

