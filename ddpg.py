import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
from collections import namedtuple

tau = 0.02
GAMMA = 0.9
buffer_size = 10000
batch_size = 32

#定义神经网络
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = torch.sigmoid(self.Linear3(x))
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear3.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.Linear1(x))
        x = self.Linear3(x)
        return x

#nametuple容器
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        # 用定义 ReplayBuffer


    def push(self, *args):
        # 将经验数据加入ReplayMemory中


    def sample(self, batch_size):
        # 将从ReplayBuffer中采样

    def __len__(self):
        # ReplayBuffer的大小


class DDPG(object):
    def __init__(self, input_size, action_shape, hidden_size, output_size):
        # 定义 actor、critic、目标网络、ReplayMemory等

    def put(self, s0, a0, r, s1, done):
        # 将经验数据加入ReplayMemory中

    def select_action(self, state):
        # 采样动作（通过actor网络），


    def update_parameters(self):
        # 更新 actor、critic、目标网络等网络的参数

        # critic更新

        # actor更新

        # soft update 目标网络参数延迟更新


    def train(self, env):
        s0 = env.reset()
        tot_reward = 0
        done = 0
        while not done:
            a0 = self.select_action(s0)
            s1, r, done, _ = env.step(round(a0.detach().numpy()[0]))
            tot_reward += r
            self.put(s0, a0, r, s1, 1 - done)  # 结束状态很重要，不然会很难学习。
            s0 = s1
            self.update_parameters()
        return tot_reward




