import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
import collections


batch_size = 128
minimal_size = 500
target_entropy = -1
class Actor(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        ret = F.relu(self.Linear1(states))
        return F.softmax(self.Linear2(ret), dim=1)


class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        ret = F.relu(self.Linear1(states))
        return self.Linear2(ret)

class ReplayBuffer:
    def __init__(self, capacity):
        # 用队列定义 ReplayBuffer


    def add(self, state, action, reward, next_state, done):
        # 将经验数据加入ReplayMemory中


    def sample(self, batch_size):
        # 将从ReplayBuffer中采样


    def size(self):
        # ReplayBuffer的大小

class SAC:

    def __init__(self, state_dim, hidden_dim, action_dim, target_entropy, tau, gamma):
        # 定义 actor、critic、目标网络等


    def calc_td_target(self, rewards, next_states, dones):
        # 计算 TD Target


    def update(self, data):
        # 更新 actor、critic、目标网络等网络的参数



    def take_action(self, state):
        # 采样动作（通过actor网络）


    def soft_update(self, net, target_net):
        # soft update 目标网络参数延迟更新

    def train(self, env, replay_buffer):
        state = env.reset()
        done = 0
        rewards = 0
        while not done:
            action = self.take_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            if replay_buffer.size() > minimal_size:
                data = replay_buffer.sample(batch_size)
                self.update(data)
            rewards += reward
        return rewards
