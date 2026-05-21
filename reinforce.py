import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.softmax(self.Linear3(x), dim=1)
        return x

class ReinforceAgent(object):
    def __init__(self, input_size, hidden_size, output_size, gamma):
        # 定义策略网络


    def update_parameters(self, rewards, log_probs):
        # 更新策略网络参数


    def select_action(self, s):
        # 采样动作


    def train(self, env):
        state = env.reset()
        done = 0
        log_probs = []
        rewards = []
        while not done:
            action, prob = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(prob)
            rewards.append(reward)
            state = next_state
        self.update_parameters(rewards, log_probs)
        return np.sum(rewards)
