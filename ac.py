import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        out = F.relu(self.Linear1(states))
        out = F.softmax(self.Linear2(out), dim=1)
        return out
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, states):
        out = F.relu(self.Linear1(states))
        out = self.Linear2(out)
        return out

class ActorCriticAgent:

    def __init__(self, state_dim, hidden_dim, action_dim, gamma):
        # 定义 actor、critic网络


    def update(self, data):
        # 更新 actor、critic 网络参数


    def take_action(self, state):
        # 采样动作（通过actor网络）


    def train(self, env):
        data = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'done': [],
        }
        done = 0
        state = env.reset()
        rewards = 0
        while not done:
            action = self.take_action(state)
            next_state, reward, done, _ = env.step(action)
            data['states'].append(state)
            data['actions'].append(action)
            data['next_states'].append(next_state)
            data['rewards'].append(reward)
            data['done'].append(done)
            state = next_state
            rewards += reward
        self.update(data)
        return rewards











