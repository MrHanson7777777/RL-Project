import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from networks.mlp import PolicyNetwork
from utils.experiment import reset_env, step_env


class ReinforceAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.98, lr=1e-3, device="cpu"):
        self.gamma = gamma
        self.device = torch.device(device)
        self.policy = PolicyNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = Categorical(self.policy(state))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs):
        returns = []
        ret = 0.0
        for reward in reversed(rewards):
            ret = reward + self.gamma * ret
            returns.append(ret)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for log_prob, ret in zip(log_probs, returns):
            loss = loss - log_prob * ret

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, env):
        state = reset_env(env)
        done = False
        rewards, log_probs = [], []
        while not done:
            action, log_prob = self.take_action(state)
            next_state, reward, done, _ = step_env(env, action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        loss = self.update(rewards, log_probs)
        return {"reward": float(np.sum(rewards)), "loss/policy": loss, "steps": len(rewards)}
