import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.mlp import PolicyNetwork, ValueNetwork
from utils.experiment import reset_env, step_env


class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.98, lr=1e-3, device="cpu"):
        self.gamma = gamma
        self.device = torch.device(device)
        self.actor = PolicyNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.critic = ValueNetwork(state_dim, (hidden_dim,)).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = Categorical(self.actor(state))
        return dist.sample().item()

    def update(self, data):
        states = torch.tensor(np.array(data["states"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(data["next_states"]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(data["dones"], dtype=torch.float32, device=self.device)

        values = self.critic(states)
        with torch.no_grad():
            targets = rewards + self.gamma * self.critic(next_states) * (1.0 - dones)
            td_delta = targets - values

        probs = self.actor(states)
        log_probs = torch.log(probs.gather(1, actions.view(-1, 1)).squeeze(1) + 1e-8)
        actor_loss = -(log_probs * td_delta.detach()).mean()
        critic_loss = F.mse_loss(values, targets)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train(self, env):
        data = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
        state = reset_env(env)
        done = False
        total_reward = 0.0
        while not done:
            action = self.take_action(state)
            next_state, reward, done, _ = step_env(env, action)
            data["states"].append(state)
            data["actions"].append(action)
            data["next_states"].append(next_state)
            data["rewards"].append(reward)
            data["dones"].append(done)
            state = next_state
            total_reward += reward

        actor_loss, critic_loss = self.update(data)
        return {
            "reward": total_reward,
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
            "steps": len(data["rewards"]),
        }
