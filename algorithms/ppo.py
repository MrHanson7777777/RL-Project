import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.mlp import PolicyNetwork, ValueNetwork
from utils.experiment import reset_env, step_env


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.98,
        lamda=0.95,
        epochs=10,
        eps=0.2,
        lr=1e-3,
        device="cpu",
    ):
        self.gamma = gamma
        self.lamda = lamda
        self.epochs = epochs
        self.eps = eps
        self.device = torch.device(device)
        self.actor = PolicyNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.critic = ValueNetwork(state_dim, (hidden_dim,)).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = Categorical(self.actor(state))
        return dist.sample().item()

    def cal_advantage(self, td_delta):
        advantage = []
        adv = 0.0
        for delta in reversed(td_delta.detach().cpu().numpy()):
            adv = self.gamma * self.lamda * adv + delta
            advantage.append(adv)
        advantage.reverse()
        advantage = torch.tensor(advantage, dtype=torch.float32, device=self.device)
        return (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    def update(self, data):
        states = torch.tensor(np.array(data["states"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(data["next_states"]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(data["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            targets = rewards + self.gamma * self.critic(next_states) * (1.0 - dones)
            td_delta = targets - self.critic(states)
            advantage = self.cal_advantage(td_delta)
            old_probs = self.actor(states).gather(1, actions.view(-1, 1)).squeeze(1)
            old_log_probs = torch.log(old_probs + 1e-8)

        actor_loss_value, critic_loss_value = 0.0, 0.0
        for _ in range(self.epochs):
            probs = self.actor(states).gather(1, actions.view(-1, 1)).squeeze(1)
            log_probs = torch.log(probs + 1e-8)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(self.critic(states), targets)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            actor_loss_value = actor_loss.item()
            critic_loss_value = critic_loss.item()

        return actor_loss_value, critic_loss_value

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
