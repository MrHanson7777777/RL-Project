import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.mlp import PolicyNetwork, QNetwork
from utils.experiment import reset_env, step_env
from utils.replay_buffer import ReplayBuffer


class DDPGAgent:
    """Discrete-action DDPG-style baseline for class experiments."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.98,
        tau=0.005,
        lr=1e-3,
        buffer_size=10000,
        batch_size=64,
        minimal_size=200,
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.minimal_size = minimal_size
        self.device = torch.device(device)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.actor = PolicyNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.target_critic = QNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = Categorical(self.actor(state))
        return dist.sample().item()

    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, data):
        states = torch.tensor(data["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(data["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_q = self.target_critic(next_states).max(dim=1).values
            td_target = rewards + self.gamma * target_q * (1.0 - dones)

        q = self.critic(states).gather(1, actions.view(-1, 1)).squeeze(1)
        critic_loss = F.mse_loss(q, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        probs = self.actor(states)
        actor_loss = -(probs * self.critic(states)).sum(dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()
        return actor_loss.item(), critic_loss.item()

    def train(self, env):
        state = reset_env(env)
        done = False
        total_reward = 0.0
        steps = 0
        metrics = {"loss/actor": None, "loss/critic": None}
        while not done:
            action = self.take_action(state)
            next_state, reward, done, _ = step_env(env, action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if len(self.replay_buffer) >= self.minimal_size:
                batch = self.replay_buffer.sample(self.batch_size)
                actor_loss, critic_loss = self.update(batch)
                metrics = {"loss/actor": actor_loss, "loss/critic": critic_loss}

        return {"reward": total_reward, "steps": steps, "buffer_size": len(self.replay_buffer), **metrics}
