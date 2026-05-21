import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from networks.mlp import PolicyNetwork, QNetwork
from utils.experiment import reset_env, step_env
from utils.replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.98,
        tau=0.005,
        lr=3e-4,
        buffer_size=10000,
        batch_size=128,
        minimal_size=500,
        target_entropy=None,
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.minimal_size = minimal_size
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.device = torch.device(device)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.actor = PolicyNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.target_critic1 = QNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.target_critic2 = QNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = Categorical(self.actor(state))
        return dist.sample().item()

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, data):
        states = torch.tensor(data["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(data["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q = torch.min(self.target_critic1(next_states), self.target_critic2(next_states))
            next_value = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)
            td_target = rewards + self.gamma * next_value * (1.0 - dones)

        q1 = self.critic1(states).gather(1, actions.view(-1, 1)).squeeze(1)
        q2 = self.critic2(states).gather(1, actions.view(-1, 1)).squeeze(1)
        critic1_loss = F.mse_loss(q1, td_target)
        critic2_loss = F.mse_loss(q2, td_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        q = torch.min(self.critic1(states), self.critic2(states))
        actor_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(dim=1).mean()
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy) * probs.detach()).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), self.alpha.item()

    def train(self, env):
        state = reset_env(env)
        done = False
        total_reward = 0.0
        steps = 0
        metrics = {"loss/actor": None, "loss/critic1": None, "loss/critic2": None, "alpha": self.alpha.item()}
        while not done:
            action = self.take_action(state)
            next_state, reward, done, _ = step_env(env, action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if len(self.replay_buffer) >= self.minimal_size:
                batch = self.replay_buffer.sample(self.batch_size)
                actor_loss, critic1_loss, critic2_loss, alpha = self.update(batch)
                metrics = {
                    "loss/actor": actor_loss,
                    "loss/critic1": critic1_loss,
                    "loss/critic2": critic2_loss,
                    "alpha": alpha,
                }

        return {"reward": total_reward, "steps": steps, "buffer_size": len(self.replay_buffer), **metrics}
