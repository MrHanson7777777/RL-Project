import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from networks.mlp import PolicyNetwork
from utils.experiment import reset_env, step_env


class ReinforceAgent:
    """REINFORCE (Monte-Carlo policy-gradient) with optional entropy regularisation.

    Entropy regularisation (controlled by ``entropy_coef``) adds a bonus
    -H(π) to the loss, encouraging the policy to stay stochastic and
    explore more broadly.  Set ``entropy_coef=0`` to recover vanilla REINFORCE.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.98,
        lr=1e-3,
        entropy_coef=0.01,
        normalize_returns=True,
        device="cpu",
    ):
        self.config = {
            "hidden_dim": hidden_dim,
            "gamma": gamma,
            "lr": lr,
            "entropy_coef": entropy_coef,
            "normalize_returns": normalize_returns,
            "device": device,
        }
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.normalize_returns = bool(normalize_returns)
        self.device = torch.device(device)
        self.policy = PolicyNetwork(state_dim, action_dim, (hidden_dim,)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = Categorical(self.policy(state))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs, states):
        # ---- 1. Compute normalised discounted returns ----
        returns = []
        ret = 0.0
        for reward in reversed(rewards):
            ret = reward + self.gamma * ret
            returns.append(ret)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if self.normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # ---- 2. Policy-gradient loss ----
        policy_loss = sum(-lp * ret for lp, ret in zip(log_probs, returns))

        # ---- 3. Entropy regularisation ----
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        entropy = Categorical(self.policy(states_t)).entropy().mean()

        loss = policy_loss - self.entropy_coef * entropy

        # ---- 4. Gradient step with clipping ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return loss.item(), entropy.item()

    def train(self, env):
        state = reset_env(env)
        done = False
        rewards, log_probs, states = [], [], []
        while not done:
            action, log_prob = self.take_action(state)
            states.append(state)
            next_state, reward, done, _ = step_env(env, action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        loss, entropy = self.update(rewards, log_probs, states)
        return {
            "reward": float(np.sum(rewards)),
            "loss/policy": loss,
            "entropy": entropy,
            "normalize_returns": self.normalize_returns,
            "steps": len(rewards),
        }
