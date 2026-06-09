import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128,)):
        super().__init__()
        layers = []
        last_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return F.softmax(self.net(states), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=(128,)):
        super().__init__()
        layers = []
        last_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return self.net(states).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128,)):
        super().__init__()
        layers = []
        last_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return self.net(states)
