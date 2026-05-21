import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=1.0)
        nn.init.constant_(layer.bias, 0.0)


class MLP(nn.Module):
    """Shared MLP backbone used by all algorithms."""

    def __init__(self, input_dim, hidden_dims=(128,), output_dim=None):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], output_dim) if output_dim is not None else nn.Identity()
        self.apply(init_layer)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.head(self.body(x.float()))


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128,)):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dims, action_dim)

    def logits(self, states):
        return self.mlp(states)

    def forward(self, states):
        return F.softmax(self.logits(states), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=(128,)):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dims, 1)

    def forward(self, states):
        return self.mlp(states).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128,)):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dims, action_dim)

    def forward(self, states):
        return self.mlp(states)
