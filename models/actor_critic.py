import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, std=0.0):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 256)

        # actor's layer
        self.action_head = nn.Linear(256, 1)

        # critic's layer
        self.value_head = nn.Linear(256, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

        self.log_std = nn.Parameter(torch.ones(1)*std)

    def forward(self, x):
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        mu = self.action_head(x)
        std = self.log_std.exp().expand_as(mu)
        #dist = Normal(mu, std)
        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values fro both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return mu, std, state_values

