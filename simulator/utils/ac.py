import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
        )

        self.log_std = nn.Parameter(torch.ones(num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        #print(f'nn.Parameter: {nn.Parameter()}\nmu shape: {mu.shape}')
        #print(f'self.log_std: {self.log_std}, {type(self.log_std)}')
        std = self.log_std.exp().expand_as(mu)
        #print(f'std: {std}')
        #dist = Normal(mu, std.squeeze(0))
        dist = Normal(mu, std)
        return dist, value

