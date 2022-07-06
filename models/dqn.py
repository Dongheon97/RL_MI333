import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.input = num_inputs
        self.output = num_actions
        # mps: M1 of Apple Silicon Chip
        self.USE_CUDA = torch.backends.mps.is_available()
        self.layers = nn.Sequential(
                nn.Linear(num_inputs, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions)
        )

    def Variable(self, *args, **kwargs):
        if self.USE_CUDA:
            return autograd.Variable(*args, **kwargs).to('mps')
        else:
            return autograd.Variable(*args, **kwargs)

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = self.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            '''
            with torch.no_grad():
                state = self.Variable(torch.FloatTensor(state).unsqueeze(0))
            '''
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
            #action = int(q_value.max(1)[1].data[0].cpu().int().numpy())
        else:
            action = random.randrange(self.output)
        return action

