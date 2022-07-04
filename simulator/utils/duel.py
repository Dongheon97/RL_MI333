import random
import torch
import torch.nn as nn
import torch.autograd as autograd

class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQN, self).__init__()
        
        self.USE_CUDA = torch.backends.mps.is_available()
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.feature = nn.Sequential(
                nn.Linear(self.inputs, 128),
                nn.ReLU()
        )

        self.advantage = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.outputs)
        )

        self.value = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon):
        Variable = lambda *args, **kwargs: \
            autograd.Variable(*args, **kwargs).to('mps') if self.USE_CUDA \
            else autograd.Variable(*args, **kwargs)
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.outputs)
        return action
