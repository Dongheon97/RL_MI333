import numpy as np
from collections import deque
import math, random

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        #:print(f'input state: {state}, input done: {done}')
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = \
                zip(*random.sample(self.buffer, batch_size))
        #print(f'state: {state}, action: {action}, reward: {reward}, next_state: \
        #        {next_state}, done: {done}')
        return np.concatenate(state), action, reward, \
                np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

