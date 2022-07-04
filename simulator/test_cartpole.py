#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import random, math
import itertools as it
from utils.dqn import DQN
from utils.replayBuffer import ReplayBuffer

class MjCartPole():
    def __init__(self, xml_file, num_frames, batch_size, gamma):
        self.sim = MjSim(xml_file)
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(1000)
        self.USE_CUDA = torch.backends.mps.is_available()
        # DQN
        self.model = DQN(4, 2)
        self.optimizer = optim.Adam(self.model.parameters())

    def epsilon_by_frame(self, frame_idx):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        return epsilon_final + (epsilon_start - epsilon_final) \
                * math.exp(-1. * frame_idx / epsilon_decay)

    def Variable(self, *args, **kwargs):
        if self.USE_CUDA:
            return autograd.Variable(*args, **kwargs).to('mps')
        else:
            return autograd.Variable(*args, **kwargs)

    def reorder_state(self, qpos, qvel):
        # qpos: [cart_position, pole_angle]
        # qvel: [cart_velocity, pole_angular_velocity]
        # -> [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        reordered = list(it.chain(*zip(qpos, qvel)))
        return reordered

    def is_fall(self, curr_state):
        # -pi/15 <= pole_angle(rad) <= pi/15
        if(curr_state[2] >= -0.20944 and curr_state[2] <= 0.20944):
            return False
        else:
            return True

    def mj_step(self, givenAction):
        # set_action -> step -> next_state, reward, done 
        self.sim.data.ctrl[:] = givenAction
        self.sim.step()
        
        obv = self.sim.get_state()
        next_state = self.reorder_state(obv[1], obv[2])
        
        done = self.is_fall(next_state)
        reward = 1
        return next_state, reward, done

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = \
                self.replay_buffer.sample(batch_size)
        state = self.Variable(torch.FloatTensor(np.float32(state)))
        #next_state = self.Variable(torch.FloatTensor(np.float32(next_state)), \
                #volatile=True)
        with torch.no_grad():
            next_state = self.Variable(torch.FloatTensor(np.float32(next_state)))
        action = self.Variable(torch.LongTensor(action))
        reward = self.Variable(torch.FloatTensor(reward))
        done = self.Variable(torch.FloatTensor(done))

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1-done)

        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        if self.USE_CUDA:
            self.model = self.model.to('mps')

        losses = []
        all_rewards = []
        episode_reward = 0
        
        viewer = MjViewer(self.sim)
        init_state = self.sim.get_state()
        state = self.reorder_state(init_state[1], init_state[2])
        #print(f"state: {state}")
        '''
        for frame_idx in range(1, self.num_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)
            sampling = self.model.act(state, epsilon)
            if(sampling == 0):
                action = -1
            else:
                action = 1
            next_state, reward, done = self.mj_step(action)
            #print(f"next state: {next_state}")
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                state = self.sim.set_state(init_state)
                all_rewards.append(episode_reward)
                episode_reward = 0
            
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss(self.batch_size)
                losses.append(loss.item())

        print(all_rewards)
        '''
        while True:
            episode_reward = 0
            self.sim.set_state(init_state)
            for i in range(10000):
                '''
                if((i%2) == 0):
                    action = 0.5
                else:
                    action = -0.5
                '''
                action = np.array([[1]])
                next_state, reward, done = self.mj_step(action)
                episode_reward += reward
                #print(f'next_state: {next_state}')
                #print(f'Episode_reward: {episode_reward}, Frame_reward: {reward}')
                #print(f'done: {done}')
                print(next_state)
                viewer.render()
            if os.getenv('TESTING') is not None:
                break

if __name__=="__main__":
    PATH = '/Users/dongheon97/dev/Practice/mi333/simulator/xmls/cartpole.xml'
    num_frames = 500
    batch_size = 32
    gamma = 0.99
    xml_file = load_model_from_path(PATH)
    cartpole = MjCartPole(xml_file, num_frames, batch_size, gamma)
    cartpole.train()
