#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjSimState, MjViewer
import os
import wandb
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

    def order_state(self, qpos, qvel):
        # qpos: [cart_position, pole_angle]
        # qvel: [cart_velocity, pole_angular_velocity]
        # -> [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        ordered = list(it.chain(*zip(qpos, qvel)))
        return ordered

    def reorder_state(self, ordered):
        qpos = [ordered[0], ordered[2]]
        qvel = [ordered[1], ordered[3]]
        return np.array(qpos), np.array(qvel)

    def is_done(self, curr_state):
        # -pi/15 <= pole_angle(rad):curr_state[2] <= pi/15
        # -1.0 <= cart_position(m):curr_state[0] <= 1.0
        if( (curr_state[0] >= -1.0 and curr_state[0] <= 1.0) \
                and (curr_state[2] >= -0.20944 and curr_state[2] <= 0.20944)):
            return False
        else:
            print(f'done position: {curr_state[0]}, done angle: {curr_state[2]}')
            return True

    def mj_step(self, givenAction):
        # set_action -> step -> next_state, reward, done 
        self.sim.data.ctrl[:] = givenAction
        self.sim.step()

        obv = self.sim.get_state()
        next_state = self.order_state(obv[1], obv[2])
        #next_state = self.to_tensor(next_state)
        done = self.is_done(next_state)
        reward = 1
        return np.array(next_state), reward, done

    def mj_reset(self):
        # -0.048 <= cart_position, cart_velocity, pole_angle, pole_angular_velocity <= +0.048
        cp = random.uniform(-0.048, 0.048)
        cv = random.uniform(-0.048, 0.048)
        pa = random.uniform(-0.048, 0.048)
        pv = random.uniform(-0.048, 0.048)
        reset_state = [cp, cv, pa, pv]
        qpos, qvel = self.reorder_state(reset_state)
        old_state = self.sim.get_state()
        new_state = MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        #reset_state = self.to_tensor(reset_state)
        return np.array(reset_state)

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = \
                self.replay_buffer.sample(batch_size)
        state = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)), \
                volatile=True)
        '''
        with torch.no_grad():
            next_state = self.Variable(torch.FloatTensor(np.float32(next_state)))
        '''
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
        '''
        wandb.init(project="dqn", entity="dongheon97")
        wandb.config = {
                "learning_rate": self.gamma,
                "num_frames": self.num_frames,
                "batch_size": self.batch_size
                }
        wandb.watch(self.model)
        '''
        if self.USE_CUDA:
            self.model = self.model.to('mps')

        losses = []
        all_rewards = []
        episode_reward = 0
        
        viewer = MjViewer(self.sim)
        init_state = self.sim.get_state()
        state = self.order_state(init_state[1], init_state[2])
        #print(f"state: {state}")
        for frame_idx in range(1, self.num_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)
            sampling = self.model.act(state, epsilon)
            if(sampling == 0):
                action = -1
            else:
                action = 1
            #wandb.log({"action": action})
            print(f'action: {action}')
            next_state, reward, done = self.mj_step(action)

            #print(f"next state: {next_state}")
            self.replay_buffer.push(state, action, reward, next_state, done)
            #wandb.log({"size of buffer": len(self.replay_buffer)})
            state = next_state
            episode_reward += reward

            if done:
                state = self.mj_reset()
                all_rewards.append(episode_reward)
                #wandb.log({"episode_reward": episode_reward})
                print(f'episode_reward: {episode_reward}')
                episode_reward = 0
            
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss(self.batch_size)
                losses.append(loss.item())
                #wandb.log({"loss": loss.item()})
            
            viewer.render()

        print(f'all_rewards: {all_rewards}')
        print(f'losses: {losses[-1]}')
    
if __name__=="__main__":
    PATH = '/Users/dongheon97/dev/Practice/mi333/simulator/xmls/cartpole.xml'
    num_frames = 1000
    batch_size = 32
    gamma = 0.99
    xml_file = load_model_from_path(PATH)
    cartpole = MjCartPole(xml_file, num_frames, batch_size, gamma)
    cartpole.train()
