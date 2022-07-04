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
from utils.ac import ActorCritic

class MjCartPole():
    def __init__(self, xml_file):
        self.sim = MjSim(xml_file)
        self.hidden_size = 256
        self.USE_CUDA = torch.backends.mps.is_available()
        self.device = torch.device('mps' if self.USE_CUDA else 'cpu')
        # Actor-Critic
        self.model = ActorCritic(4, 1, self.hidden_size).to(self.device)

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

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            # Q(s, a) = R + gamma * value(s')
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
            return returns

    def test_env(self, vis=False):
        viewer = MjViewer(self.sim)
        state = self.mj_reset()
        if vis: viewer.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            next_state, reward, done = self.mj_step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis: viewer.render()
            total_reward += reward
        return total_reward

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

        optimizer = optim.Adam(self.model.parameters())
        
        lr = 3e-4
        num_steps = 5

        max_frames = 20000
        frame_idx = 0
        test_rewards = []

        viewer = MjViewer(self.sim)
        state = self.mj_reset()
        while frame_idx < max_frames:
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(num_steps):
                state = torch.FloatTensor(state).to(self.device)
                dist, value = self.model(state)

                # Actor
                action = dist.sample()
                next_state, reward, done = self.mj_step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)

                # Critic
                values.append(log_prob)

                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(self.device))

                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    test_rewards.append(np.mean([self.test_env() for _ in range(10)]))

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            # Reward + gamma * value(s')
            returns = self.compute_returns(next_values, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            # Advantage = Q(s,a) - V(s) = R + gamma * V(s') - V(s)
            advantage = returns - values

            # Actor Loss Function = cross_entropy * advantage
            actor_loss = -(log_probs * advantage.detach()).mean()

            # Critic Loss Funtion = advantage**2
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizier.step()


if __name__=="__main__":
    PATH = '/Users/dongheon97/dev/Practice/mi333/simulator/xmls/cartpole.xml'
    xml_file = load_model_from_path(PATH)
    cartpole = MjCartPole(xml_file)
    cartpole.train()
