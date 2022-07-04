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
from utils.actor import ActorCritic

class MjCartPole():
    def __init__(self, xml_file):
        self.sim = MjSim(xml_file)
        self.hidden_size = 256
        self.lr = 3e-4
        self.USE_CUDA = torch.backends.mps.is_available()
        self.device = torch.device('mps' if self.USE_CUDA else 'cpu')
        # Actor-Critic
        self.model = ActorCritic(4, 1, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def order_state(self, qpos, qvel):
        # qpos: [cart_position, pole_angle]
        # qvel: [cart_velocity, pole_angular_velocity]
        # -> [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        ordered = list(it.chain(*zip(qpos, qvel)))
        return ordered

    def reorder_state(self, ordered):
        qpos = [ordered[0][0], ordered[0][2]]
        qvel = [ordered[0][1], ordered[0][3]]
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
        next_state = [next_state]
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
        reset_state = [[cp, cv, pa, pv]]
        qpos, qvel = self.reorder_state(reset_state)
        old_state = self.sim.get_state()
        new_state = MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        #reset_state = self.to_tensor(reset_state)
        return np.array(reset_state)

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], \
                    log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage \
                    in ppo_iter(mini_vatch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test_env(self, vis=False):
        viewer = self.sim.MjViewer
        state = self.mj_reset()
        if vis: viewer.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to('mps')
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


        num_steps = 20
        mini_batch_size = 5
        ppo_epochs = 4

        max_frames = 15000
        frame_idx = 0
        test_rewards = []

        viewer = MjViewer(self.sim)
        state = self.mj_reset()
        #print(f"state: {state}")
        while frame_idx < max_frames:
            
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(num_steps):
                state = torch.FloatTensor(state).to(self.device)
                dist, value = self.model(state)

                action = dist.sample()
                next_state, reward, done = self.mj_step(action.detach().cpu().numpy())
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(self.device))

                states.append(state)
                actions.append(action)
                
                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test_env() for _ in range(10)])
                    test_rewards.append(test_reward)
            
            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

if __name__=="__main__":
    PATH = '/Users/dongheon97/dev/Practice/mi333/simulator/xmls/cartpole.xml'
    num_frames = 1000
    hidden_size = 256
    mini_batch_size = 5
    lr = 3e-4
    xml_file = load_model_from_path(PATH)
    cartpole = MjCartPole(xml_file)
    cartpole.train()
