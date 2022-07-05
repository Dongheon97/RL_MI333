#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjSimState, MjViewer
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import itertools as it
from collections import namedtuple

from critic import Policy

class MjCartPole():
    def __init__(self, xml_file):
        self.sim = MjSim(xml_file)
        #self.viewer = MjViewer(self.sim)
        self.render = False
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.USE_CUDA else 'cpu')
        self.eps = np.finfo(np.float32).eps.item()
        self.threshold = 475
        # Actor-Critic
        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-2)

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
                or (curr_state[2] >= -0.20944 and curr_state[2] <= 0.20944)):
        #if( (curr_state[0][0] >= -1.0 and curr_state[0][0] <= 1.0) \
        #        or (curr_state[0][2] >= -0.20944 and curr_state[0][2] <= 0.20944)):
            return False
        else:
            #print(f'done position: {curr_state[0]}, done angle: {curr_state[2]}')
            return True

    def norm_action(self, action):
        low = -1.0
        high = 1.0
        action = low + (action+1.0) * 0.5 * (high-low)
        action = np.clip(action, low, high)
        return action

    def mj_step(self, givenAction):
        # set_action -> step -> next_state, reward, done 
        action = self.norm_action(givenAction)
        self.sim.data.ctrl[:] = action
        self.sim.step()

        obv = self.sim.get_state()
        next_state = self.order_state(obv[1], obv[2])
        next_state = np.array(next_state)
        
        #next_state = np.expand_dims(next_state, axis=0)
        
        done = self.is_done(next_state)
        reward = 1
        return next_state, reward, done

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
        reset_state = np.array(reset_state)
        
        #reset_state = np.expand_dims(reset_state, axis=0)
        
        return reset_state

    def select_action(self, state):
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        state = torch.from_numpy(state).float()
        
        probs, std, state_value = self.model(state)
        
        m = Normal(probs, std)
        #action = m.sample()
        
        action = torch.tanh(m.sample())
        #print(f'm: {m}, action: {action}')
        #print(f'action: {action}')
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def finish_episode(self, gamma=0.99):
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.model.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            
        self.optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

        # reset reward and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def main(self):
        running_reward = 10

        # run infinitely many episodes
        for i_episode in it.count(1):
            # reset environment and episode reward
            state = self.mj_reset()
            episode_reward = 0

            for t in range(1, 10000):
                # select action from policy
                action = self.select_action(state)
                #print(f'action: {action}')
                # take the action
                next_state, reward, done = self.mj_step(action)

                '''
                if self.render:
                    self.viewer.render()
                '''

                self.model.rewards.append(reward)
                episode_reward += reward
                if done:
                    break
                state = next_state

            # update cumulative reward
            running_reward = 0.05 * episode_reward + (1-0.05) * running_reward

            # perform backpropagation
            self.finish_episode()

            # log results
            if i_episode % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, episode_reward, running_reward))

            # check if we have 'solved' the cartpole problem
            if running_reward > self.threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} times steps!".format(running_reward, t))
                break

        torch.save(self.model.state_dict(), './model.pt')

if __name__=="__main__":
    PATH = './simulator/xmls/cartpole.xml'
    xml_file = load_model_from_path(PATH)
    cartpole = MjCartPole(xml_file)
    cartpole.main()
