import gym
import mujoco_py
import mujoco
from mujoco import *

'''
#env = gym.make('Humanoid-v4')
env = gym.make('HalfCheetah-v4')
#env = gym.make('Ant-v2')
#env = gym.make('Pendulum-v1')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()
'''

ASSETS = dict()

model = mujoco.MjModel.from_xml_path('./xmls/cartpole.xml', ASSETS)
data = mujoco.MjData(model)

while data.time < 0.01:
    act = mujoco.mj_step(model, data)
    print(act)
    print(data.geom_xpos)
