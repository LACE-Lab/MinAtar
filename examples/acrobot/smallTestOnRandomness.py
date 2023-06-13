import gymnasium as gym
from customAcrobot import CustomAcrobot
import random

env1 = gym.make("Acrobot-v1")
env1.reset(seed=42)
env1.state = [0,0,0,0]

env2 = gym.make("Acrobot-v1")
env2.reset(seed=24)
env2.state = [0,0,0,0]

random.seed(1)

for i in range(10000):
    choice = random.choice([0,1,2])
    env1.step(choice)
    env2.step(choice)
    if env1.state != env2.state:
        print(env1.state, env2.state)
        
env3 = CustomAcrobot()
env3.reset()
env3.set_state_from_observation([0,0,0,0,0,0])

env4 = CustomAcrobot()
env4.reset()
env4.set_state_from_observation([0,0,0,0,0,0])

for i in range(10000):
    choice = random.choice([0,1,2])
    env3.step(choice)
    env4.step(choice)
    if env3.state.all() != env4.state.all():
        print(env3.state, env4.state)