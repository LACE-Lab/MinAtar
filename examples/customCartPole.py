import gym
from gym.envs.classic_control import CartPoleEnv

class CustomCartPole(CartPoleEnv):
    def __init__(self):
        super(CustomCartPole, self).__init__()

    def set_state(self, state):
        self.state = state