import gym
from gym.envs.classic_control import AcrobotEnv

class CustomAcrobot(AcrobotEnv):
    def __init__(self):
        super(CustomAcrobot, self).__init__()

    def set_state(self, state):
        self.state = state