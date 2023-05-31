import numpy as np
import gym
from gym.envs.classic_control import AcrobotEnv

class CustomAcrobot(AcrobotEnv):
    def __init__(self):
        super(CustomAcrobot, self).__init__()

    def set_state_from_observation(self, obs):
        assert len(obs) == 6, "Observation should be of length 6: (cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1dot, theta2dot)"
        obs = obs.detach().numpy()
        theta1 = np.arctan2(obs[1], obs[0])
        theta2 = np.arctan2(obs[3], obs[2])
        theta1dot = obs[4]
        theta2dot = obs[5]
        self.state = np.array([theta1, theta2, theta1dot, theta2dot], dtype=np.float32)

    def step(self, action):
        state, reward, terminated, truncated, _ = super().step(action)
        return state, reward, terminated, truncated, {}

    def reset(self):
        super().reset()
        return self.get_obs()
        
    def get_obs(self):
        s = self.state
        if isinstance(s, dict): # Added for debugging. Remove this check if the error does not occur.
            print(f"Unexpected state type: {type(s)}, value: {s}")
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])