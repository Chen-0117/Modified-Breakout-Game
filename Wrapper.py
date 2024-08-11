import numpy as np
import gym

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env, uncertain_prob):
        super().__init__(env)
        self.uncertain_prob = uncertain_prob

    def action(self, action):
        """
        Change to a random action with probability `uncertain_prob`
        """
        if np.random.random() > self.uncertain_prob:
            return action
        else:
            return self.action_space.sample()


