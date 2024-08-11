class Hyperparameters:
    def __init__(self):
        # Environment specific parameters
        self.env_name = 'Breakout-v4'
        self.max_episode_steps = 10000

        self.discount = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.n_episodes = 500

        self.n_states = 210 * 160 * 3
        self.rand = 0.2

    def change(self, learning_rate=None, num_episodes=None, epsilon_decay=None):
        '''
        This method allows changing specific hyperparameters if called.
        '''
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if num_episodes is not None:
            self.n_episodes = num_episodes
        if epsilon_decay is not None:
            self.epsilon_decay = epsilon_decay
