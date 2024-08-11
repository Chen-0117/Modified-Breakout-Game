
class Hyperparameters():
    def __init__(self):
        self.RL_load_path = f'./ep_change/final_weights.pth'
        self.save_path = f'./ep_change/final_weights'
        self.learning_rate = 0.005
        self.discount_factor = 0.99
        self.batch_size = 32
        self.targetDQN_update_rate = 10
        self.num_episodes = 500
        self.num_test_episodes = 10
        self.epsilon_decay = 0.99
        self.buffer_size = 80000
        self.uncertain_prob = 0.2

        self.epsilon_max = 1
        self.epsilon_min = 0.1
        self.epsilon = 1



    def change(self, batch_size = 32, learning_rate = 5e-4, num_episodes = 3000, epsilon_decay = 0.999):
        '''
        This method can change
        map_size,
        Also can change the following argument if called:
        batch_size , learning_rate , num_episodes
        '''
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.epsilon_decay = epsilon_decay
