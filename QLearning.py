import gymnasium as gym
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import pickle
import torch

class Agent:
    def __init__(self, n_states, n_actions, discount, lr, epsilon, epsilon_decay, min_epsilon, env: gym.wrappers.time_limit.TimeLimit, device=False):
        self.gamma = discount
        self.alpha = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.env = env

        if not device:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device


        self.Q = {} # Dictionary can handle dynamic state space indexing while numpy cant

    def preprocess_state(self, state):
        # Example preprocessing: downscale the image and convert to grayscale
        state = np.mean(state[35:195:2, ::2], axis=2).astype(np.uint8)
        return tuple(state.flatten())

    def epsGreedy(self, Q, s):
        p = np.random.uniform(0, 1)
        if p <= self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            ActionValues = Q[s]
            max_indices = np.where(ActionValues == np.max(ActionValues))[0]
            action = np.random.choice(max_indices)
        return action
    
    # def epsGreedy(self, Q, s):
    #     p = np.random.uniform(0, 1)
    #     if p <= self.epsilon:
    #         action = np.random.randint(0, self.n_actions)
    #     else:
    #         ActionValues = Q[s, :]
    #         max_indices = np.where(ActionValues == np.max(ActionValues))[0]
    #         action = np.random.choice(max_indices)
    #     return action

    def QLearning(self, n_episodes):
        K = trange(n_episodes)
        reward_array = np.zeros(n_episodes)
        Q_avg = 0

        for k in K:
            state, _ = self.env.reset()
            s = self.preprocess_state(state)
            if s not in self.Q:
                self.Q[s] = np.zeros(self.n_actions)
            terminated = False
            total_reward = 0

            while not terminated:
                a = self.epsGreedy(self.Q, s)

                next_state, reward, terminated, _, _ = self.env.step(a)
                s_next = self.preprocess_state(next_state)
                if s_next not in self.Q:
                    self.Q[s_next] = np.zeros(self.n_actions)

                best_next_action = np.argmax(self.Q[s_next])
                G = reward + self.gamma * self.Q[s_next][best_next_action] - self.Q[s][a]
                self.Q[s][a] += self.alpha * G
                total_reward += reward

                if terminated:
                    K.set_description(f'Episode {k+1} ended with reward {total_reward}')
                    K.refresh()
                    Q_avg = Q_avg + (total_reward - Q_avg) / (k + 1)
                    reward_array[k] = Q_avg
                    break

                s = s_next

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

        self.env.close()

        plt.figure('Learning Curve')
        plt.plot([k + 1 for k in range(n_episodes)], reward_array, color='black', linewidth=0.5)
        plt.ylabel('Average Reward', fontsize=12)
        plt.xlabel('Episode', fontsize=12)
        plt.title(f'Learning by Q-Learning for {n_episodes} Episodes', fontsize=12)
        plt.show()

        return self.Q, reward_array
    # def QLearning(self, n_episodes):
    #     K = trange(n_episodes)
    #     reward_array = np.zeros(n_episodes)
    #     Q = self.Q
    #     Q_avg = 0

    #     for k in K:
    #         s, _ = self.env.reset()
    #         terminated = False
    #         total_reward = 0

    #         while not terminated:
    #             a = self.epsGreedy(Q, s)
    #             s_next, reward, terminated, _, _ = self.env.step(a)
    #             best_next_action = np.argmax(Q[s_next])
    #             G = reward + self.gamma * Q[s_next, best_next_action] - Q[s, a]
    #             Q[s, a] = Q[s, a] + self.alpha * G
    #             total_reward += reward

    #             if terminated:
    #                 K.set_description(f'Episode {k+1} ended')
    #                 K.refresh()
    #                 Q_avg = Q_avg + (reward - Q_avg) / (k+1)
    #                 reward_array[k] = Q_avg
    #                 break

    #             s = s_next

    #         if self.epsilon > self.min_epsilon:
    #             self.epsilon *= self.epsilon_decay

    #     self.env.close()
    #     self.Q = Q

    #     plt.figure('Learning Curve')
    #     plt.plot([k + 1 for k in range(n_episodes)], reward_array, color='black', linewidth=0.5)
    #     plt.ylabel('Average Reward', fontsize=12)
    #     plt.xlabel('Episode', fontsize=12)
    #     plt.title(f'Learning by Q-Learning for {n_episodes} Episodes', fontsize=12)
    #     plt.show()

    #     return Q, reward_array

    # def eval(self, n_episodes, Q=None):
    #     if Q is not None:
    #         self.Q = Q

    #     K = trange(n_episodes)
    #     success_rate = 0

    #     for k in K:
    #         s, _ = self.env.reset()
    #         ActionValues = self.Q[s]
    #         max_indices = np.where(ActionValues == np.max(ActionValues))[0]
    #         a = np.random.choice(max_indices)
    #         terminated = False

    #         while not terminated:
    #             s_next, reward, terminated, _, _ = self.env.step(a)
    #             ActionValues_next = self.Q[s_next]
    #             max_indices_next = np.where(ActionValues_next == np.max(ActionValues_next))[0]
    #             a_next = np.random.choice(max_indices_next)

    #             if terminated:
    #                 K.set_description(f'Episode {k+1} ended with Reward {reward}')
    #                 K.refresh()
    #                 success_rate += reward
    #                 break

    #             s, a = s_next, a_next

    #     self.env.close()
    #     return success_rate

    def eval(self, n_episodes, Q=None):
        if Q is not None:
            self.Q = Q

        K = trange(n_episodes)
        success_rate = 0

        for k in K:
            state, _ = self.env.reset()
            s = self.preprocess_state(state)
            if s not in self.Q:
                self.Q[s] = np.zeros(self.n_actions)

            ActionValues = self.Q[s]
            max_indices = np.where(ActionValues == np.max(ActionValues))[0]
            a = np.random.choice(max_indices)
            terminated = False

            while not terminated:
                next_state, reward, terminated, _, _ = self.env.step(a)
                s_next = self.preprocess_state(next_state)
                if s_next not in self.Q:
                    self.Q[s_next] = np.zeros(self.n_actions)

                ActionValues_next = self.Q[s_next]
                max_indices_next = np.where(ActionValues_next == np.max(ActionValues_next))[0]
                a_next = np.random.choice(max_indices_next)

                if terminated:
                    K.set_description(f'Episode {k+1} ended with Reward {reward}')
                    K.refresh()
                    success_rate += reward
                    break

                s, a = s_next, a_next

        self.env.close()
        return success_rate

def saveQ(agent: Agent):
    with open('Qfunction.pkl', 'wb') as outp:
        pickle.dump(agent.Q, outp, pickle.HIGHEST_PROTOCOL)

def loadAgent(n_states, n_actions, discount, lr, epsilon, epsilon_decay, min_epsilon, env):
    agent = Agent(n_states, n_actions, discount, lr, epsilon, epsilon_decay, min_epsilon, env)
    with open('Qfunction.pkl', 'rb') as inp:
        agent.Q = pickle.load(inp)
    return agent

