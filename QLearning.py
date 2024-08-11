import gymnasium as gym
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import pickle
import torch
import gzip
import pickle
import os
import time
import cv2
import traceback
from gym.wrappers.record_video import RecordVideo

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

    def preprocess_state(self, state, new_size=(84, 84)):
        """
        Preprocess the input state by cropping, converting to grayscale, and resizing.

        :param state: Original state (image).
        :param new_size: Tuple indicating the new size (height, width) of the resized state.
        :return: Processed state as a tuple.
        """
        # Crop the image to remove the scoreboard and other irrelevant parts
        cropped_state = state[35:195]  # Cropping from 35 to 195 vertically

        # Convert to grayscale
        gray_state = cv2.cvtColor(cropped_state, cv2.COLOR_RGB2GRAY)

        # Resize the image
        resized_state = cv2.resize(gray_state, new_size, interpolation=cv2.INTER_AREA)

        # Flatten and convert to tuple
        return tuple(resized_state.flatten())

    def epsGreedy(self, Q, s):
        p = np.random.uniform(0, 1)
        if p <= self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            ActionValues = Q[s]
            max_indices = np.where(ActionValues == np.max(ActionValues))[0]
            action = np.random.choice(max_indices)
        return action

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

                # Debug information
                # print(f"Training Episode {k+1}, State {s[:5]}, Action {a}, Reward {reward}, Terminated {terminated}, Q[s][a]: {self.Q[s][a]}, G: {G}")

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

    def eval(self, n_episodes, Q=None, record_dir='recordings'):
        if Q is not None:
            self.Q = Q
            
        # Create directory for recordings if it doesn't exist
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
            
        # Wrap the environment with the RecordVideo wrapper
        env = RecordVideo(self.env, video_folder=record_dir, episode_trigger = lambda episode_number: True)
        env.metadata['render_modes'] = ['rgb_array']
        env.metadata['render_fps'] = 30

        K = trange(n_episodes, desc="Evaluating", leave=True)
        total_reward_sum = 0
        success_count = 0

        for k in K:
            state, _ = env.reset()
            s = self.preprocess_state(state)
            if s not in self.Q:
                self.Q[s] = np.zeros(self.n_actions)

            terminated = False
            total_reward = 0
            step_count = 0

            while not terminated:
                if s in self.Q:
                    ActionValues = self.Q[s]
                    max_indices = np.where(ActionValues == np.max(ActionValues))[0]
                    a = np.random.choice(max_indices)
                else:
                    a = env.action_space.sample()

                next_state, reward, terminated, _, _ = env.step(a)
                step_count += 1
                total_reward += reward

                s_next = self.preprocess_state(next_state)
                if s_next not in self.Q:
                    self.Q[s_next] = np.zeros(self.n_actions)

                if terminated:
                    K.set_description(f'Episode {k+1} ended with Reward {total_reward}')
                    K.refresh()
                    total_reward_sum += total_reward
                    if total_reward > 0:
                        success_count += 1
                    print(f"Episode {k+1} ended after {step_count} steps with total reward {total_reward}.")
                    break

                s = s_next

                if step_count > 10000:
                    print(f"Episode {k+1} stuck in infinite loop, terminating manually after {step_count} steps.")
                    terminated = True
                    reward = 0
    
        env.close()
        print(f'Overall success rate: {success_count / n_episodes}')
        print(f'Average reward per episode: {total_reward_sum / n_episodes}')
        return success_count / n_episodes, total_reward_sum / n_episodes

def saveQ(agent: Agent, filename='Qfunction1.pkl'):
    try:
        print("Starting to save Q-values.")
        start_time = time.time()
        total_entries = len(agent.Q)
        
        with gzip.open(filename, 'wb') as outp:
            # Track progress
            keys = list(agent.Q.keys())
            chunk_size = 1000
            
            for i in range(0, total_entries, chunk_size):
                chunk_keys = keys[i:i + chunk_size]
                chunk_data = {key: agent.Q[key] for key in chunk_keys}
                pickle.dump(chunk_data, outp, pickle.HIGHEST_PROTOCOL)
                
                # Log progress every chunk
                print(f"Saved {min(i + chunk_size, total_entries)}/{total_entries} entries ({(min(i + chunk_size, total_entries)) / total_entries:.2%})")
        
        end_time = time.time()
        file_size = os.path.getsize(filename)
        print(f"Finished saving Q-values. Time taken: {end_time - start_time:.2f} seconds. File size: {file_size / (1024 * 1024):.2f} MB.")
    except Exception as e:
        print(f"Error saving Q-values: {e}")
        traceback.print_exc()

def loadAgent(n_states, n_actions, discount, lr, epsilon, epsilon_decay, min_epsilon, env, filename='Q-learning weights/Qfunction1.pkl'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
    if os.path.getsize(filename) == 0:
        raise EOFError(f"The file {filename} is empty.")

    agent = Agent(n_states, n_actions, discount, lr, epsilon, epsilon_decay, min_epsilon, env)
    try:
        with gzip.open(filename, 'rb') as inp:
            agent.Q = pickle.load(inp)
        print(f"Q-values loaded from {filename}.")
    except EOFError as eof:
        print(f"EOFError: {eof}")
        raise
    except Exception as e:
        print(f"Error loading Q-values: {e}")
        raise
    return agent
