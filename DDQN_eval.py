from DQN_Agent import DQNAgent
from DDQN_hyperparameters import Hyperparameters
import numpy as np
import torch
import gym
from tqdm import tqdm
from Wrapper import ActionWrapper
import matplotlib.pyplot as plt
from gym.wrappers.record_video import RecordVideo

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

from DDQN_train import DDQNAgent



from collections import deque
def eval(env):
    steps_recording = []
    rewards_recording = []
    loss_recording = []
    average_reward = 0
    steps = 0
    env = RecordVideo(env, video_folder="result", episode_trigger=lambda episode_number: True)
    env = ActionWrapper(env, hp.uncertain_prob)
    ini_state = None
    for episode in range(25):
        state, info = env.reset()

        old_lives = info['lives']

        # if episode == 0:
        #     ini_state = state
        # if episode == 1:
        #     print(np.array_equal(ini_state, state))
        #     break


        # shoot ball so that the ball becomes visible
        state = env.step(1)
        state = state[0]
        state = agent.preprocess(state)
        state = torch.stack([state, state, state, state])

        ep_steps = 0
        total_reward = 0
        ended = False
        truncated = False
        while not ended or truncated:
            action = agent.act(state)
            (next_state, reward, ended, truncated, info) = env.step(action)

            next_state = agent.preprocess(next_state)

            next_state = torch.stack([*state[1:], next_state])

            agent.cache((state, action, reward, next_state, ended or truncated))
            state = next_state
            total_reward += reward
            ep_steps += 1
            steps +=1

            if old_lives > info['lives']:
                state = env.step(1)
                state = state[0]
                state = agent.preprocess(state)
                state = torch.stack([state, state, state, state])
                old_lives = info['lives']

        rewards_recording.append(total_reward)
        #loss_recording.append(current_loss)

        steps_recording.append(ep_steps)
        print(f"Episode: {episode} Total Reward: {total_reward} Average Reward: {total_reward/5} ep_Steps: {ep_steps}  Steps: {steps}")

    print(f"Average Reward Episode: {sum(rewards_recording)/25}")

    return steps_recording, rewards_recording, loss_recording,sum(rewards_recording)/25

r = []
for i in range(10):

    hp = Hyperparameters()
    env = gym.make('Breakout-v4', render_mode='rgb_array')
    env = ActionWrapper(env, hp.uncertain_prob)
    env.reset()

    env.metadata['render_fps'] = 24

    ## Agent
    agent = DDQNAgent(
        env,
        hp
    )

    agent.load("DDQNAgent-final-500(1).chkpt")

    steps_recording, rewards_recording, loss_recording, ar = eval(env)
    r.append(ar)

print(r)

plt.figure()
plt.title("Average Reward")
plt.plot(r, label='average reward', color='green')
plt.xlabel("iteration")
plt.ylabel("Average Reward")
plt.show()

