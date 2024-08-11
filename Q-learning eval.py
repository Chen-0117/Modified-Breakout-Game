import gym
from QLearning import loadAgent
from gymnasium.wrappers import TimeLimit
from Qlearning_hyperparameter import Hyperparameters
from warnings import filterwarnings
import moviepy.editor as mpy
from Wrapper import ActionWrapper
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

# Set up the environment variables for ROMs
import os
os.environ['PYTHONWARNINGS'] = "default::ImportWarning:ale_py.roms"
os.environ['ALE_PY_ROM_PATH'] = "./roms"

# Load hyperparameters
hyperparams = Hyperparameters()

# Initialize the Breakout-v4 environment
env = TimeLimit(gym.make(hyperparams.env_name, render_mode='rgb_array'), max_episode_steps=hyperparams.max_episode_steps)
# env = ActionWrapper(env, hyperparams.rand)

# Define the number of actions from the environment
n_actions = env.action_space.n

# Load the trained agent
agent = loadAgent(
    hyperparams.n_states,
    n_actions,
    discount=hyperparams.discount,
    lr=hyperparams.learning_rate,
    epsilon=hyperparams.epsilon,
    epsilon_decay=hyperparams.epsilon_decay,
    min_epsilon=hyperparams.min_epsilon,
    env=env
)


def combine_videos(video_folder, output_path):
    # List all the video files in the folder
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
    video_files.sort()  # Ensure the files are in the correct order

    # Load each video file into a VideoFileClip object
    clips = [mpy.VideoFileClip(video) for video in video_files]

    # Concatenate all the clips into a single video
    final_clip = mpy.concatenate_videoclips(clips)

    # Write the result to a file
    final_clip.write_videofile(output_path, codec='libx264', fps=24)

# Example usage after evaluation:
n_test = 25 # Number of episodes to evaluate
success_rate, avg_reward = agent.eval(n_test)
print(f'Success rate: {success_rate}, Average reward: {avg_reward}')

# Combine the videos
combine_videos('recordings', 'combined_output.mp4')