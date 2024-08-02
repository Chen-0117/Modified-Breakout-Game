import gym
from QLearning import Agent, saveQ, loadAgent
from gymnasium.wrappers import TimeLimit
from hyperparameter import Hyperparameters

# Set up the environment variables for ROMs
import os
os.environ['PYTHONWARNINGS'] = "default::ImportWarning:ale_py.roms"
os.environ['ALE_PY_ROM_PATH'] = "./roms"

# # Create the environment
# env = gym.make("Breakout-v4")

# Load hyperparameters
hyperparams = Hyperparameters()

# Initialize the Breakout-v4 environment
env = TimeLimit(gym.make(hyperparams.env_name), max_episode_steps=hyperparams.max_episode_steps)

# Define the number of actions from the environment
n_actions = env.action_space.n

# Instantiate the agent
agent = Agent(
    hyperparams.n_states,
    n_actions,
    hyperparams.discount,
    hyperparams.learning_rate,
    hyperparams.epsilon,
    hyperparams.epsilon_decay,
    hyperparams.min_epsilon,
    env
)

# Train the agent using Q-Learning
Q, reward_array = agent.QLearning(hyperparams.n_episodes)

# Save the trained agent
saveQ(agent)

# Evaluate the trained agent
success_rate = agent.eval(100)
print(f'Success rate: {success_rate}')

