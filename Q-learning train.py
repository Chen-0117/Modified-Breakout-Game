import gym
from QLearning import Agent, saveQ
from gymnasium.wrappers import TimeLimit
from Qlearning_hyperparameter import Hyperparameters
import os
from Wrapper import ActionWrapper
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

# Set up the environment variables for ROMs
os.environ['PYTHONWARNINGS'] = "default::ImportWarning:ale_py.roms"
os.environ['ALE_PY_ROM_PATH'] = "./roms"

# Load hyperparameters
hyperparams = Hyperparameters()

# Initialize the Breakout-v4 environment
env = TimeLimit(gym.make(hyperparams.env_name), max_episode_steps=hyperparams.max_episode_steps)
env = ActionWrapper(env, hyperparams.rand)

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
print("Starting training.")
# Train the agent using Q-Learning
Q, reward_array = agent.QLearning(hyperparams.n_episodes)
print("Saving trained agent.")
# Save the trained agent
saveQ(agent)
print("Closing training environment.")
# Ensure the environment is closed properly
env.close()







