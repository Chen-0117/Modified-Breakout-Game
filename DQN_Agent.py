from DQN_Model import DQN
from DDQN_hyperparameters import Hyperparameters
import numpy as np
import cv2
import torch
import torch.nn as nn
from copy import deepcopy
import gym
from collections import deque
import random

class DQNAgent(nn.Module):
    def __init__(
            self, env: gym.Env,
            hyperparameters: Hyperparameters,
            device = False
    ):
        super().__init__()
        if not device:
            if torch.backends.cuda.is_built():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.env = env
        self.hp = hyperparameters
        self.action_space = self.env.action_space.n
        self.epsilon = self.hp.epsilon
        self.e_min = self.hp.epsilon_min
        self.batch_size = self.hp.batch_size
        self.online = DQN(self.action_space).to(self.device) #
        self.T = deepcopy(self.online)
        self.discount_factor = self.hp.discount_factor #
        self.r = self.hp.targetDQN_update_rate

        # no need to track target network's gradients
        for p in self.T.parameters():
            p.requires_grad = False

        self.update_target_network()
        self.memory = deque(maxlen=self.hp.buffer_size)

    def preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)[32:, 8:152]
        state = cv2.resize(src=state, dsize=(84, 84)) / 255.
        return torch.from_numpy(state.astype(np.float32)).to(self.device)

    def update_target_network(self):
        # sync main and target network
        self.T.load_state_dict(self.online.state_dict())

    def act(self, state):
        # epsilon greedy strategy:
        # we select random action with epsilon prob
        # and follow policy otherwise

        if np.random.uniform(0, 1) <= (self.epsilon):
            action = np.random.randint(self.action_space)
        else:
            action = self.greedy(state)
        return action

    def greedy(self, state):
        state = state.unsqueeze(0)
        self.online.eval()
        with torch.no_grad():
            actions = self.online(state)
        self.online.train()
        return torch.argmax(actions).item()

    def cache(self, exp):
        # store data in replay buffer
        s, a, r, s_, d = exp
        a = torch.tensor(a).to(self.device)
        r = torch.sign(torch.tensor(r)).to(self.device)
        d = torch.tensor(d).to(self.device)
        self.memory.append((s, a, r, s_, d))

    def memory_size(self):
        return len(self.memory)

    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)

    def learn(self):
        if self.memory_size() < self.batch_size: return
        exps = self.sample_memory()
        s, a, r, s_, d = map(torch.stack, zip(*exps))

        pred_q = self.online(s)[np.arange(self.batch_size), a]

        # bellman backup for DQN algorithm, here action selection and q value
        # computation is both done using target network
        with torch.no_grad():
            # Compute the maximum Q-value for off-policy update and call it <next_target_q_value>
            next_target_q_value = self.T(s_).max(dim=1, keepdim=True)[0]

        target_q = r + (self.discount_factor * next_target_q_value * (1 - d.float())).squeeze(1)

        # backprop
        self.online.optimizer.zero_grad()
        loss = self.online.loss_fn(pred_q, target_q)
        current_loss = loss
        loss.backward()
        self.online.optimizer.step()
        return current_loss.item()

    def save(self, steps):
        torch.save(
            dict(model=self.online.state_dict(), exploration_rate=self.epsilon),
            f'DQNAgent-{steps}.chkpt',
        )

    def load(self, path):
        model_state_dict, epsilon = torch.load(path).values()
        self.online.load_state_dict(model_state_dict)
        self.T.load_state_dict(model_state_dict)
        self.epsilon = epsilon


class DDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(self):
        if self.memory_size() < self.batch_size:
            return

        exps = self.sample_memory()
        state, action, reward, next_state, done = map(torch.stack, zip(*exps))

        # bellman backup for DDQN algorithm, here action selection
        pred_q = self.online(state)[np.arange(self.batch_size), action]
        target_q = reward + (1 - done.float()) * self.discount_factor * self.T(next_state)[
            np.arange(self.b),
            self.Q(next_state).argmax(axis=1)
        ]

        # backprop
        self.online.optimizer.zero_grad()
        loss = self.online.loss_fn(pred_q, target_q)
        current_loss = loss
        loss.backward()
        self.online.optimizer.step()

        return current_loss.item()

    def save(self, steps):
        torch.save(
            dict(model=self.online.state_dict(), exploration_rate=self.epsilon),
            f'DDQNAgent-{steps}.chkpt',
        )


