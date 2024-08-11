import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_action_space, learning_rate=1e-4):
        super().__init__()
        self.num_action_space = num_action_space

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_action_space)

        self.relu = nn.ReLU()

        self.loss_fn = nn.MSELoss()
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #x = self.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
