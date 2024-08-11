# Reinforcement Learning With Modified Breakout Game

## Abstract

The "Modified Breakout Game" is a project dedicated to implementing Reinforcement Learning (RL)
algorithms for playing the Atari breakout game. The Breakout game consists of a moving ball, a
paddle, and blocks and the goal of the game is to control the moving ball by batting it with the paddle
when it reaches the bottom of the frame, to knock as many blocks as possible. The game will be
terminated when the ball passes the paddle or knocks all blocks without losing the ball.
The environment that the project focused on with the frame has a dimension of 210 pixels x 160
pixels x 3 channels. The project is to develop two types of reinforcement learning agents: one uses a
traditional non-deep learning algorithm and another employs a deep reinforcement learning algorithm.
The main objective is to compare the effectiveness of these approaches in dealing with action
uncertainties, which are emblematic of issues faced in practical implementations of reinforcement
learning.

## Requirements

All library requirements are listed in the requirements.txt file.

```bash
$ pip install -r requirements.txt
```

## How to run the code

### Q-Learning

#### Training

1. Modify the filename parameter in saveQ() function in QLearning.py.

2. ```bash
   $ python train.py
   ```

3. The result will be saved in "Qfunction.pkl" file and a training curve will be plotted.

#### Evaluating

1. Modify the filename parameter in loadAgent() function in QLearning.py.

2. ``` bash
   $ python eval.py
   ```

3. Terminal will print the final average reward.

### DDQN

#### Training

1.  Modify the plot saving path in main.py.

2. ``` bash
   $ python main.py
   ```

#### Evaluating

1.  Modify the model weight path file in test.py.

2.  ```bash
    $ python test.py
    ```