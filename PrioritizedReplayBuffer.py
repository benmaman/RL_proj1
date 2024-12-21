import copy

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchrl.data import ListStorage, PrioritizedReplayBuffer
torch.manual_seed(0)

import neptune

import random
import time
import pandas as pd
import numpy as np

run = neptune.init_run(
    project="benmaman/rl-proj1-section3",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYjZiNWViYS1kYzc4LTQxYmUtOWUxNC02NzI5MzRjZGU5ZDcifQ==",
)  # your credentials

random.seed(time.time())

num_episodes=600


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, n_actions)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        x = F.gelu(self.layer3(x))
        x = F.gelu(self.layer4(x))
        x = self.dropout(x)
        return self.layer5(x)


BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.001
EPS_DECAY = 2000
TAU = 0.01
LR = 1e-4
REPLAY_MEMORY_SIZE = 16000

C = 1  # number of steps to update the target network

TERMINAL_STATE = torch.tensor([torch.inf, torch.inf, torch.inf, torch.inf], device=device, dtype=torch.float32).view(4,)

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

state, info = env.reset()
n_observations = len(state)



current_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(current_net.state_dict())



steps_done = 0
last_episode_durations = deque(maxlen=100)

optimizer = optim.AdamW(current_net.parameters(), lr=LR, amsgrad=True)

prioritized_memory = PrioritizedReplayBuffer(batch_size=BATCH_SIZE, alpha=1.0, beta=1.0, eps=1e-6, storage=ListStorage(REPLAY_MEMORY_SIZE))

params = {"type": "prioritized experience buffer DQN",
          "learning_rate": LR,
          "discount factor": GAMMA,
          "batch size": BATCH_SIZE,
          "tau": TAU,
          "C": C,
          "optimizer": type(optimizer).__name__}
run["params"] = params


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            s = state.unsqueeze(0)
            a = current_net(s)
            return a.max(1).indices
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model(t):
    if len(prioritized_memory) < BATCH_SIZE:
        return

    transitions, info = prioritized_memory.sample(return_info=True)
    # batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    mask = transitions.next_state == TERMINAL_STATE # device=device, dtype=torch.bool)
    final_state_mask = mask.all(dim=1)
    non_final_next_states = transitions.next_state[~final_state_mask]
    state_action_values = current_net(transitions.state).gather(1, transitions.action).squeeze(1)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[~final_state_mask] = target_net(non_final_next_states).max(1).values  # double Q learning

    expected_state_action_values = (next_state_values * GAMMA) + transitions.reward.squeeze(1)
    td_errors = state_action_values - expected_state_action_values
    run["train/td_errors"].append(td_errors.mean())
    run["train/weights"].append(td_errors.abs().mean())
    prioritized_memory.update_priority(info["index"], td_errors.abs())
    # Huber loss
    # huber = nn.SmoothL1Loss()
    mse = nn.MSELoss()
    loss = mse(state_action_values, expected_state_action_values)

    run["train/weights"].append(info["_weight"].mean())
    weights = torch.tensor(info["_weight"], device=device)


    run["train/loss"].append(loss)
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_norm_(current_net.parameters(), 100)
    torch.nn.utils.clip_grad_value_(current_net.parameters(), 100)
    optimizer.step()




for i_episode in range(num_episodes):
    # global C
    if i_episode % 100 == 0:
        random.seed(time.time())
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device) #.unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = TERMINAL_STATE  # None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)

        prioritized_memory.add(Transition(state, action, next_state, reward))
        state = next_state
        optimize_model(t)

        if t % C == 0:
            current_net_state_dict = current_net.state_dict()
            target_net_state_dict = target_net.state_dict()
            for key in current_net_state_dict:
                target_net_state_dict[key] = current_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            last_episode_durations.append(t + 1)
            run["train/duration"].append(t + 1)
            run["train/last_100_avg_duration"].append(np.mean(last_episode_durations))
            # plot_durations()
            break

double_dqn_durations = pd.Series(episode_durations, name="durations")

print('Complete')
run.stop()
plot_durations(show_result=True)
plt.ioff()
plt.show()
























###################################################################################################
import copy
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torchrl.data import ListStorage, PrioritizedReplayBuffer
torch.manual_seed(0)

import neptune

import random
import time
import pandas as pd
import numpy as np


# run = neptune.init_run(
#     project="yaarigur/DRLass1",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NWEwODEwOS0zZDY0LTQzNjQtYmM1YS0xNjUxY2VkNjI2NzEifQ==",
# )  # your credentials



random.seed(time.time())







# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)




Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, n_actions)
        self.dropout = nn.Dropout(p=0.1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        x = F.gelu(self.layer3(x))
        x = F.gelu(self.layer4(x))
        # x = self.dropout(x)
        return self.layer5(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 64
GAMMA = 0.99 # a lower makes rewards from the uncertain far future less important. encourage agents to collect reward closer in time
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TAU = 0.005
LR = 1e-4
REPLAY_MEMORY_SIZE = 16000
FINAL_STATE = torch.tensor([torch.inf, torch.inf, torch.inf, torch.inf], device=device, dtype=torch.float32).unsqueeze(0)

env = gym.make("CartPole-v1")
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
n_states = env.observation_space.shape[0]

state, info = env.reset()
n_observations = len(state)

current_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
# target_net.load_state_dict(current_net.state_dict())

steps_done = 0
C = 10   # number of steps to update the target network
last_episodes_durations = deque(maxlen=100)
K = 1
last_k_target_nets = deque(maxlen=K)
last_k_target_nets.append(copy.deepcopy(target_net))


optimizer = optim.AdamW(current_net.parameters(), lr=LR, amsgrad=True)
# memory = ReplayMemory(REPLAY_MEMORY_SIZE)
prioritized_memory = PrioritizedReplayBuffer(batch_size=BATCH_SIZE, alpha=0.7, beta=0.9, storage=ListStorage(REPLAY_MEMORY_SIZE))

# params = {"learning_rate": LR,
#           "discount factor": GAMMA,
#           "batch size": BATCH_SIZE,
#           "tau": TAU,
#           "C": C,
#           "optimizer": type(optimizer).__name__}
# run["params"] = params



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return current_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model(t):
    if len(prioritized_memory) < BATCH_SIZE:
        return
    # print(len(prioritized_memory))

    batch, info = prioritized_memory.sample(return_info=True)
    # print(info)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # batch = Transition(*zip(*transitions))


    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not FINAL_STATE,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not FINAL_STATE])

    # print("non_final_next_states:", non_final_next_states)
    # print("non_final_mask:", non_final_mask)

    print("batch.state:", batch.state.shape)
    stat_batch = batch.state.squeeze() # state_batch = torch.cat(batch.state)
    print("stat_batch:", stat_batch.shape)
    print("stat_batch:", stat_batch)
    action_batch = batch.action.squeeze(-1) # action_batch = torch.cat(batch.action)
    reward_batch = batch.reward.squeeze(-1) # reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_action_values = current_net(state_batch).gather(1, action_batch)
    print("batch_action:", action_batch.shape)
    print("batch_action:", action_batch)
    print("current_net(stat_batch): ", current_net(stat_batch))
    state_action_values = current_net(stat_batch).gather(1, action_batch)
    # print("state_action_values:", state_action_values.shape)
    print("state_action_values:", state_action_values)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_k_state_values = torch.zeros(K, BATCH_SIZE, device=device)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values  # double Q learning
        for k in range(len(last_k_target_nets)):
            next_k_state_values[k][non_final_mask] = (last_k_target_nets[k])(non_final_next_states).max(1).values
    # Compute the expected Q values
    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    print("next_k_state_values:", next_k_state_values.shape)
    print("batch.reward:", batch.reward.shape)
    expected_state_action_values = (next_k_state_values.mean(dim=0) * GAMMA) + reward_batch

    print("expected_state_action_values:", expected_state_action_values.shape)
    print("state_action_values:", state_action_values.shape)
    print("state_action_values:", state_action_values)
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values # double Q learning
    #     next_k_state_values[non_final_mask] = avg_k_target_nets(non_final_next_states)
    # Compute the expected Q values
    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    huber = nn.SmoothL1Loss()
    mse = nn.MSELoss()
    loss = huber(state_action_values, expected_state_action_values.unsqueeze(1))

    # run["train/loss"].append(0.9 ** t)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(current_net.parameters(), 100)
    optimizer.step()



for i_episode in range(num_episodes):
    # global C
    # Initialize the environment and get its state
    if i_episode % 100 == 0:
        random.seed(time.time())
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        if action.item() < 0:
            print("state:", state)
            print("action:", action)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = FINAL_STATE # None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        # print("transition:", state, action, next_state, reward)
        prioritized_memory.add(Transition(state, action, next_state, reward))


        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(t)

        # Every C steps soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        if t % C == 0:
            target_net_state_dict = target_net.state_dict()
            current_net_state_dict = current_net.state_dict()
            for key in current_net_state_dict:
                target_net_state_dict[key] = current_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            last_k_target_nets.append(copy.deepcopy(target_net))

        if done:
            episode_durations.append(t + 1)
            last_episodes_durations.append(t + 1)
            # run["train/duration"].append(t + 1)
            # run["train/last_100_avg_duration"].append(np.mean(last_episodes_durations))
            # plot_durations()
            break

double_dqn_durations = pd.Series(episode_durations, name="durations")

print('Complete')
# run.stop()
plot_durations(show_result=True)
plt.ioff()
plt.show()