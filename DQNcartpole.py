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
import neptune

import itertools
import random
import time
import pandas as pd
import numpy as np





random.seed(time.time())


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


class experience_replay(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_5_layer(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_5_layer, self).__init__()
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


class DQN_3_layer(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_3_layer, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)






def train_agent(env, num_episodes, current_net, target_net, optimizer, memory, params, run):
    """
    Train the DQN agent.

    Args:
        env: Gym environment.
        num_episodes: Number of training episodes.
        current_net: The DQN network for action selection.
        target_net: The target DQN network.
        optimizer: Optimizer for training the current_net.
        memory: Replay buffer.
        params: Dictionary containing training parameters.
        run: Neptune run object for logging.

    Returns:
        current_net: The most updated trained DQN agent.
        episode_durations: A Pandas Series containing the durations of each episode.
    """
    EPS_START = params['eps_start']
    EPS_END = params['eps_end']
    EPS_DECAY = params['eps_decay']
    BATCH_SIZE = params['batch_size']
    GAMMA = params['gamma']
    TAU = params['tau']
    C = params['target_update_frequency']

    steps_done = 0
    episode_durations = []
    last_episode_durations = deque(maxlen=100)

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return current_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return

        sample_batch = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*sample_batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = current_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        run["train/loss"].append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(current_net.parameters(), 100)
        optimizer.step()

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

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
                break

    return current_net, pd.Series(episode_durations, name="durations")

def test_agent(env, trained_net, render=False):
    """
    Test the trained DQN agent.

    Args:
        env: Gym environment.
        trained_net: The trained DQN network.
        render: Whether to render the environment during testing.

    Returns:
        total_rewards: Total rewards accumulated during the episode.
    """

    env = gym.make("CartPole-v1",render_mode='human')
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_rewards = 0
    for t in count():
        if render:
            env.render()

        with torch.no_grad():
            action = trained_net(state).max(1).indices.view(1, 1)

        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_rewards += reward

        if terminated or truncated:
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    env.close()
    return total_rewards


test=True #run test agent after the trained game
num_episodes=600
DQN=DQN_3_layer
hyperparams = {
    "LR": [0.01,0.001,0.0001],
    "GAMMA": [0.95,0.97, 0.99],
    "batch_size": [32],
    "tau": [0.01,0.1,0.5],
    "replay_memory_size": [16000]
}
hyperparams = {
    "LR": [0.001],
    "GAMMA": [ 0.99],
    "batch_size": [32],
    "tau": [0.1],
    "replay_memory_size": [16000]
}


param_combinations = list(itertools.product(
    hyperparams["LR"],
    hyperparams["GAMMA"],
    hyperparams["batch_size"],
    hyperparams["tau"],
    hyperparams["replay_memory_size"]
))

# Usage Example:
for i, (lr, gamma, batch_size, tau, memory_size) in enumerate(param_combinations):
    print(f"Testing combination {i + 1}/{len(param_combinations)}:")
    print(f"lr={lr}, gamma={gamma}, batch_size={batch_size}, tau={tau}, memory_size={memory_size}")

    run = neptune.init_run(
        project="benmaman/RL-proj1-section2-5layer",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYjZiNWViYS1kYzc4LTQxYmUtOWUxNC02NzI5MzRjZGU5ZDcifQ==",
    ) 
    params = {
        "eps_start": 0.9,
        "eps_end": 0.001,
        "eps_decay": 2000,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "target_update_frequency": 1
    }

    memory = experience_replay(memory_size)
    env = gym.make("CartPole-v1")
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    current_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(current_net.state_dict())
    optimizer = optim.AdamW(current_net.parameters(), lr=lr, amsgrad=True)

    run["params"] = params
    current_net, durations = train_agent(env, num_episodes, current_net=current_net, target_net=target_net,
                                         optimizer=optimizer, memory=memory, params=params, run=run)
    print("Training Complete.")
    if test==True:
        test_agent(env,current_net,True)
    run.stop()





# # Loop through each combination
# for i, (lr, gamma, batch_size, tau, memory_size) in enumerate(param_combinations):
#     print(f"Testing combination {i + 1}/{len(param_combinations)}:")
#     print(f"lr={lr}, gamma={gamma}, batch_size={batch_size}, tau={tau}, memory_size={memory_size}")
    
#     run = neptune.init_run(
#         project="benmaman/RL-proj1-section2-5layer",
#         api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYjZiNWViYS1kYzc4LTQxYmUtOWUxNC02NzI5MzRjZGU5ZDcifQ==",
#     ) 

#     BATCH_SIZE = batch_size
#     GAMMA = gamma
#     EPS_START = 0.9
#     EPS_END = 0.001
#     EPS_DECAY = 2000
#     TAU = tau # update rate of the target network - soft update
#     LR = lr
#     REPLAY_MEMORY_SIZE = memory_size
#     num_episodes=600
#     C = 1   # number of steps to update the target network
#     K = 1   # number of target networks to average over


#     env = gym.make("CartPole-v1")
#     n_actions = env.action_space.n
#     n_states = env.observation_space.shape[0]

#     state, info = env.reset()
#     n_observations = len(state)

#     last_k_target_nets = deque(maxlen=K)

#     current_net = DQN(n_observations, n_actions).to(device)
#     target_net = DQN(n_observations, n_actions).to(device)
#     target_net.load_state_dict(current_net.state_dict())




#     steps_done = 0
#     last_episode_durations = deque(maxlen=100)


#     optimizer = optim.AdamW(current_net.parameters(), lr=LR, amsgrad=True)
#     memory = 	experience_replay(REPLAY_MEMORY_SIZE)
#     params = {"type": "Double DQN",
#             "learning_rate": LR,
#             "discount factor": GAMMA,
#             "batch size": BATCH_SIZE,
#             "tau": TAU,
#             "C": C,
#             "optimizer": type(optimizer).__name__}
#     run["params"] = params





#     episode_durations = []


#     def plot_durations(show_result=False):
#         plt.figure(1)
#         durations_t = torch.tensor(episode_durations, dtype=torch.float)
#         if show_result:
#             plt.title('Result')
#         else:
#             plt.clf()
#             plt.title('Training...')
#         plt.xlabel('Episode')
#         plt.ylabel('Duration')
#         plt.plot(durations_t.numpy())
#         # Take 100 episode averages and plot them too
#         if len(durations_t) >= 100:
#             means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#             means = torch.cat((torch.zeros(99), means))
#             plt.plot(means.numpy())

#         plt.pause(0.001)  # pause a bit so that plots are updated
#         if is_ipython:
#             if not show_result:
#                 display.display(plt.gcf())
#                 display.clear_output(wait=True)
#             else:
#                 display.display(plt.gcf())


#     def optimize_model(t):
#         if len(memory) < BATCH_SIZE:
#             return

#         sample_batch = memory.sample(BATCH_SIZE)
#         batch = Transition(*zip(*sample_batch))

#         # Compute a mask of non-final states and concatenate the batch elements
#         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                             batch.next_state)), device=device, dtype=torch.bool)
#         non_final_next_states = torch.cat([s for s in batch.next_state
#                                                     if s is not None])
#         state_batch = torch.cat(batch.state)
#         action_batch = torch.cat(batch.action)
#         reward_batch = torch.cat(batch.reward)

#         state_action_values = current_net(state_batch).gather(1, action_batch)
#         next_state_values = torch.zeros(BATCH_SIZE, device=device) # 0 in case the state was final.
#         with torch.no_grad():
#             next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values # double Q learning
#         # Compute the expected Q values
#         expected_state_action_values = (next_state_values * GAMMA) + reward_batch


#         # Huber loss
#         # huber = nn.SmoothL1Loss()
#         mse = nn.MSELoss()
#         loss = mse(state_action_values, expected_state_action_values.unsqueeze(1))
#         # for epoch in range(10):
#         run["train/loss"].append(loss)
#         # Optimize the model
#         optimizer.zero_grad()
#         loss.backward()
#         # In-place gradient clipping
#         # torch.nn.utils.clip_grad_norm_(current_net.parameters(), 100)
#         torch.nn.utils.clip_grad_value_(current_net.parameters(), 100)
#         optimizer.step()


#     for i_episode in range(num_episodes):
#         # global C

#         if i_episode % 100 == 0:
#             random.seed(time.time())
#         state, info = env.reset()
#         state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#         for t in count():
#             sample_action,steps_done = select_action(state,EPS_END,EPS_START,steps_done,EPS_DECAY)
#             observation, reward, terminated, truncated, _ = env.step(sample_action.item())
#             reward = torch.tensor([reward], device=device)
#             done = terminated or truncated

#             if terminated:
#                 next_state = None
#             else:
#                 next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

#             memory.push(state, sample_action, next_state, reward)
#             state = next_state
#             optimize_model(t)

#             # Every C steps soft update of the target network's weights
#             if t % C == 0:
#                 current_net_state_dict = current_net.state_dict()
#                 target_net_state_dict = target_net.state_dict()
#                 for key in current_net_state_dict:
#                     target_net_state_dict[key] = current_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
#                 target_net.load_state_dict(target_net_state_dict)



#             if done:
#                 episode_durations.append(t + 1)
#                 last_episode_durations.append(t + 1)
#                 run["train/duration"].append(t + 1)
#                 run["train/last_100_avg_duration"].append(np.mean(last_episode_durations))
                
#                 # plot_durations()
#                 break

#     double_dqn_durations = pd.Series(episode_durations, name="durations")

#     print('Complete')
#     run.stop()
#     # plot_durations(show_result=True)
#     plt.ioff()
#     plt.show()