import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the neural network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(DQN, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Define experience replay
class ExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
HYPERPARAMS = {
    'batch_size': 64,
    'gamma': 0.8,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 100,
    'learning_rate': 0.001,
    'target_update': 10,  # Frequency of target network update
    'memory_size': 10000,
    'n_episodes': 2000,
    'max_steps': 1000,
}

# Decaying epsilon-greedy action selection
def select_action(state, epsilon, n_actions, q_network):
    if np.random.rand() < epsilon:
        return random.choice(range(n_actions))  # Explore
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()  # Exploit


# Main function
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env.reset()

    # Train with 3 hidden layers
    print("Training with 3 hidden layers...")
    q_network_3, rewards_3, losses_3 = train_agent_double_dqn(env, hidden_layers=[512, 128, 64])
    
    # Train with 5 hidden layers
    print("Training with 5 hidden layers...")
    q_network_5, rewards_5, losses_5 =  q_network_3, rewards_3, losses_3
    
    # Plot rewards
    plt.plot(rewards_3, label="3 Hidden Layers")
    plt.plot(rewards_5, label="5 Hidden Layers")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards per Episode")
    plt.legend()
    plt.show()
    
    # Plot losses
    plt.plot(losses_3, label="3 Hidden Layers")
    plt.plot(losses_5, label="5 Hidden Layers")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss per Training Step")
    plt.legend()
    plt.show()
    
    # Test the agent
    print("Testing the agent with 5 hidden layers...")
    test_env = gym.make("CartPole-v1", render_mode="human")  # Render during testing

    avg_reward = test_agent(test_env, q_network_5, render=True)
    print(f"Average reward over 10 episodes: {avg_reward}")




import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from itertools import product
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the neural network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(DQN, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define experience replay
class ExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Decaying epsilon-greedy action selection
def select_action(state, epsilon, n_actions, q_network):
    if np.random.rand() < epsilon:
        return random.choice(range(n_actions))  # Explore
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()  # Exploit

# Training function
def train_agent_double_dqn(env, hidden_layers):
    # Initialize neural networks and experience replay
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_network = DQN(state_dim, action_dim, hidden_layers)
    target_network = DQN(state_dim, action_dim, hidden_layers)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    optimizer = optim.Adam(q_network.parameters(), lr=HYPERPARAMS['learning_rate'])
    memory = ExperienceReplay(HYPERPARAMS['memory_size'])
    
    epsilon = HYPERPARAMS['epsilon_start']
    epsilon_decay_rate = (HYPERPARAMS['epsilon_start'] - HYPERPARAMS['epsilon_end']) / HYPERPARAMS['epsilon_decay']
    rewards_per_episode = []
    losses = []
    
    for episode in tqdm(range(HYPERPARAMS['n_episodes'])):
        state, _ = env.reset()
        total_reward = 0
        for t in range(HYPERPARAMS['max_steps']):
            # Select action
            action = select_action(state, epsilon, action_dim, q_network)
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = reward - 3*abs(state[0])  # Penalize based on cart position

            done = terminated or truncated
            
            # Store experience
            memory.store((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward

            # Train Q-network
            if len(memory) >= HYPERPARAMS['batch_size']:
                minibatch = memory.sample(HYPERPARAMS['batch_size'])
                states, actions, rewards, next_states, dones = zip(*minibatch)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # Compute target Q-values using Double DQN
                with torch.no_grad():
                    # Use the online network to select the best action
                    next_actions = q_network(next_states).argmax(1).unsqueeze(1)
                    # Use the target network to evaluate the value of the selected action
                    next_q_values = target_network(next_states).gather(1, next_actions).squeeze(1)
                    target_q_values = rewards + HYPERPARAMS['gamma'] * next_q_values * (1 - dones)
                
                # Compute predicted Q-values
                current_q_values = q_network(states).gather(1, actions).squeeze(1)
                
                # Compute loss
                loss = nn.MSELoss()(current_q_values, target_q_values)
                losses.append(loss.item())
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(HYPERPARAMS['epsilon_end'], epsilon - epsilon_decay_rate)
        
        # Update target network
        if episode % HYPERPARAMS['target_update'] == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        rewards_per_episode.append(total_reward)
    
    return q_network, rewards_per_episode, losses



# Testing function
def test_agent(env, q_network, n_episodes=1000, render=True):
    total_rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        while True:
            if render:
                env.render()  # Render the environment
            action = select_action(state, 0.0, env.action_space.n, q_network)  # Fully exploit
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()
    return np.mean(total_rewards)

# Main function
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    summary={}
        # Hyperparameters
    HYPERPARAMS_grid = {
        'batch_size': [128],
        'gamma': [ 0.999],
        'epsilon_start': [0.9],
        'epsilon_end': [0.05],
        'epsilon_decay': [700],
        'learning_rate': [0.00005],
        'target_update': [ 10],
        'memory_size': [ 50000],
        'n_episodes': [600],
        'max_steps': [1000],
    }

    # Flatten the grid into combinations
    hyperparam_combinations = list(product(*HYPERPARAMS_grid.values()))
    param_names = list(HYPERPARAMS_grid.keys())

    # Store results
    results = []

    # Training loop for hyperparameter tuning
    for params in hyperparam_combinations:
        # Create a hyperparameter dictionary for the current combination
        current_params = dict(zip(param_names, params))
        print(f"Training with parameters: {current_params}")
        
        try:
            # Train the agent with the current hyperparameters
            q_network, rewards, losses, best_reward, best_episode = train_agent(
                env, [128, 128, 128], current_params
            )
            
            # Store the results
            results.append({
                **current_params,
                'best_reward': best_reward,
                'best_episode': best_episode
            })
            
        except Exception as e:
            # Handle exceptions gracefully and log failed configurations
            print(f"Failed with parameters {current_params}: {e}")
            results.append({
                **current_params,
                'best_reward': None,
                'best_episode': None,
                'error': str(e)
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv("resluts_for_3_layer")



    # Plot rewards
    plt.plot(rewards, label="3 Hidden Layers")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards per Episode")
    plt.legend()
    plt.show()
    
    # Plot losses
    plt.plot(losses, label="3 Hidden Layers")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss per Training Step")
    plt.legend()
    plt.show()
    
    # Test the agent
    print("Testing the agent")
    test_env = gym.make("CartPole-v1", render_mode="human")  # Render during testing

    avg_reward = test_agent(test_env, q_network, render=True)
    print(f"Average reward over 10 episodes: {avg_reward}")