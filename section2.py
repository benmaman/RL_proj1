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
def train_agent(env, hidden_layers,HYPERPARAMS):
    # Initialize neural networks and experience replay
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_network = DQN(state_dim, action_dim, hidden_layers)
    target_network = DQN(state_dim, action_dim, hidden_layers)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    optimizer = optim.Adam(q_network.parameters(), lr=HYPERPARAMS['learning_rate'],amsgrad=True)
    memory = ExperienceReplay(HYPERPARAMS['memory_size'])
    
    epsilon = HYPERPARAMS['epsilon_start']
    epsilon_decay_rate = (HYPERPARAMS['epsilon_start'] - HYPERPARAMS['epsilon_end']) / HYPERPARAMS['epsilon_decay']
    rewards_per_episode = []
    losses = []
    
    # store the best results
    best_episode=-1
    best_score=-1

    for episode in tqdm(range(HYPERPARAMS['n_episodes'])):
        state, _ = env.reset()
        total_reward = 0
        episode_loss=0

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
                sample_batch = memory.sample(HYPERPARAMS['batch_size'])
                states, sample_action, rewards, next_states, dones = zip(*sample_batch)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(sample_action).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
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
                episode_loss+=loss.item()
            if done:
                break
        
        # Decay epsilon
        epsilon = max(HYPERPARAMS['epsilon_end'], epsilon - epsilon_decay_rate)
        
        # Update target network
        if episode % HYPERPARAMS['target_update'] == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        rewards_per_episode.append(total_reward)
        losses.append(episode_loss / max(1, t))
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Loss: {episode_loss:.4f}, Epsilon: {epsilon:.4f}")
            if episode >100:
                long_avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"Wow the model reach average reward of:{long_avg_reward} in the last 100 consectutive, current episode: {episode}")
                if long_avg_reward>450:
                    best_score=long_avg_reward
                    best_episode=episode

    return target_network, rewards_per_episode, losses,best_score,best_episode

# Testing function
def test_agent(env, q_network, n_episodes=100, render=True):
    total_rewards = []
    for episode in range(n_episodes):
        # Reset the environment
        state, _ = env.reset()
        
        # Randomize the initial state to ensure varied scenarios
        env.state = np.random.uniform(
            low=-0.05, high=0.05, size=(4,)
        )  # `CartPole-v1` has a 4-dimensional state
        
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