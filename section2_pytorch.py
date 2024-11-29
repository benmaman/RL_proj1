import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Define the Neural Network
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(DQNetwork, self).__init__()
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


# Experience Replay
class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# Sample action using epsilon-greedy
def sample_action(state, epsilon, env, q_network):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state)
        return torch.argmax(q_values).item()


# Train Agent
def train_agent(env, q_network, target_network, replay_buffer, optimizer, batch_size, gamma, epsilon_decay, max_episodes):
    epsilon = 1.0
    min_epsilon = 0.01
    total_rewards = []
    losses = []

    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = sample_action(state, epsilon, env, q_network)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            if replay_buffer.size() >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.bool)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = target_network(next_states).max(1)[0]
                    target_q_values = rewards + gamma * max_next_q_values * (~dones)

                loss = nn.MSELoss()(q_values, target_q_values)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                total_rewards.append(total_reward)
                print(f"Episode {episode}, Reward: {total_reward}, Loss: {np.mean(losses[-10:]):.4f}, Epsilon: {epsilon:.4f}")

        # Update target network periodically
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Evaluate agent
        if episode % 10 == 0 and len(total_rewards) >= 100:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode}, Average Reward (last 100): {avg_reward:.2f}")
            if avg_reward >= 475.0:
                print(f"Solved in {episode} episodes!")
                break

    return total_rewards, losses


# Test Agent
def test_agent(env, q_network, render=True):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = sample_action(state, 0.0, env, q_network)  # Always choose the best action
        state, reward, done, _ = env.step(action)
        total_reward += reward
    env.close()
    print(f"Total Reward during test: {total_reward}")


# Main function
if __name__ == "__main__":
    # Hyperparameters
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_layers_1 = [64, 64, 64]  # 3-layer network
    hidden_layers_2 = [128, 128, 128, 128, 128]  # 5-layer network
    learning_rate = 0.001
    gamma = 0.99
    batch_size = 64
    max_episodes = 500
    replay_capacity = 10000
    epsilon_decay = 0.995

    # Initialize networks and optimizer
    q_network = DQNetwork(state_dim, action_dim, hidden_layers_1)
    target_network = DQNetwork(state_dim, action_dim, hidden_layers_1)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    # Experience Replay Buffer
    replay_buffer = ExperienceReplay(replay_capacity)

    # Train and evaluate
    rewards, losses = train_agent(env, q_network, target_network, replay_buffer, optimizer, batch_size, gamma, epsilon_decay, max_episodes)

    # Test the trained agent
    test_agent(env, q_network, render=True)

    # Plot results
    plt.figure()
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    plt.figure()
    plt.plot(losses)
    plt.title("Loss per Training Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()
