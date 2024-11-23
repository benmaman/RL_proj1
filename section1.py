import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm

# Initialize the FrozenLake environment- we use v1 instead v0 because v0 is deprecated
env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize a look-up table with zeros
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.9            # Learning rate
gamma = 0.99           # Discount factor
epsilon = 1.0          # Initial epsilon for exploration
epsilon_decay = 0.995  # Decay rate for epsilon
epsilon_min = 0.1      # Minimum epsilon
n_episodes = 5000      # Total episodes
max_steps = 100        # Max steps per episode

# Tracking metrics
rewards_per_episode = []
steps_to_goal = []

# Training the agent
for episode in tqdm(range(n_episodes)):
    state = env.reset()[0]
    total_reward = 0
    steps = 0

    for _ in range(max_steps):
        # Choose action using epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state, :])    # Exploit
        
        # Take action and observe results
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-value using Q-learning formula
        best_next_action = np.argmax(Q[next_state, :])
        Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Track performance
    rewards_per_episode.append(total_reward)
    steps_to_goal.append(steps if total_reward > 0 else max_steps)

# Post-training evaluation
def plot_q_table(q_table, title):
    """Plot Q-value table as a colormap."""
    plt.figure(figsize=(8, 6))
    cmap = colors.ListedColormap(["white", "lightblue", "blue", "darkblue"])
    plt.imshow(q_table, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_performance(rewards, steps):
    """Plot reward and steps to goal metrics."""
    # Plot rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    # Plot steps to goal
    avg_steps = [np.mean(steps[i:i + 100]) for i in range(0, len(steps), 100)]
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(steps), 100), avg_steps)
    plt.title("Average Steps to Goal (per 100 Episodes)")
    plt.xlabel("Episodes (x100)")
    plt.ylabel("Average Steps")
    plt.tight_layout()
    plt.show()

# Plots
plot_performance(rewards_per_episode, steps_to_goal)

# Colormap of Q-values
plot_q_table(Q, "Final Q-Value Table")
