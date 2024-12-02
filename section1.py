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
epsilon_decay = 0.999  # Decay rate for epsilon
epsilon_min = 0.1      # Minimum epsilon
n_episodes = 5000      # Total episodes
max_steps = 100        # Max steps per episode

# Tracking metrics
rewards_per_episode = []
steps_to_goal = []
Q_table_storage={}

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

    #store the Q table for 500,2000 and 5000
    if episode in [499,1999,4999]:
        Q_table_storage[episode+1]=Q.copy()

# Post-training evaluation
def plot_q_tables_combined(q_table_storage):
    """Plot multiple Q-value tables with a classic white background and soft colors."""
    num_tables = len(q_table_storage)
    fig, axes = plt.subplots(1, num_tables, figsize=(15, 5), facecolor='white')
    
    for ax, (title, q_table) in zip(axes, q_table_storage.items()):
        im = ax.imshow(q_table, cmap='Blues', interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Actions", fontsize=10)
        ax.set_ylabel("States", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xticks(range(q_table.shape[1]))  # Actions
        ax.set_yticks(range(q_table.shape[0]))  # States
        
    # Add a single colorbar for all subplots
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.7, label="Q-value")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Q-value", fontsize=10)
    plt.savefig("output/section_1/q_table_heatmap.png")

def plot_performance(rewards, steps):
    """Plot reward and steps to goal metrics."""
    # Plot rewards
    plt.figure(figsize=(12, 5))
    plt.plot(rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("output/section_1/reward_vs_episode.png")

    # Plot steps to goal
    avg_steps = [np.sum(rewards[i:i + 100]) for i in range(0, len(rewards)-100)]
    plt.figure(figsize=(12, 5))
    plt.plot(range(0, len(avg_steps)), avg_steps)
    plt.title("Average reward for moving 100 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.tight_layout()
    plt.savefig("output/section_1/average_reward_vs_episode.png")


# Plots
plot_performance(rewards_per_episode, steps_to_goal)

# Colormap of Q-values
plot_q_tables_combined( Q_table_storage)
