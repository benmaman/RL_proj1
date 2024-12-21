import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
from itertools import  product
import pandas as pd
import datetime 
from PIL import Image
import os 


# Initialize the FrozenLake environment- we use v1 instead v0 because v0 is deprecated
seed_value = 42
np.random.seed(seed_value)  # NumPy random seed

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode='rgb_array')
state, info = env.reset(seed=seed_value)

env.action_space.seed()  # Set the seed for the action space

n_states = env.observation_space.n
n_actions = env.action_space.n


##save a frame

output_dir = "output/section_1"
frame = env.render() # Render the environment as an RGB array
# Convert the frame to an image and save it
img = Image.fromarray(frame)
img.save(os.path.join(output_dir, f"frame_frozen_lake_frames.png"))

# Close the environment
env.close()

print(f"Frames saved in the '{output_dir}' directory.")



results_df = pd.DataFrame(columns=['alpha', 'gamma', 'epsilon', 'epsilon_decay', 
                                   'epsilon_min', 'n_episodes', 'max_steps', 
                                   'total_reward', 'mean_reward', 'final_epsilon'])

HYPERPARAMS_grid = {
    'alpha' : [0.3,0.2,0.1,0.05,0.01,0.005],            # Learning rate
    'gamma' : [0.9999,0.99,0.97,0.95,0.9],          # Discount factor
    'epsilon' : [1.0],         # Initial epsilon for exploration
    'epsilon_decay' :[0.995,0.99,0.9], # Decay rate for epsilon
    'epsilon_min' : [0.01],     # Minimum epsilon
    'n_episodes' : [5000],    # Total episodes
    'max_steps' : [100]        # Max steps per episode
    }

# HYPERPARAMS_grid = {
#     'alpha' : [0.1],            # Learning rate
#     'gamma' : [0.99],          # Discount factor
#     'epsilon' : [1.0],         # Initial epsilon for exploration
#     'epsilon_decay' :[0.9], # Decay rate for epsilon
#     'epsilon_min' : [0.01],     # Minimum epsilon
#     'n_episodes' : [5000],    # Total episodes
#     'max_steps' : [100]        # Max steps per episode
#     }
HYPERPARAMS_grid = {
    'alpha' : [0.3],            # Learning rate
    'gamma' : [0.97],          # Discount factor
    'epsilon' : [1.0],         # Initial epsilon for exploration
    'epsilon_decay' :[ 0.995], # Decay rate for epsilon
    'epsilon_min' : [0.01],     # Minimum epsilon
    'n_episodes' : [5000],    # Total episodes
    'max_steps' : [100]        # Max steps per episode
    }

# Flatten the grid into combinations
hyperparam_combinations = list(product(*HYPERPARAMS_grid.values()))
param_names = list(HYPERPARAMS_grid.keys())

# Store results
results = []

# Training loop for hyperparameter tuning
for params in hyperparam_combinations:

    # Initialize a look-up table with zeros
    Q = np.zeros((n_states, n_actions))

    # Create a hyperparameter dictionary for the current combination
    current_params = dict(zip(param_names, params))
    print(f"Training with parameters: {current_params}")
        
    # Hyperparameters
    alpha = current_params['alpha']         
    gamma = current_params['gamma']             
    epsilon =current_params['epsilon']            
    epsilon_decay =current_params['epsilon_decay']    
    epsilon_min = current_params['epsilon_min']       
    n_episodes = current_params['n_episodes']       
    max_steps = current_params['max_steps']         

    # Tracking metrics
    rewards_per_episode = []
    steps_to_goal = []
    Q_table_storage={}
    total_steps=0
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
            total_steps+=1

            #store the q table according to specific steps
            if total_steps in [499,1999]:
                Q_table_storage[total_steps+1]=Q.copy()

            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Track performance
        rewards_per_episode.append(total_reward)
        steps_to_goal.append(steps if total_reward > 0 else max_steps)

    #store the last Q table 
    Q_table_storage[total_steps+1]=Q.copy()
    
    #store the result of each hyper-params
    total_reward_agent=np.sum(rewards_per_episode)
    mean_reward_agent=np.mean(rewards_per_episode)
    last_100_episodes_reward=np.mean(rewards_per_episode[-100:])
    results_df = results_df.append({
    'alpha': alpha,
    'gamma': gamma,
    'epsilon': epsilon,
    'epsilon_decay': epsilon_decay,
    'epsilon_min': epsilon_min,
    'n_episodes': n_episodes,
    'max_steps': max_steps,
    'total_reward': total_reward_agent,
    'mean_reward': mean_reward_agent,
    'final_epsilon': epsilon * (epsilon_decay ** n_episodes)  # Epsilon at the end of training
    ,'last_100_episodes_reward':last_100_episodes_reward
    }, ignore_index=True)

current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%m-%d %H:%M:%S").replace(":","_")
results_df.to_csv(f"output/section_1/summary_results_hp_{formatted_datetime}.csv")

# Post-training evaluation
def plot_q_tables_combined(q_table_storage):
    """Plot multiple Q-value tables with a classic white background and soft colors."""
    num_tables = len(q_table_storage)
    fig, axes = plt.subplots(1, num_tables, figsize=(15, 5), facecolor='white')
    action_labels = ['0: LEFT', '1: DOWN', '2: RIGHT', '3: UP']

    for ax, (title, q_table) in zip(axes, q_table_storage.items()):
        im = ax.imshow(q_table, cmap='Blues', interpolation="nearest",)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Actions", fontsize=10)
        ax.set_ylabel("States", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xticks(range(q_table.shape[1]))  # Actions
        ax.set_yticks(range(q_table.shape[0]))  # States
        ax.set_xticklabels(action_labels, fontsize=7,rotation=45)

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
