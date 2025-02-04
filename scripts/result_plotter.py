"""
File used for generating the figures used in the original BlueSky-Gym publication, requires the log-files from the experiments.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def time_steps_per_episode(df):
    temp = np.append(np.array([0]),df['timesteps'])
    time_steps = df['timesteps'] - temp[:-1]
    return time_steps

# Sample data
models = ["PPO", "SAC", "TD3", "DDPG"]
envs = ["DescentEnv-v0","VerticalCREnv-v0","PlanWaypointEnv-v0","HorizontalCREnv-v0","SectorCREnv-v0","StaticObstacleEnv-v0","MergeEnv-v0"]

# Plotting
ave_window = 1000
feature = 'total_reward'
sns.set_theme(style="darkgrid")
sns.set_context("talk")
fig, axs = plt.subplots(2, 4, figsize=(20, 8))

for i, ax in enumerate(axs.flatten()[:-1]):
    env = envs[i]
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([0,1e6,2e6])
    ax.set_xticklabels(['0','1e6','2e6'])
    for model in models:
        y_data = pd.read_csv(f'common/results/logs_backup/{env}/{env}_{model}.csv')
        sns.lineplot(x=y_data['timesteps'][:-(ave_window-1)],y=moving_average((y_data)[feature],ave_window), legend=False ,ax=ax,label=model)
        
    ax.set_title(f'{env}',fontsize = 18)



# Setting the last subplot for the legend
axs[-1, -1].axis('off')  # Turn off the axis for the legend subplot

sns.set_theme(style="white")
# Create an invisible plot for the legend
for model in models:
    axs[-1, -1].plot([], [], label=model)

axs[-1, -1].legend(loc='center', fontsize=24)

plt.tight_layout()
plt.show()