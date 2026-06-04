import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# data_sac = pd.read_csv('logs/PathPlanningEnv-v0/PathPlanningEnv-v0_SAC.csv')
# data_ppo = pd.read_csv('logs/PathPlanningEnv-v0/PathPlanningEnv-v0_PPO.csv')

# data_sac['mov'] = data_sac['total_reward'].rolling(window=1000).mean()
# data_ppo['mov'] = data_ppo['total_reward'].rolling(window=1000).mean()

# sns.lineplot(data_sac['mov'])
# sns.lineplot(data_ppo['mov'])
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
data_sac = pd.read_csv('logs/PathPlanningEnv-v0/PathPlanningEnv-v0_SAC.csv')

# Define window size
window = 1000

# Compute rolling quantiles and mean
data_sac['mov_mean'] = data_sac['total_reward'].rolling(window=window).mean()
data_sac['q1'] = data_sac['total_reward'].rolling(window=window).quantile(0.25)
data_sac['q3'] = data_sac['total_reward'].rolling(window=window).quantile(0.75)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(x=data_sac.index, y='mov_mean', data=data_sac, ax=ax, color='tab:blue')
plt.fill_between(data_sac.index, data_sac['q1'], data_sac['q3'], color='tab:blue', alpha=0.2)
ax.set_xlabel('Episode')
ax.set_ylabel('Average return')
# Create inset zoom
axins = inset_axes(ax, width="50%", height="50%", loc=('center'))
sns.lineplot(x=data_sac.index, y='mov_mean', data=data_sac, ax=axins, color='tab:blue')
axins.fill_between(data_sac.index, data_sac['q1'], data_sac['q3'], color='tab:blue', alpha=0.2)

# Zoom to later part
axins.set_xlim(20000, 233000)
axins.set_xlabel('')
axins.set_ylabel('')
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_ylim(data_sac['mov_mean'][20000:233000].min()*0.95,
               data_sac['mov_mean'][20000:233000].max()*1.05)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.show()