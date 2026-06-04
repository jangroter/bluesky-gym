import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

window = 500
# Load data
MAsep = pd.read_csv("sac_merge_v2/weights/reward.csv")
MAsep['smoothed_reward'] = MAsep['# reward'].rolling(window=window, center=True).mean()
MAsep['q1'] = MAsep['# reward'].rolling(window=window).quantile(0.25)
MAsep['q3'] = MAsep['# reward'].rolling(window=window).quantile(0.75)


# SAsep = pd.read_csv('logs/SectorCREnv-v0/SectorCREnv-v0_SAC.csv')
# SAsep['smoothed_reward'] = SAsep['total_reward'].rolling(window=window, center=True).mean()
# print(SAsep.__len__())
# SAsep['q1'] = SAsep['total_reward'].rolling(window=window).quantile(0.25)
# SAsep['q3'] = SAsep['total_reward'].rolling(window=window).quantile(0.75)



from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(x=MAsep.index, y='smoothed_reward', data=MAsep, ax=ax, color='tab:blue')
plt.fill_between(MAsep.index, MAsep['q1'], MAsep['q3'], color='tab:blue', alpha=0.2)
# sns.lineplot(x=SAsep.index, y='smoothed_reward', data=SAsep, ax=ax, color='tab:orange')
# plt.fill_between(SAsep.index, SAsep['q1'], SAsep['q3'], color='tab:orange', alpha=0.2)
ax.set_xlabel('Episode')
ax.set_ylabel('Average return')

axins = inset_axes(ax, width="50%", height="50%", loc=('center'))
sns.lineplot(x=MAsep.index, y='smoothed_reward', data=MAsep, ax=axins, color='tab:blue')
axins.fill_between(MAsep.index, MAsep['q1'], MAsep['q3'], color='tab:blue', alpha=0.2)

# Zoom to later part
axins.set_xlim(5000, 90000)
axins.set_xlabel('')
axins.set_ylabel('')
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_ylim(MAsep['smoothed_reward'][5000:90000].min()*1.05,
               MAsep['q3'][5000:90000].max()*0.95)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()