import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

reward = pd.read_csv('sac_merge_v2/weights_2/reward.csv')
reward['mov'] = reward['# reward'].rolling(window=100).mean()
sns.lineplot(reward['mov'])
plt.show()