import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

window= 1000

data1 = pd.read_csv("sac_cr_hrl/weights_final/reward.csv")
data2 = pd.read_csv("sac_cr_hrl/weights_small_network/reward.csv")
data3 = pd.read_csv("sac_cr_hrl/weights_no_silu/reward.csv")
data4 = pd.read_csv("sac_cr_hrl/weights_no_per/reward.csv")
data5 = pd.read_csv("sac_cr_hrl/weights_base/reward.csv")

fig, ax = plt.subplots(figsize=(10,6))

data_arr = [data1,data2,data3,data4,data5]
colors = ['blue','orange','red','green','black']
legends = ['Full Model','Small Network (256x256)','ReLU activation','No PER','Default']

for data, color,legend, in zip(data_arr,colors,legends):
    data['smoothed_reward'] = data['# reward'].rolling(window=window, center=True).mean()
    data['q1'] = data['# reward'].rolling(window=window).quantile(0.25)
    data['q3'] = data['# reward'].rolling(window=window).quantile(0.75)

    plot_data = data.dropna(subset=['smoothed_reward'])
    plot_data = plot_data.reset_index()

    sns.lineplot(x=plot_data.index, 
                 y='smoothed_reward', 
                 data=plot_data, 
                 ax=ax, 
                 color=color, 
                 label=legend)
    # plt.fill_between(data.index, data['q1'], data['q3'], color=color, alpha=0.2)
plt.legend()
plt.ylabel('Reward (average of 1000)')
plt.xlabel('Episode')

# data1['smoothed_reward'] = data1['# reward'].rolling(window=window_size, center=True).mean()
# data2['smoothed_reward'] = data2['# reward'].rolling(window=window_size, center=True).mean()
# data3['smoothed_reward'] = data3['# reward'].rolling(window=window_size, center=True).mean()
# data4['smoothed_reward'] = data4['# reward'].rolling(window=window_size, center=True).mean()

# plt.plot(data1['smoothed_reward'])
# plt.plot(data2['smoothed_reward'])
# plt.plot(data3['smoothed_reward'])
# plt.plot(data4['smoothed_reward'])

plt.show()

