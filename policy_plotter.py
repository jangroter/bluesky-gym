
import functions as fn
import areafilter as af
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.cm as cm
import pandas as pd

from stable_baselines3 import PPO, SAC, TD3, DDPG

model = SAC.load(f"saved_models/model_50north2")

pop_array = np.genfromtxt('bluesky_gym/envs/data/population_1km.csv',delimiter=' ')

x_index_min = int(((500000)/1000)-300)
x_index_max = int(((500000)/1000)+300)
y_index_min = int(((500000)/1000)-300)
y_index_max = int(((500000)/1000)+300)

x_array = np.genfromtxt('bluesky_gym/envs/data/x_array.csv',delimiter=' ')
y_array = np.genfromtxt('bluesky_gym/envs/data/y_array.csv',delimiter=' ')

pop_array = pop_array[y_index_min:y_index_max,x_index_min:x_index_max]

# actor = agent.Agent(modelname="E_3_256-x1/model/actor.pt")

circle_lat = 52.3322
circle_lon = 4.75

lat_0 = 0
lon_0 = 0

lat_1 = 0
lon_1 = 0

bearing = np.arange(0,360,1)
distance = np.random.uniform(250,250,360)

# bearing = np.random.randint(0,360,720)
# distance = np.sqrt(np.random.random(720))*275

# value = agent.Value()

# x0 = np.arange(-1, 1, 0.005)
# y0 = np.arange(-1, 1, 0.005)
# x,y = np.meshgrid(x0,y0)

# index = np.arange(len(x0))

# z = np.zeros(np.shape(x))

# for i in index:
#     for j in index:
#         state = [x0[i],y0[j]]
#         z[i,j] = value.step(state)[0].detach().cpu().numpy() * -1

c_map = cm.get_cmap("Blues").copy()
c_map.set_bad('w')

# Polderbaan 30,160,20
# East 20,60,-60
line = af.poly_arc2()
fig, ax = plt.subplots(figsize=(15, 15))
#ax.imshow(pop_array,extent=[np.min(x_array),np.max(x_array),np.min(y_array),np.max(y_array)],norm=LogNorm(vmin=1000,vmax=100000))
# ax.imshow(np.rot90(z),extent=[-277800,277800,-277800,277800], cmap=c_map)
ax.imshow(pop_array,cmap=c_map,extent=[-300000,300000,-300000,300000],norm=LogNorm(vmin=100,vmax=100000))

# c_map = cm.get_cmap("Reds").copy()
# c_map.set_bad('w')
# ax.imshow(pop_array[0:300,300:],cmap=c_map,extent=[0,300000,0,300000],norm=LogNorm(vmin=100,vmax=100000))

for i,j in zip(bearing,distance):
    # for j in distance:
    make_line = True
    count = 0

    lat = np.array([])
    lon = np.array([])

    x = np.array([])
    y = np.array([])
    alpha = 0.1
    color = 'k'
    while make_line:
        if count == 0:
            lat_0, lon_0 = fn.get_spawn(i, radius = j)
            lat = np.append(lat,lat_0)
            lon = np.append(lon,lon_0)
            x_, y_ = fn.get_xy(lat_0,lon_0)
            x = np.append(x,x_)
            y = np.append(y,y_)

        state = fn.get_state(lat_0, lon_0)
        action, _states = model.predict(state, deterministic=True)
        lat_1, lon_1 = np.rad2deg(fn.do_action(action,lat_0,lon_0))
        lat = np.append(lat,lat_1)
        lon = np.append(lon,lon_1)
        x_, y_ = fn.get_xy(lat_1,lon_1)
        x = np.append(x,x_)
        y = np.append(y,y_)
        if count > 100:
            make_line = False
        if af.checkIntersect(line,lat_0,lon_0,lat_1,lon_1):
            make_line = False
            color = 'r'
            alpha = 1
        
        count += 1
        lat_0 = lat_1
        lon_0 = lon_1
    ax.plot(x,y,color,alpha=alpha)

fig.savefig('test2.png')
plt.close(fig)