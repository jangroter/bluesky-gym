# Wrappers
Wrappers are a tool to extend the functionality of an environment, and should ideally be environment independent.
Usage of a wrapper requires initialization of the environment, and then `wrapping' this initialized environment.

See the following example for creating a simple 'noisy observation' environment

```python
import gymnasium as gym
import bluesky_gym
from bluesky_gym.wrappers.wind import NoisyObservationWrapper
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')
noisy_env = NoisyObservationWrapper(env, noise_level=0.2)
```

In theory wrappers can overwrite or add functionality to all of the core functions of the environments.
Below is a more complex example utilizing the windfield to create a static windfield with 4 sources of wind and no altitude gradient:

```python
import gymnasium as gym
import bluesky_gym
from bluesky_gym.wrappers.wind import WindFieldWrapper
import numpy as np
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')

# Define the coordinates of the wind
lat = np.array([51.9,51.9,52.1,52.1])
lon = np.array([3.9,4.1,3.9,4.1])
vnorth = np.array([[16, 12, 14, 15]]) # 2D because one dimension is used of altitude, which we leave at None
veast = np.array([[3, 7, 9, 4]])

windy_env = WindFieldWrapper(env, lat=lat, lon=lon, vnorth=vnorth, veast=veast, augment_obs=True)
```
