import gymnasium as gym
from gymnasium.spaces import Dict, Box
import bluesky as bs
import numpy as np

MAX_WIND = 50 # Crude value used for normalization of the wind values

class WindFieldWrapper(gym.Wrapper):
    def __init__(self, env, lat, lon, vnorth, veast, alt=None, augment_obs=False):
        super().__init__(env)
        self.lat = lat
        self.lon = lon
        self.vnorth = vnorth
        self.veast = veast
        self.alt = alt
        self.augment_obs = augment_obs

        if self.augment_obs:
            assert isinstance(self.observation_space, Dict), "This wrapper only supports Dict observation spaces."
            self.observation_space = Dict({
                **self.observation_space.spaces,  
                "wind_u": Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "wind_v": Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            })

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        bs.traf.wind.addpointvne(self.lat, self.lon, self.vnorth, self.veast, self.alt)

        if self.augment_obs: 
            wind_u, wind_v = self._get_wind_observation()
            wind = {
                "wind_u": np.array([wind_u]),
                "wind_v": np.array([wind_v])
            }
            observation = {**observation, **wind} 
            # import code
            # code.interact(local=locals())

        return observation, info
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)

        if self.augment_obs: 
            wind_u, wind_v = self._get_wind_observation()
            wind = {
                "wind_u": np.array([wind_u]),
                "wind_v": np.array([wind_v])
            }
            observation = {**observation, **wind}

        return observation, reward, done, truncated, info
    
    def _get_wind_observation(self):
        acidx = bs.traf.id2idx('kl001')
        lat, lon, alt = bs.traf.lat[acidx], bs.traf.lon[acidx], bs.traf.alt[acidx]
        wind_n, wind_e = bs.traf.wind.getdata(lat,lon,alt)

        hdg = np.deg2rad(bs.traf.hdg[acidx])

        wind_u = (wind_n * np.cos(hdg) + wind_e * np.sin(hdg)) / MAX_WIND
        wind_v = (-wind_n * np.sin(hdg) + wind_e * np.cos(hdg)) / MAX_WIND

        return wind_u, wind_v 