import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import gymnasium as gym
from gymnasium import spaces

# Define constants
ALT_MEAN = 1500
ALT_STD = 3000
VZ_MEAN = 0
VZ_STD = 5
RWY_DIS_MEAN = 100
RWY_DIS_STD = 200

ACTION_2_MS = 12.5

ALT_DIF_REWARD_SCALE = -5/3000
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -50/3000

ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

AC_SPD = 150

ACTION_FREQUENCY = 30

class DescentEnv(gym.Env):
    """ 
    Very simple environment that requires the agent to climb / descend to a target altitude.
    As the runway approaches the aircraft has to start descending, knowing when to start
    the descent.

    TODO:
    - better commenting
    - proper normalization functionality
    - Monitor Wrapper class for monitoring progress, can be something to be used by all envs.
    """

    # information regarding the possible rendering modes of the environment
    # for BlueSkyGym probably only implement 1 for now together with None, which is default
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 256
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.observation_space = spaces.Dict(
            {
                "altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "target_altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "runway_distance": spaces.Box(-np.inf, np.inf, dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.final_altitude = 0

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        """
        Observation consists of altitude, vertical speed, target altitude and distance to runway
        Very crude normalization in place for now
        """

        DEFAULT_RWY_DIS = 200 
        RWY_LAT = 52
        RWY_LON = 4
        NM2KM = 1.852

        self.altitude = bs.traf.alt[0]
        self.vz = bs.traf.vs[0]
        self.runway_distance = (DEFAULT_RWY_DIS - bs.tools.geo.kwikdist(RWY_LAT,RWY_LON,bs.traf.lat[0],bs.traf.lon[0])*NM2KM)

        # very crude normalization
        obs_altitude = np.array([(self.altitude - ALT_MEAN)/ALT_STD])
        obs_vz = np.array([(self.vz - VZ_MEAN) / VZ_STD])
        obs_target_alt = np.array([((self.target_alt- ALT_MEAN)/ALT_STD)])
        obs_runway_distance = np.array([(self.runway_distance - RWY_DIS_MEAN)/RWY_DIS_STD])

        observation = {
                "altitude": obs_altitude,
                "vz": obs_vz,
                "target_altitude": obs_target_alt,
                "runway_distance": obs_runway_distance,
            }
        
        return observation
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "total_reward": self.total_reward,
            "final_altitude": self.final_altitude
        }
    
    def _get_reward(self):

        # reward part of the function
        if self.runway_distance > 0 and self.altitude > 0:
            reward = abs(self.target_alt - self.altitude) * ALT_DIF_REWARD_SCALE
            self.total_reward += reward
            return reward, 0
        elif self.altitude <= 0:
            reward = CRASH_PENALTY
            self.final_altitude = -100
            self.total_reward += reward
            return reward, 1
        elif self.runway_distance <= 0:
            reward = self.altitude * RWY_ALT_DIF_REWARD_SCALE
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, 1
        
    def _get_action(self,action):
        # Transform action to the meters per second
        action = action * ACTION_2_MS

        # Bluesky interpretes vertical velocity command through altitude commands 
        # with a vertical speed (magnitude). So check sign of action and give arbitrary 
        # altitude command

        # The actions are then executed through stack commands;
        if action >= 0:
            bs.traf.selalt[0] = 1000000 # High target altitude to start climb
            bs.traf.selvs[0] = action
        elif action < 0:
            bs.traf.selalt[0] = 0 # High target altitude to start descent
            bs.traf.selvs[0] = action

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        
        # reset episodic logging variables
        self.total_reward = 0
        self.final_altitude = 0

        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF,TARGET_ALT_DIF)

        bs.traf.cre('KL001',actype="A320",acalt=alt_init,acspd=AC_SPD)
        bs.traf.swvnav[0] = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        
        self._get_action(action)

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
                observation = self._get_obs()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info
    
    def render(self):
        pass

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        zero_offset = 25
        max_distance = 180 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        # draw a ground surface
        pygame.draw.rect(
            canvas, 
            (154,205,50),
            pygame.Rect(
                (0,self.window_height-50),
                (self.window_width, 50)
                ),
        )
        
        # draw target altitude
        max_alt = 5000
        target_alt = int((-1*(self.target_alt-max_alt)/max_alt)*(self.window_height-50))

        pygame.draw.line(
            canvas,
            (255,255,255),
            (0,target_alt),
            (self.window_width,target_alt)
        )

        # draw runway
        runway_length = 30
        runway_start = int(((self.runway_distance + zero_offset)/max_distance)*self.window_width)
        runway_end = int(runway_start + (runway_length/max_distance)*self.window_width)

        pygame.draw.line(
            canvas,
            (119,136,153),
            (runway_start,self.window_height - 50),
            (runway_end,self.window_height - 50),
            width = 3
        )

        # draw aircraft
        aircraft_alt = int((-1*(self.altitude-max_alt)/max_alt)*(self.window_height-50))
        aircraft_start = int(((zero_offset)/max_distance)*self.window_width)
        aircraft_end = int(aircraft_start + (4/max_distance)*self.window_width)

        pygame.draw.line(
            canvas,
            (0,0,0),
            (aircraft_start,aircraft_alt),
            (aircraft_end,aircraft_alt),
            width = 5
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass