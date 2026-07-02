import numpy as np
import bluesky as bs

import gymnasium as gym
from gymnasium import spaces

from core.observations import OwnAltitudeObservation, TargetAltitudeObservation, RunwayDistanceObservation
from core.rendering import (
    PygameCanvas, SideProfileProjection,
    draw_side_aircraft, draw_ground, draw_runway, draw_target_altitude,
)
from core.actions import VerticalSpeedAction

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

        self.alt_obs = OwnAltitudeObservation()
        self.target_alt_obs = TargetAltitudeObservation()
        self.runway_obs = RunwayDistanceObservation(rwy_lat=52, rwy_lon=4)

        self.observation_space = spaces.Dict({
            **self.alt_obs.space(),
            **self.target_alt_obs.space(),
            **self.runway_obs.space(),
        })

        self.agent = "KL001"

        self.vertical_action = VerticalSpeedAction(vs_scale=ACTION_2_MS)
        self.action_space = self.vertical_action.space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # set correct sim speed
        bs.stack.stack('DT 1;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.final_altitude = 0

        self.pygame_canvas = PygameCanvas(self.window_width, self.window_height)
        self.projection = SideProfileProjection(
            max_distance=180, max_altitude=5000,
            window_size=(self.window_width, self.window_height),
        )


    def _get_obs(self):
        return {
            **self.alt_obs.observe(self.agent),
            **self.target_alt_obs.observe(self.target_alt),
            **self.runway_obs.observe(self.agent),
        }
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "total_reward": self.total_reward,
            "final_altitude": self.final_altitude
        }
    
    def _get_reward(self):
        self.altitude = bs.traf.alt[0]
        self.runway_distance = (
            200 - bs.tools.geo.kwikdist(52, 4, bs.traf.lat[0], bs.traf.lon[0]) * 1.852
        )
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
        
    def _get_action(self, action):
        self.vertical_action.execute(self.agent, action)

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        
        # reset episodic logging variables
        self.total_reward = 0
        self.final_altitude = 0

        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF,TARGET_ALT_DIF)

        bs.traf.cre(self.agent,actype="A320",acalt=alt_init,acspd=AC_SPD)
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
        self.altitude = bs.traf.alt[0]
        self.runway_distance = (
            200 - bs.tools.geo.kwikdist(52, 4, bs.traf.lat[0], bs.traf.lon[0]) * 1.852
        )
        canvas = self.pygame_canvas.begin_frame()

        draw_ground(canvas, self.projection)

        _, target_y = self.projection.project(0, self.target_alt)
        draw_target_altitude(canvas, target_y, self.projection)

        rwy_x, rwy_y = self.projection.project(self.runway_distance, 0)
        draw_runway(canvas, rwy_x, rwy_y, self.projection.scale_horizontal(30))

        ac_x, ac_y = self.projection.project(0, self.altitude)
        draw_side_aircraft(canvas, ac_x, ac_y, self.projection.scale_horizontal(4))

        self.pygame_canvas.end_frame(canvas)
        
    def close(self):
        bs.stack.stack('quit')