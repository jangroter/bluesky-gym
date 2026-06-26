import numpy as np
import pygame

import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

from core.observations import (
    OwnAltitudeObservation, TargetAltitudeObservation,
    RunwayDistanceObservation, IntruderObservation,
)


DISTANCE_MARGIN = 5 # km
NM2KM = 1.852

INTRUSION_PENALTY = -50
ALT_DIF_REWARD_SCALE = -5/3000
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -50/3000

NUM_INTRUDERS = 3
INTRUSION_DISTANCE = 5 # NM
VERTICAL_MARGIN = 1000 * 0.3048 # ft

# Define constants
ALT_MEAN = 1500
ALT_STD = 3000
VZ_MEAN = 0
VZ_STD = 5
RWY_DIS_MEAN = 100
RWY_DIS_STD = 200
DEFAULT_RWY_DIS = 200 
RWY_LAT = 52
RWY_LON = 4

ACTION_2_MS = 12.5  # approx 2500 ft/min

ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

AC_SPD = 150

ACTION_FREQUENCY = 30

NUM_INTRUDERS = 5

class VerticalCREnv(gym.Env):
    """ 
    Vertical CR environment, aircraft needs to descend to the runway while avoiding intruders.
    Fixed limit on the resolution manouevres. 

    TODO:
    * Look at changing action space to vertical speed and altitude(change)
    * Change intruder generation logic -> more focused around descent area and target altitude
    * Improve visualization
    """

    # information regarding the possible rendering modes of the environment
    # for BlueSkyGym probably only implement 1 for now together with None, which is default
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 256
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.altitude_obs = OwnAltitudeObservation(alt_mean=ALT_MEAN, alt_std=ALT_STD, vz_mean=VZ_MEAN, vz_std=VZ_STD)
        self.target_alt_obs = TargetAltitudeObservation(alt_mean=ALT_MEAN, alt_std=ALT_STD)
        self.runway_obs = RunwayDistanceObservation(
            rwy_lat=RWY_LAT, rwy_lon=RWY_LON,
            default_distance=DEFAULT_RWY_DIS, dist_mean=RWY_DIS_MEAN, dist_std=RWY_DIS_STD,
        )
        self.intruder_obs = IntruderObservation(
            n=NUM_INTRUDERS, sort_by="distance", include_vertical=True, alt_norm=ALT_STD,
        )

        self.observation_space = spaces.Dict({
            **self.altitude_obs.space(),
            **self.target_alt_obs.space(),
            **self.runway_obs.space(),
            **self.intruder_obs.space(),
        })
       
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        self.agent = "KL001"

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # set correct sim speed
        bs.stack.stack('DT 1;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.total_intrusions = 0
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
        ac_idx = bs.traf.id2idx(self.agent)

        # raw values retained for _get_reward, _render_frame (approach C)
        self.altitude = bs.traf.alt[ac_idx]
        self.vz = bs.traf.vs[ac_idx]
        self.runway_distance = DEFAULT_RWY_DIS - bs.tools.geo.kwikdist(
            RWY_LAT, RWY_LON, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]) * NM2KM

        # intruder render data in creation order (approach C)
        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        ac_hdg = bs.traf.hdg[ac_idx]
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            self.intruder_distance.append(int_dis * NM2KM)
            bearing = fn.bound_angle_positive_negative_180(ac_hdg - int_qdr)
            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

        return {
            **self.altitude_obs.observe(self.agent),
            **self.target_alt_obs.observe(self.target_alt),
            **self.runway_obs.observe(self.agent),
            **self.intruder_obs.observe(self.agent),
        }
    
    def _generate_conflicts(self):
        target_idx = bs.traf.id2idx(self.agent)
        altitude = bs.traf.alt[target_idx]
        spd = bs.traf.gs[target_idx]
        for i in range(NUM_INTRUDERS):
            dpsi = np.random.randint(45,315)
            cpa = np.random.randint(0,INTRUSION_DISTANCE)
            tlosh = np.random.randint(100,int((DEFAULT_RWY_DIS*0.9)*1000/spd))
            average_tod = (DEFAULT_RWY_DIS*1000/spd) - 2*self.target_alt/ACTION_2_MS
            if tlosh > average_tod:
                dH = np.random.randint(int(-altitude + 500),int((self.target_alt - altitude) + 100))
            else:
                dH = np.random.randint(int((self.target_alt - altitude) - 500),int((self.target_alt - altitude) + 500))
            tlosv = 100000000000.

            bs.traf.creconfs(acid=f'{i}',actype="A320",targetidx=target_idx,dpsi=dpsi,dcpa=cpa,tlosh=tlosh,dH=dH,tlosv=tlosv)
            bs.traf.alt[i+1] = bs.traf.alt[target_idx] + dH
            bs.traf.ap.selaltcmd(i+1, bs.traf.alt[target_idx] + dH, 0)
            

    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "total_reward": self.total_reward,
            "total_intrusions": self.total_intrusions,
            "final_altitude": self.final_altitude
        }
    
    def _get_reward(self):
        int_penalty = self._check_intrusion()
        done = 0
        if self.runway_distance > 0 and self.altitude > 0:
            alt_penalty = abs(self.target_alt - self.altitude) * ALT_DIF_REWARD_SCALE
        elif self.altitude <= 0:
            alt_penalty = CRASH_PENALTY
            self.final_altitude = -100
            done = 1
        elif self.runway_distance <= 0:
            alt_penalty = self.altitude * RWY_ALT_DIF_REWARD_SCALE
            self.final_altitude = self.altitude
            done = 1
        reward = alt_penalty + int_penalty
        self.total_reward += reward
        return reward, done

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx(self.agent)
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            vert_dis = bs.traf.alt[ac_idx] - bs.traf.alt[int_idx]
            if int_dis < INTRUSION_DISTANCE and abs(vert_dis) < VERTICAL_MARGIN:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        return reward
        
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
        bs.traf.reset()
        # reset episodic logging variables
        self.total_reward = 0
        self.total_intrusions = 0
        self.final_altitude = 0

        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF,TARGET_ALT_DIF)

        bs.traf.cre(self.agent, actype="A320", acalt=alt_init, acspd=AC_SPD)
        bs.traf.swvnav[0] = False

        self._generate_conflicts()

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

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_alt = int((-1*(bs.traf.alt[int_idx]-max_alt)/max_alt)*(self.window_height-50))
            int_x_dis = self.intruder_distance[int_idx - 1] * self.cos_bearing[int_idx - 1]
            int_y_dis = self.intruder_distance[int_idx - 1] * self.sin_bearing[int_idx - 1]
            width_temp = int(5+int_y_dis/20)
            aircraft_start = int(((zero_offset + int_x_dis )/max_distance)*self.window_width)
            aircraft_end = int(aircraft_start + (4/max_distance)*self.window_width)
            color = (255,255,255) if abs(int_y_dis) > DISTANCE_MARGIN else 'red'

            pygame.draw.line(
                canvas,
                color,
                (aircraft_start,int_alt),
                (aircraft_end,int_alt),
                width = width_temp
            )

            hor_margin = (DISTANCE_MARGIN*NM2KM/max_distance)*self.window_width
            ver_margin = (VERTICAL_MARGIN/max_alt)*self.window_height

            pygame.draw.line(
                canvas,
                'black',
                (aircraft_start-hor_margin/2,int_alt-ver_margin),
                (aircraft_end+hor_margin/2,int_alt-ver_margin),
                width = 1
            )
            pygame.draw.line(
                canvas,
                'black',
                (aircraft_start-hor_margin/2,int_alt+ver_margin),
                (aircraft_end+hor_margin/2,int_alt+ver_margin),
                width = 1
            )
            pygame.draw.line(
                canvas,
                'black',
                (aircraft_start-hor_margin/2,int_alt-ver_margin),
                (aircraft_start-hor_margin/2,int_alt+ver_margin),
                width = 1
            )
            pygame.draw.line(
                canvas,
                'black',
                (aircraft_end+hor_margin/2,int_alt-ver_margin),
                (aircraft_end+hor_margin/2,int_alt+ver_margin),
                width = 1
            )



        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        bs.stack.stack('quit')