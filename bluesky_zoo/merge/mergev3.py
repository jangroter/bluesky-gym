"""
This table enumerates the observation space for 3 neigbours in observation space,
if number of neighbours included is changed, indices will change aswell (of course):

| Index: [start, end) | Description                                                  |   Values    |
|:-----------------:|------------------------------------------------------------|:---------------:|
|          0          | cos(track deviation)                                         | [-inf, inf] |
|          1          | sin(track deviation)                                         | [-inf, inf] |
|          2          | airspeed                                                     | [-inf, inf] |
|          3          | final approach fix distance                                  | [-inf, inf] |
|          4          | faf reached boolean                                          |    (0,1)    |
|         5-7         | Relative X postions with 3 closest neighbours                | [-inf, inf] |
|         8-10        | Relative Y postions with 3 closest neighbours                | [-inf, inf] |
|        11-13        | Relative X velocity with 3 closest neighbours                | [-inf, inf] |
|        14-16        | Relative Y velocity with 3 closest neighbours                | [-inf, inf] |
|        17-19        | cos(heading difference) with 3 closest neighbours            | [-inf, inf] |
|        20-22        | sin(heading difference) with 3 closest neighbours            | [-inf, inf] |
|        23-25        | distance with 3 closest neighbours                           | [-inf, inf] |

"""

import functools
from pettingzoo import ParallelEnv
import gymnasium as gym
import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import bluesky_gym.envs.common.functions as fn
import random

DISTANCE_MARGIN = 20 # km
REACH_REWARD = 0#1

# Model parameters
ACTION_FREQUENCY = 5
NUM_AC_STATE = 3
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1

INTRUSION_DISTANCE = 5 # NM

SPAWN_DISTANCE_MIN = 15
SPAWN_DISTANCE_MAX = 200

MERGE_ANGLE_MIN = 5
MERGE_ANGLE_MAX = 45

D_HEADING = 22.5 # deg
D_VELOCITY = 20/3 # kts

ALTITUDE = 350 # In FL

# Aircraft parameters
AC_SPD = 150
AC_TYPE = "A320"

NM2KM = 1.852
MpS2Kt = 1.94384

RWY_LAT = 52.36239301495972
RWY_LON = 4.713195734579777

distance_faf_rwy = 200 # NM
bearing_faf_rwy = 0
FIX_LAT, FIX_LON = fn.get_point_at_distance(RWY_LAT, RWY_LON, distance_faf_rwy, bearing_faf_rwy)

CENTER = np.array([51.990426702297746, 4.376124857109851]) # TU Delft AE Faculty coordinates

class MergeEnv(ParallelEnv):
    metadata = {
        "name": "merge_v0",
        "render_modes": ["rgb_array","human"],
         "render_fps": 120}

    def __init__(self, render_mode=None, n_agents=10, time_limit=150):
        self.window_width = 750
        self.window_height = 500
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.num_ac = n_agents
        self.agents = self._get_agents(self.num_ac)
        self.possible_agents = self.agents[:]

        self.time_limit = time_limit
        self.steps = 0

        self.observation_spaces = {agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3+7*NUM_AC_STATE,), dtype=np.float64) for agent in self.agents}
        self.action_spaces = {agent: gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64) for agent in self.agents}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.reward_array = np.array([])
        self.num_episodes = 0
        self.average_drift = np.array([])
        self.total_intrusions = 0
        self.faf_reached = 0

        self.window = None
        self.clock = None

        self.wpt_reach = {a: 0 for a in self.agents}
        self.wpt_lat = FIX_LAT
        self.wpt_lon = FIX_LON
        self.rwy_lat = RWY_LAT
        self.rwy_lon = RWY_LON

    def reset(self, seed=None, options=None):
        
        bs.traf.reset()
        self.steps = 0
        self.agents = self.possible_agents[:]
        self.wpt_reach = {a: 0 for a in self.agents}

        self.total_reward = 0
        self.average_drift = np.array([])
        self.total_intrusions = 0
        self.faf_reached = 0

        self._gen_aircraft()

        observations = self._get_observation()
        infos = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos
    
    def step(self, actions):
        
        self._get_action(actions)

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                observations = self._get_observation()
                self._render_frame()

        rewards, dones = self._get_reward()
        observations = self._get_observation()
        
        if self.time_limit < self.steps:
            time_exceeded = True
        else:
            time_exceeded = False
        trunc = [time_exceeded] * len(self.agents)
        truncates = {
            a: d
            for a,d in zip(self.agents,trunc)
        }

        infos = self._get_info()

        self.steps += 1

        # important step to reset the agents, mandatory by pettingzoo API
        if any(dones.values()) or all(truncates.values()):
            self.agents = []

        return observations, rewards, dones, truncates, infos
    
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent] # have to define observation_spaces & action_spaces, probably in init

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_agents(self, n_agents):
        return [f'kl00{i+1}'.upper() for i in range(n_agents)]
    
    def _gen_aircraft(self):
        self.merge_angle = np.random.randint(MERGE_ANGLE_MIN,MERGE_ANGLE_MAX)
        for agent, idx in zip(self.agents,np.arange(self.num_ac)):
            bearing_to_pos = random.uniform(-self.merge_angle, self.merge_angle) # heading radial towards FAF
            distance_to_pos = random.uniform(SPAWN_DISTANCE_MIN,SPAWN_DISTANCE_MAX) # distance to faf 
            lat_ac, lon_ac = fn.get_point_at_distance(self.wpt_lat, self.wpt_lon, distance_to_pos, bearing_to_pos)

            bs.traf.cre(agent,actype="A320",acspd=AC_SPD,aclat=lat_ac,aclon=lon_ac,achdg=bearing_to_pos-180,acalt=ALTITUDE)

    def _get_observation(self):
        obs = []
        self.waypoint_dist = {a: 0 for a in self.agents}

        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)

            x_r = np.array([])
            y_r = np.array([])
            vx_r = np.array([])
            vy_r = np.array([])
            cos_track = np.array([])
            sin_track = np.array([])
            distances = np.array([])

            ac_hdg = bs.traf.hdg[ac_idx]
            
            # Get and decompose agent aircaft drift
            if self.wpt_reach[agent] == 0: # pre-faf check
                wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat, self.wpt_lon)
            else: # post-faf check
                wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.rwy_lat, self.rwy_lon)

            drift = ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            cos_drift = np.cos(np.deg2rad(drift))
            sin_drift = np.sin(np.deg2rad(drift))

            self.waypoint_dist[agent] = wpt_dist

            # Get agent aircraft airspeed, m/s
            airspeed = bs.traf.tas[ac_idx]

            vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
            vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
            
            ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
            dist = [fn.euclidean_distance(ac_loc, fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000) for i in range(self.num_ac)]
            ac_idx_by_dist = np.argsort(dist)

            for i in range(self.num_ac):
                int_idx = ac_idx_by_dist[i]
                if int_idx == ac_idx:
                    continue
                int_hdg = bs.traf.hdg[int_idx]
                
                # Intruder AC relative position, m
                int_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[int_idx], bs.traf.lon[int_idx]])) * NM2KM * 1000
                x_r = np.append(x_r, int_loc[0] - ac_loc[0])
                y_r = np.append(y_r, int_loc[1] - ac_loc[1])

                # Intruder AC relative velocity, m/s
                vx_int = np.cos(np.deg2rad(int_hdg)) * bs.traf.gs[int_idx]
                vy_int = np.sin(np.deg2rad(int_hdg)) * bs.traf.gs[int_idx]
                vx_r = np.append(vx_r, vx_int - vx)
                vy_r = np.append(vy_r, vy_int - vy)

                # Intruder AC relative track, rad
                track = np.arctan2(vy_int - vy, vx_int - vx)
                cos_track = np.append(cos_track, np.cos(track))
                sin_track = np.append(sin_track, np.sin(track))

                distances = np.append(distances, dist[int_idx])

            # very crude normalization for the observation vectors
            observation = {
                "cos(drift)": np.array([cos_drift]),
                "sin(drift)": np.array([sin_drift]),
                "airspeed": np.array([(airspeed-150)/6]),
                "x_r": np.array(x_r[:NUM_AC_STATE]/1000000),
                "y_r": np.array(y_r[:NUM_AC_STATE]/1000000),
                "vx_r": np.array(vx_r[:NUM_AC_STATE]/150),
                "vy_r": np.array(vy_r[:NUM_AC_STATE]/150),
                "cos(track)": np.array(cos_track[:NUM_AC_STATE]),
                "sin(track)": np.array(sin_track[:NUM_AC_STATE]),
                "distances": np.array(distances[:NUM_AC_STATE]-50000.)/15000.
            }

            obs.append(np.concatenate(list(observation.values())))
        
        observations = {
            a: o
            for a, o in zip(self.agents, obs)
        }
        
        return observations
    
    def _get_info(self):
        # for now just multiply the global agent info, could not be bothered
        return {
            a: {'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean()}
            for a in self.agents
        }

    def _get_reward(self):
        rew = []
        dones = []
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            reach_reward, done = self._check_waypoint(agent)
            drift_reward = self._check_drift(ac_idx, agent)
            intrusion_reward = self._check_intrusion(ac_idx)

            reward = reach_reward + drift_reward + intrusion_reward

            rew.append(reward)
            dones.append(done)
            self.total_reward += reward

        rewards = {
            a: r
            for a, r in zip(self.agents, rew)
        }

        dones = [True] * len(dones) if any(dones) else dones # ensure all False or all True
        done = {
            a: d
            for a,d in zip(self.agents,dones)
        }

        return rewards, done
        
    def _check_waypoint(self, agent):
        reward = 0
        index = 0
        done = 0
        if self.waypoint_dist[agent] < DISTANCE_MARGIN and self.wpt_reach[agent] != 1:
            self.wpt_reach[agent] = 1
            self.faf_reached = 1
            reward += REACH_REWARD
        elif self.waypoint_dist[agent] < 2*DISTANCE_MARGIN and self.wpt_reach[agent] == 1:
            self.faf_reached = 2
            done = 1 
        return reward, done

    def _check_drift(self, ac_idx, agent):
        ac_hdg = bs.traf.hdg[ac_idx]
            
        # Get and decompose agent aircaft drift
        if self.wpt_reach[agent] == 0: # pre-faf check
            wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat, self.wpt_lon)
        else: # post-faf check
            wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.rwy_lat, self.rwy_lon)

        drift = ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        drift = abs(np.deg2rad(drift))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY

    def _check_intrusion(self, ac_idx):
        reward = 0
        for i in range(self.num_ac):
            int_idx = i
            if i == ac_idx:
                continue
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        
        return reward

    def _get_action(self,actions):
        for agent in self.agents:
            action = actions[agent]
            dh = action[0] * D_HEADING
            dv = action[1] * D_VELOCITY
            heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(agent)] + dh)
            speed_new = (bs.traf.cas[bs.traf.id2idx(agent)] + dv) * MpS2Kt

            # print(speed_new)
            bs.stack.stack(f"HDG {agent} {heading_new}")
            bs.stack.stack(f"SPD {agent} {speed_new}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 300 # width of screen in km
        px_per_km = self.window_width/max_distance

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235)) 

        circle_x = self.window_width/2
        circle_y = self.window_height/2

        pygame.draw.circle(
            canvas, 
            (255,255,255),
            (circle_x,circle_y),
            radius = 4,
            width = 0
        )
        
        pygame.draw.circle(
            canvas, 
            (255,255,255),
            (circle_x,circle_y),
            radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
            width = 2
        )

        # draw line to faf
        heading_length = 5000
        heading_end_x = ((np.cos(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        pygame.draw.line(canvas,
        (0,0,0),
        (circle_x,circle_y),
        (circle_x+heading_end_x/2,circle_y-heading_end_y/2),
        width = 2
        )

        # heading boundary lines
        he_x_l = ((np.cos(np.deg2rad(-self.merge_angle)) * heading_length)/max_distance)*self.window_width
        he_y_l = ((np.sin(np.deg2rad(-self.merge_angle)) * heading_length)/max_distance)*self.window_width
        he_x_r = ((np.cos(np.deg2rad(self.merge_angle)) * heading_length)/max_distance)*self.window_width
        he_y_r = ((np.sin(np.deg2rad(self.merge_angle)) * heading_length)/max_distance)*self.window_width
        pygame.draw.line(canvas,
        (3,252,11),
        (circle_x,circle_y),
        (circle_x+he_x_l/2,circle_y-he_y_l/2),
        width = 4
        )
        pygame.draw.line(canvas,
        (3,252,11),
        (circle_x,circle_y),
        (circle_x+he_x_r/2,circle_y-he_y_r/2),
        width = 4
        )

        # draw rwy start
        rwy_faf_qdr, rwy_faf_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, RWY_LAT, RWY_LON)
        x_pos = (circle_x)+(np.cos(np.deg2rad(rwy_faf_qdr))*(rwy_faf_dis * NM2KM)/max_distance)*self.window_width
        y_pos = (circle_y)-(np.sin(np.deg2rad(rwy_faf_qdr))*(rwy_faf_dis * NM2KM)/max_distance)*self.window_height
        heading_length = 5000
        heading_end_x = ((np.cos(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        pygame.draw.line(canvas,
        (255,255,255),
        (x_pos,y_pos),
        (circle_x+heading_end_x/2,circle_y-heading_end_y/2),
        width = 4
        )

        # draw ownship
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            ac_length = 8
            heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width

            own_qdr, own_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
            x_pos = (circle_x)+(np.cos(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (circle_y)-(np.sin(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_distance)*self.window_height

            separation = bs.tools.geo.kwikdist(np.concatenate((bs.traf.lat[:ac_idx], bs.traf.lat[ac_idx+1:])), 
                                               np.concatenate((bs.traf.lon[:ac_idx], bs.traf.lon[ac_idx+1:])), 
                                               bs.traf.lat[ac_idx], 
                                               bs.traf.lon[ac_idx])

            # Determine color
            if np.any(separation < INTRUSION_DISTANCE):
                color = (220,20,60)
            else: 
                color = (80,80,80)

            pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                ((x_pos)+heading_end_x/2,(y_pos)-heading_end_y/2),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width

            pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = INTRUSION_DISTANCE*NM2KM*px_per_km/2,
                width = 2
            )

        # PyGame update
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass

class MergeEnv_ATT(MergeEnv):

    def __init__(self, render_mode=None, n_agents=10):
        super().__init__(render_mode, n_agents)
        self.observation_spaces = {agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64) for agent in self.agents}

    def _get_observation(self):
        obs = []
        self.waypoint_dist = {a: 0 for a in self.agents}

        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            ac_hdg = bs.traf.hdg[ac_idx]
            
            # Get and decompose agent aircaft drift
            if self.wpt_reach[agent] == 0: # pre-faf check
                wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat, self.wpt_lon)
            else: # post-faf check
                wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.rwy_lat, self.rwy_lon)

            drift = ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)
            cos_drift = np.cos(np.deg2rad(drift))
            sin_drift = np.sin(np.deg2rad(drift))

            self.waypoint_dist[agent] = wpt_dist

            # Get agent aircraft airspeed, m/s
            airspeed = bs.traf.tas[ac_idx]

            vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
            vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]

            ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
            x = ac_loc[0]
            y = ac_loc[1]

            observation = {
                "cos(drift)": np.array([cos_drift]),
                "sin(drift)": np.array([sin_drift]),
                "airspeed": np.array([(airspeed-150)/50]),
                "x": np.array([x/50000]),
                "y": np.array([y/50000]),
                "vx": np.array([vx/150]),
                "vy": np.array([vy/150])
            }

            obs.append(np.concatenate(list(observation.values())))

        observations = {
            a: o
            for a, o in zip(self.agents, obs)
        }

        return observations