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

DRIFT_PENALTY = -0.02 #0.1
INTRUSION_PENALTY = -0.2#-0.2 #1

INTRUSION_DISTANCE = 4 # NM

SPAWN_DISTANCE_MIN = 50
SPAWN_DISTANCE_MAX = 200

INTRUDER_DISTANCE_MIN = 20
INTRUDER_DISTANCE_MAX = 500

D_HEADING = 3 #15
D_SPEED = 4 #20 

AC_SPD = 150

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10

NUM_AC = 20
NUM_AC_STATE = 5
NUM_WAYPOINTS = 1

RWY_LAT = 52.36239301495972
RWY_LON = 4.713195734579777

distance_faf_rwy = 200 # NM
bearing_faf_rwy = 0
FIX_LAT, FIX_LON = fn.get_point_at_distance(RWY_LAT, RWY_LON, distance_faf_rwy, bearing_faf_rwy)

class MergeEnv(ParallelEnv):
    """ 
    Multi-agent arrival manager environment - all aircraft are required to merge into a single traffic stream.
    """
    metadata = {
        "name": "merge_v0",
        "render_modes": ["rgb_array","human"],
         "render_fps": 120}

    def __init__(self, render_mode=None, n_agents=10, time_limit=500):
        self.window_width = 750
        self.window_height = 500
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.n_agents = n_agents
        self.agents = self._get_agents(self.n_agents)
        self.possible_agents = self.agents[:]

        self.time_limit = time_limit
        self.steps = 0

        self.observation_spaces = {agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5+7*NUM_AC_STATE,), dtype=np.float64) for agent in self.agents}
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
        self.nac = NUM_AC
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

        self.num_episodes += 1
        if self.num_episodes > 1:
            self.reward_array = np.append(self.reward_array, self.total_reward)
            print(self.num_episodes)
            print(self.reward_array[-100:].mean())

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
        for agent, idx in zip(self.agents,np.arange(self.n_agents)):
            bearing_to_pos = random.uniform(-D_HEADING, D_HEADING) # heading radial towards FAF
            distance_to_pos = random.uniform(INTRUDER_DISTANCE_MIN,INTRUDER_DISTANCE_MAX) # distance to faf 
            lat_ac, lon_ac = fn.get_point_at_distance(self.wpt_lat, self.wpt_lon, distance_to_pos, bearing_to_pos)

            bs.traf.cre(agent,actype="A320",acspd=AC_SPD,aclat=lat_ac,aclon=lon_ac,achdg=bearing_to_pos-180,acalt=10000)
            bs.stack.stack(f"{agent} addwpt {FIX_LAT} {FIX_LON}")
            bs.stack.stack(f"{agent} dest {RWY_LAT} {RWY_LON}")
        bs.stack.stack('reso off')
        return

    def _get_observation(self):
        obs = []
        self.waypoint_dist = {a: 0 for a in self.agents}

        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)

            cos_drift = np.array([])
            sin_drift = np.array([])
            airspeed = np.array([])
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

            cos_drift = np.append(cos_drift, np.cos(np.deg2rad(drift)))
            sin_drift = np.append(sin_drift, np.sin(np.deg2rad(drift)))

            self.waypoint_dist[agent] = wpt_dist

            airspeed = np.append(airspeed, bs.traf.tas[ac_idx])
            vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
            vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
            
            distances = bs.tools.geo.kwikdist_matrix(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat,bs.traf.lon)
            ac_idx_by_dist = np.argsort(distances) # sort aircraft by distance to ownship

            for i in range(self.n_agents):
                int_idx = ac_idx_by_dist[i]
                if int_idx == ac_idx:
                    continue
                int_hdg = bs.traf.hdg[int_idx]
                
                # Intruder AC relative position, m
                brg, dist = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx],bs.traf.lon[int_idx]) 
                x_r = np.append(x_r, (dist * NM2KM * 1000) * np.cos(np.deg2rad(brg)))
                y_r = np.append(y_r, (dist * NM2KM * 1000) * np.sin(np.deg2rad(brg)))
                
                # Intruder AC relative velocity, m/s
                vx_int = np.cos(np.deg2rad(int_hdg)) * bs.traf.tas[int_idx]
                vy_int = np.sin(np.deg2rad(int_hdg)) * bs.traf.tas[int_idx]
                vx_r = np.append(vx_r, vx_int - vx)
                vy_r = np.append(vy_r, vy_int - vy)

                # Intruder AC relative track, rad
                track = np.arctan2(vy_int - vy, vx_int - vx)
                cos_track = np.append(cos_track, np.cos(track))
                sin_track = np.append(sin_track, np.sin(track))

                distances = np.append(distances, distances[ac_idx-1])

            # very crude normalization for the observation vectors
            observation = {
                "cos(drift)": np.array(cos_drift),
                "sin(drift)": np.array(sin_drift),
                "airspeed": np.array(airspeed-150)/6,
                "waypoint_dist": np.array([wpt_dist/250]),
                "faf_reached": np.array([self.wpt_reach[agent]]),
                "x_r": np.array(x_r[:NUM_AC_STATE]/1000000),
                "y_r": np.array(y_r[:NUM_AC_STATE]/1000000),
                "vx_r": np.array(vx_r[:NUM_AC_STATE]/150),
                "vy_r": np.array(vy_r[:NUM_AC_STATE]/150),
                "cos(track)": np.array(cos_track[:NUM_AC_STATE]),
                "sin(track)": np.array(sin_track[:NUM_AC_STATE]),
                "distances": np.array(distances[:NUM_AC_STATE]/250)
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
        # TODO: have to check if setting all dones to True does not mess up the TD error
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
        for i in range(self.n_agents):
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
            dv = action[1] * D_SPEED
            heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(agent)] + dh)
            speed_new = (bs.traf.cas[bs.traf.id2idx(agent)] + dv)# * MpS2Kt

            # print(speed_new)
            bs.stack.stack(f"HDG {agent} {heading_new}")
            bs.traf.ap.selspdcmd(bs.traf.id2idx(agent),speed_new)
            # bs.stack.stack(f"SPD {agent} {speed_new}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 500 # width of screen in km

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
        he_x_l = ((np.cos(np.deg2rad(180+135)) * heading_length)/max_distance)*self.window_width
        he_y_l = ((np.sin(np.deg2rad(180+135)) * heading_length)/max_distance)*self.window_width
        he_x_r = ((np.cos(np.deg2rad(180-135)) * heading_length)/max_distance)*self.window_width
        he_y_r = ((np.sin(np.deg2rad(180-135)) * heading_length)/max_distance)*self.window_width
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
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
                width = 2
            )

        # # draw intruders
        # ac_length = 3

        # for i in range(1,NUM_AC):
        #     int_idx = i
        #     int_hdg = bs.traf.hdg[int_idx]
        #     heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
        #     heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

        #     int_qdr, int_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, bs.traf.lat[int_idx], bs.traf.lon[int_idx])

        #     # determine color
        #     if int_dis < INTRUSION_DISTANCE:
        #         color = (220,20,60)
        #     else: 
        #         color = (80,80,80)
        #     if i==0:
        #         color = (252, 43, 28)

        #     x_pos = (circle_x)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
        #     y_pos = (circle_y)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

        #     pygame.draw.line(canvas,
        #         color,
        #         (x_pos,y_pos),
        #         ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
        #         width = 4
        #     )

        #     # draw heading line
        #     heading_length = 10
        #     heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width
        #     heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width

        #     pygame.draw.line(canvas,
        #         color,
        #         (x_pos,y_pos),
        #         ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
        #         width = 1
        #     )

        #     pygame.draw.circle(
        #         canvas, 
        #         color,
        #         (x_pos,y_pos),
        #         radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
        #         width = 2
        #     )

        # PyGame update
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass