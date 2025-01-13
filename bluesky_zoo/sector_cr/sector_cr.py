"""
This table enumerates the observation space for 3 neigbours in observation space,
if number of neighbours included is changed, indices will change aswell (of course):

| Index: [start, end) | Description                                                  |   Values    |
|:-----------------:|------------------------------------------------------------|:---------------:|
|          0          | cos(track deviation)                                         | [-inf, inf] |
|          1          | sin(track deviation)                                         | [-inf, inf] |
|          2          | airspeed                                                     | [-inf, inf] |
|         3-5         | Relative X postions with 3 closest neighbours                | [-inf, inf] |
|         6-8         | Relative Y postions with 3 closest neighbours                | [-inf, inf] |
|         9-11        | Relative X velocity with 3 closest neighbours                | [-inf, inf] |
|        12-14        | Relative Y velocity with 3 closest neighbours                | [-inf, inf] |
|        15-17        | cos(heading difference) with 3 closest neighbours            | [-inf, inf] |
|        18-20        | sin(heading difference) with 3 closest neighbours            | [-inf, inf] |
|        21-23        | distance with 3 closest neighbours                           | [-inf, inf] |

it would be nice if we could construct different observation classes for each agent, but for now it is fixed.

"""

import functools
from pettingzoo import ParallelEnv
import gymnasium as gym
import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import bluesky_gym.envs.common.functions as fn

POLY_AREA_RANGE = (2400, 3750) # In NM^2
CENTER = np.array([51.990426702297746, 4.376124857109851]) # TU Delft AE Faculty coordinates
ALTITUDE = 350 # In FL

# Aircraft parameters
AC_SPD = 150
AC_TYPE = "A320"

# Conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FL2M = 30.48

INTRUSION_DISTANCE = 5 # NM

# Model parameters
ACTION_FREQUENCY = 5
NUM_AC_STATE = 3
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1
D_HEADING = 22.5 # deg
D_VELOCITY = 20/3 # kts


class SectorCR(ParallelEnv):
    
    metadata = {
        "name": "sector_cr_v0",
        "render_modes": ["rgb_array","human"], 
        "render_fps": 120
    }

    def __init__(self, render_mode=None, n_agents=10):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.num_ac = n_agents

        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        self.reset_counter = 0
        self.poly_name = 'airspace'
        
        self.agents = self._get_agents(self.num_ac)
        self.possible_agents = self.agents[:]

        self.observation_spaces = {agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3+7*NUM_AC_STATE,), dtype=np.float64) for agent in self.agents}
        self.action_spaces = {agent: gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64) for agent in self.agents}

        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')


        self.total_reward = 0
        self.reward_array = np.array([])
        self.num_episodes = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        
        bs.traf.reset()
        bs.tools.areafilter.deleteArea(self.poly_name)
        self.agents = self.possible_agents[:]
        self.num_episodes += 1
        if self.num_episodes > 1:
            self.reward_array = np.append(self.reward_array, self.total_reward)
            print(f'episode: {self.num_episodes}, avg rew: {self.reward_array[-100:].mean()}')
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        self._generate_polygon() # Create airspace polygon
        self._generate_waypoints() # Create waypoints for aircraft
        self._generate_ac()

        observations = self._get_observation()
        infos = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def step(self, actions):
        """
        Probably requires something like:

        for agent in self.agents:
            action = actions[agent]
            
            .. do something

        for _ in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":              
                self._render_frame()
            
            .. maybe also update rewards here
        
        observation = self._get_observation()        
        reward = self._get_reward()
        info = self._get_info()

        """
        self._do_action(actions)
        action_frequency = ACTION_FREQUENCY
        for _ in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":              
                self._render_frame()
        
        observations = self._get_observation()     
        rewards = self._get_reward()
        infos = self._get_info()

        # truncate instead of terminate to avoid aircraft learning to exit sector fast
        dones = self._get_dones()
        truncates = self._check_inside_airspace()
        
        # important step to reset the agents, mandatory by pettingzoo API
        if any(dones.values()) or all(truncates.values()):
            self.agents = []

        return observations, rewards, dones, truncates, infos

    def render(self):
        pass

    def _get_observation(self):
        """
        simple oneliner observation example to showcase the structure

        observations = {
            a: (
                state_1,
                state_2,
                state_3,
            )
            for a in self.agents
        }

        probably needs something more complex for the bluesky_zoo envs
        maybe initialize empty dicts of size observation_space and then fill in loop
        """
        
        obs = []

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
            wpts = fn.nm_to_latlong(CENTER, self.wpts[ac_idx])
            wpt_qdr, _  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpts[0], wpts[1])
            drift = ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)
            cos_drift = np.cos(np.deg2rad(drift))
            sin_drift = np.sin(np.deg2rad(drift))
        
            # Get agent aircraft airspeed, m/s
            airspeed = bs.traf.tas[ac_idx]

            vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]
            vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.gs[ac_idx]

            ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
            dist = [fn.euclidean_distance(ac_loc, fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000) for i in range(self.num_ac)]
            ac_idx_by_dist = np.argsort(dist)

            # quick import code dump to not forget that this should be unit checked

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

            observation = {
                "cos(drift)": np.array([cos_drift]),
                "sin(drift)": np.array([sin_drift]),
                "airspeed": np.array([(airspeed-150)/6]),
                "x_r": x_r[:NUM_AC_STATE]/13000,
                "y_r": y_r[:NUM_AC_STATE]/13000,
                "vx_r": vx_r[:NUM_AC_STATE]/32,
                "vy_r": vy_r[:NUM_AC_STATE]/66,
                "cos(track)": cos_track[:NUM_AC_STATE],
                "sin(track)": sin_track[:NUM_AC_STATE],
                "distances": (distances[:NUM_AC_STATE]-50000.)/15000.
            }

            obs.append(np.concatenate(list(observation.values())))

        observations = {
            a: o
            for a, o in zip(self.agents, obs)
        }
        return observations

    def _get_reward(self):
        rew = []
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            drift_reward = self._check_drift(ac_idx)
            intrusion_reward = self._check_intrusion(ac_idx)

            reward = drift_reward + intrusion_reward
            rew.append(reward)
            self.total_reward += reward

        rewards = {
            a: r
            for a, r in zip(self.agents, rew)
        }

        return rewards

    def _do_action(self, actions):
        for agent in self.agents:
            action = actions[agent]
            dh = action[0] * D_HEADING
            dv = action[1] * D_VELOCITY
            heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(agent)] + dh)
            speed_new = (bs.traf.tas[bs.traf.id2idx(agent)] + dv) * MpS2Kt

            # print(speed_new)
            bs.stack.stack(f"HDG {agent} {heading_new}")
            bs.stack.stack(f"SPD {agent} {speed_new}")

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
    
    def _generate_polygon(self):
        
        R = np.sqrt(POLY_AREA_RANGE[1] / np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        
        while p_area < POLY_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        self.poly_area = p_area
        
        self.poly_points = np.array(p) # Polygon vertices are saved in terms of NM
        
        p = [fn.nm_to_latlong(CENTER, point) for point in p] # Convert to lat/long coordinateS
        
        points = [coord for point in p for coord in point] # Flatten the list of points
        bs.tools.areafilter.defineArea(self.poly_name, 'POLY', points)

    def _generate_waypoints(self):
        
        edges = []
        perim_tot = 0
        
        for i in range(len(self.poly_points)):
            p1 = np.array(self.poly_points[i])
            p2 = np.array(self.poly_points[(i+1) % len(self.poly_points)]) # Ensure wrap-around
            len_edge = fn.euclidean_distance(p1, p2)
            edges.append((p1, p2, len_edge))
            perim_tot += len_edge
        
        d_list = [np.random.uniform(0, perim_tot) for _ in range(self.num_ac)] # Each ac including agent is given a waypoint
        d_list.sort()
        
        self.wpts = []
        current_d = 0
        
        for d in d_list:
            while d > current_d + edges[0][2]:
                current_d += edges[0][2]
                edges.pop(0)
            
            edge = edges[0]
            frac = (d - current_d) / edge[2]
            p = edge[0] + frac * (edge[1] - edge[0])
            p = p*10
            self.wpts.append(p)

    def _generate_ac(self) -> None:
        # Determine bounding box of airspace
        min_x = min(self.poly_points[:, 0])
        min_y = min(self.poly_points[:, 1])
        max_x = max(self.poly_points[:, 0])
        max_y = max(self.poly_points[:, 1])
        
        init_p_latlong = []
        
        while len(init_p_latlong) < self.num_ac:
            p = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            p = fn.nm_to_latlong(CENTER, p)
            if bs.tools.areafilter.checkInside(self.poly_name, np.array([p[0]]), np.array([p[1]]), np.array([ALTITUDE*FL2M])):
                init_p_latlong.append(p)
        
        for agent, idx in zip(self.agents,np.arange(self.num_ac)):
            wpt_agent = fn.nm_to_latlong(CENTER, self.wpts[idx])
            init_pos_agent = init_p_latlong[idx]
            hdg_agent = fn.get_hdg(init_pos_agent, wpt_agent)
            bs.traf.cre(agent, actype=AC_TYPE, aclat=init_pos_agent[0], aclon=init_pos_agent[1], achdg=hdg_agent, acspd=AC_SPD, acalt=ALTITUDE)

    def _check_drift(self, ac_idx):
        ac_hdg = bs.traf.hdg[ac_idx]
        
        # Get and decompose agent aircaft drift
        wpts = fn.nm_to_latlong(CENTER, self.wpts[ac_idx])
        wpt_qdr, _  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpts[0], wpts[1])
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

    def _check_inside_airspace(self):
        truncs = []
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            if bs.tools.areafilter.checkInside(self.poly_name, np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([ALTITUDE*FL2M])):
                truncs.append(False)
            else:
                truncs.append(True)
        truncs = [False] * len(truncs) if not all(truncs) else truncs # ensure all False or all True
        trunctated = {
            a: t
            for a,t in zip(self.agents,truncs)
        }
        return trunctated

    def _get_dones(self):
        done = [False] * len(self.agents)
        dones = {
            a: d
            for a,d in zip(self.agents,done)
        }
        return dones # we are never done

    def _get_info(self):
        # for now just multiply the global agent info, could not be bothered
        return {
            a: {'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean()}
            for a in self.agents
        }

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = max(np.linalg.norm(point1 - point2) for point1 in self.poly_points for point2 in self.poly_points)*NM2KM
        
        px_per_km = self.window_width/max_distance

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))
        
        # Draw airspace
        airspace_color = (255, 0, 0)
        coords = [((self.window_width/2)+point[0]*NM2KM*px_per_km, (self.window_height/2)-point[1]*NM2KM*px_per_km) for point in self.poly_points]
        pygame.draw.polygon(canvas, airspace_color, coords, width=2)

        # Draw ownship
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            ac_length = 10
            ac_hdg = bs.traf.hdg[ac_idx]
            heading_end_x = np.cos(np.deg2rad(ac_hdg)) * ac_length
            heading_end_y = np.sin(np.deg2rad(ac_hdg)) * ac_length
            ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])

            separation = bs.tools.geo.kwikdist(np.concatenate((bs.traf.lat[:ac_idx], bs.traf.lat[ac_idx+1:])), 
                                               np.concatenate((bs.traf.lon[:ac_idx], bs.traf.lon[ac_idx+1:])), 
                                               bs.traf.lat[ac_idx], 
                                               bs.traf.lon[ac_idx])

            # Determine color
            if np.any(separation < INTRUSION_DISTANCE):
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.window_width/2)+(np.cos(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*px_per_km)
            y_pos = (self.window_height/2)-(np.sin(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*px_per_km)

            pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # Draw heading line
            heading_length = 20
            heading_end_x = np.cos(np.deg2rad(ac_hdg)) * heading_length
            heading_end_y = np.sin(np.deg2rad(ac_hdg)) * heading_length

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

        # # Draw intruders
        # ac_length = 3

        # for i in range(self.num_ac-1):
        #     int_idx = i+1
        #     int_hdg = bs.traf.hdg[int_idx]
        #     heading_end_x = np.cos(np.deg2rad(int_hdg)) * ac_length
        #     heading_end_y = np.sin(np.deg2rad(int_hdg)) * ac_length

        #     int_qdr, int_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        #     separation = bs.tools.geo.kwikdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

        #     # Determine color
        #     if separation < INTRUSION_DISTANCE:
        #         color = (220,20,60)
        #     else: 
        #         color = (80,80,80)

        #     x_pos = (self.window_width/2)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)
        #     y_pos = (self.window_height/2)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)

        #     pygame.draw.line(canvas,
        #         color,
        #         (x_pos,y_pos),
        #         ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
        #         width = 4
        #     )

        #     # Draw heading line
        #     heading_length = 20
        #     heading_end_x = np.cos(np.deg2rad(int_hdg)) * heading_length
        #     heading_end_y = np.sin(np.deg2rad(int_hdg)) * heading_length

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
        #         radius = INTRUSION_DISTANCE*NM2KM*px_per_km,
        #         width = 2
        #     )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
