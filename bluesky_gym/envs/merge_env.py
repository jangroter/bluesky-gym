# TODO: change faf_reached to standardized waypoint reached approach (building on waypoint observation functionality)

import numpy as np
import pygame

import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

import random

from core.observations import DriftObservation, OwnAirspeedObservation, WaypointObservation, IntruderObservation

DISTANCE_MARGIN = 10 # km
REACH_REWARD = 1

DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1

INTRUSION_DISTANCE = 4 # NM

SPAWN_DISTANCE_MIN = 50
SPAWN_DISTANCE_MAX = 200

INTRUDER_DISTANCE_MIN = 20
INTRUDER_DISTANCE_MAX = 500

D_HEADING = 15
D_SPEED = 20 

AC_SPD = 100

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

class MergeEnv(gym.Env):
    """ 
    Single-agent arrival manager environment - only one aircraft (ownship) is merged into NPC stream of aircraft.
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 750
        self.window_height = 500
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.drift_obs = DriftObservation()
        self.airspeed_obs = OwnAirspeedObservation(spd_mean=0.0, spd_std=1.0)
        self.waypoint_obs = WaypointObservation(n=1, distance_norm=250 * NM2KM)
        self.intruder_obs = IntruderObservation(n=NUM_AC_STATE, sort_by="distance",
                                                  pos_norm=1_000_000, spd_norm=150, dist_norm=250)

        self.observation_space = spaces.Dict({
            **self.drift_obs.space(),
            **self.airspeed_obs.space(),
            **self.waypoint_obs.space(),
            "faf_reached": spaces.Box(0, 1, shape=(1,), dtype=np.float64),
            **self.intruder_obs.space(),
        })

        self.agent = "KL001"
       
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # set correct sim speed
        bs.stack.stack('DT 5;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.average_drift = []
        self.total_intrusions = 0
        self.faf_reached = 0

        self.window = None
        self.clock = None
        self.nac = NUM_AC
        self.wpt_reach = 0
        self.wpt_lat = FIX_LAT
        self.wpt_lon = FIX_LON
        self.rwy_lat = RWY_LAT
        self.rwy_lon = RWY_LON

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wpt_reach = 0
        
        bs.traf.reset()

        self.total_reward = 0
        self.average_drift = []
        self.total_intrusions = 0
        self.faf_reached = 0

        # ownship spawn location
        bearing_to_pos = random.uniform(-D_HEADING, D_HEADING) # heading radial towards FAF
        distance_to_pos = random.uniform(SPAWN_DISTANCE_MIN,SPAWN_DISTANCE_MAX)  # distance to faf 
        rlat, rlon = fn.get_point_at_distance(FIX_LAT, FIX_LON, distance_to_pos, bearing_to_pos)

        bs.traf.cre(self.agent,actype="A320",acspd=AC_SPD, aclat= rlat, aclon= rlon, achdg=bearing_to_pos-180,acalt=10000)

        # generate other aircraft
        self._gen_aircraft()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        
        self._get_action(action)

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()
        return observation, reward, terminated, False, info
    
    def _gen_aircraft(self):
        for i in range(NUM_AC-1):
            bearing_to_pos = random.uniform(-D_HEADING, D_HEADING) # heading radial towards FAF
            distance_to_pos = random.uniform(INTRUDER_DISTANCE_MIN,INTRUDER_DISTANCE_MAX) # distance to faf 
            lat_ac, lon_ac = fn.get_point_at_distance(self.wpt_lat, self.wpt_lon, distance_to_pos, bearing_to_pos)

            bs.traf.cre(f'INT{i}',actype="A320",acspd=AC_SPD,aclat=lat_ac,aclon=lon_ac,achdg=bearing_to_pos-180,acalt=10000)
            bs.stack.stack(f"INT{i} addwpt {FIX_LAT} {FIX_LON}")
            bs.stack.stack(f"INT{i} dest {RWY_LAT} {RWY_LON}")
        bs.stack.stack('reso off')
        return

    def _get_obs(self):
        ac_idx = bs.traf.id2idx(self.agent)

        # target switches from the merge fix to the runway once the fix is reached
        if self.wpt_reach == 0:
            target_lat, target_lon = self.wpt_lat, self.wpt_lon
        else:
            target_lat, target_lon = self.rwy_lat, self.rwy_lon

        # raw waypoint values retained for _check_waypoint, _check_drift (approach C)
        wpt_qdr, wpt_dist = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], target_lat, target_lon)
        self.waypoint_dist = wpt_dist
        self.drift = fn.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - wpt_qdr)

        return {
            **self.drift_obs.observe(self.agent, wpt_qdr),
            **self.airspeed_obs.observe(self.agent),
            **self.waypoint_obs.observe(self.agent, [target_lat], [target_lon]),
            "faf_reached": np.array([self.wpt_reach]),
            **self.intruder_obs.observe(self.agent),
        }
    
    def _get_info(self):
        return {
            "total_reward": self.total_reward,
            "faf_reach": self.faf_reached,
            "average_drift": np.mean(self.average_drift),
            "total_intrusions": self.total_intrusions
        }

    def _get_reward(self):
        reach_reward, done = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()

        reward = reach_reward + drift_reward + intrusion_reward

        self.total_reward += reward

        return reward, done      
        
    def _check_waypoint(self):
        reward = 0
        index = 0
        done = 0
        if self.waypoint_dist < DISTANCE_MARGIN and self.wpt_reach != 1:
            self.wpt_reach = 1
            self.faf_reached = 1
            reward += REACH_REWARD
        elif self.waypoint_dist < 2*DISTANCE_MARGIN and self.wpt_reach == 1:
            self.faf_reached = 2
            done = 1 
        return reward, done

    def _check_drift(self):
        drift = abs(np.deg2rad(self.drift))
        self.average_drift.append(drift)
        return drift * DRIFT_PENALTY

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        for i in range(NUM_AC-1): 
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        return reward    

    def _get_action(self,action):
        dh = action[0] * D_HEADING
        dv = action[1] * D_SPEED
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx('KL001')] + dh)
        speed_new = (bs.traf.cas[bs.traf.id2idx('KL001')] + dv) * MpS2Kt

        bs.stack.stack(f"HDG KL001 {heading_new}")
        bs.stack.stack(f"SPD KL001 {speed_new}")
        
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
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width

        own_qdr, own_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        x_pos = (circle_x)+(np.cos(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_distance)*self.window_width
        y_pos = (circle_y)-(np.sin(np.deg2rad(own_qdr))*(own_dis * NM2KM)/max_distance)*self.window_height
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

        # draw intruders
        ac_length = 3

        for i in range(1,NUM_AC):
            int_idx = i
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            if int_dis < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)
            if i==0:
                color = (252, 43, 28)

            x_pos = (circle_x)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (circle_y)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width

            pygame.draw.line(canvas,
                color,
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

        # PyGame update
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        bs.stack.stack('quit')