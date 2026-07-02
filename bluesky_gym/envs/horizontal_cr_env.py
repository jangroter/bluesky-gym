
import numpy as np
import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

from core.observations import IntruderObservation, WaypointObservation
from core.rendering import (
    PygameCanvas, TopDownProjection,
    draw_aircraft, draw_intruder, draw_waypoint,
)
from core.actions import HeadingAction

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1

DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1

NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5 # NM

WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150

D_HEADING = 45

AC_SPD = 150

NM2KM = 1.852

ACTION_FREQUENCY = 10

class HorizontalCREnv(gym.Env):
    """ 
    Horizontal Conflict Resolution Environment

    TODO:
    - look at adding waypoints instead of staying straight
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment


        self.intruder_obs = IntruderObservation(n=NUM_INTRUDERS)
        self.waypoint_obs = WaypointObservation(n=NUM_WAYPOINTS, distance_norm=WAYPOINT_DISTANCE_MAX)

        self.observation_space = spaces.Dict({
            **self.intruder_obs.space(),
            **self.waypoint_obs.space(),
        })

        self.agent = "KL001"

        self.heading_action = HeadingAction(d_heading=D_HEADING)
        self.action_space = self.heading_action.space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # set correct sim speed
        bs.stack.stack('DT 5;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        self.pygame_canvas = PygameCanvas(self.window_width, self.window_height)
        self.projection = TopDownProjection(
            max_distance=200, ref_lat=0, ref_lon=0,
            window_size=(self.window_width, self.window_height),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()

        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        bs.traf.cre(self.agent,actype="A320",acspd=AC_SPD,achdg=90)

        self._generate_conflicts()
        self._generate_waypoint()
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
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info

    def _generate_conflicts(self, acid = 'KL001'):
        target_idx = bs.traf.id2idx(acid)
        for i in range(NUM_INTRUDERS):
            dpsi = np.random.randint(45,315)
            cpa = np.random.randint(0,INTRUSION_DISTANCE)
            tlosh = np.random.randint(100,1000)
            bs.traf.creconfs(acid=f'{i}',actype="A320",targetidx=target_idx,dpsi=dpsi,dcpa=cpa,tlosh=tlosh)

    def _generate_waypoint(self, acid = 'KL001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = 90

            ac_idx = bs.traf.id2idx(acid)

            wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)    
            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')
        self.ac_hdg = bs.traf.hdg[ac_idx]

        # raw waypoint values retained for _check_waypoint, _check_drift, _render_frame (approach C)
        self.waypoint_distance = []
        self.wpt_qdr = []
        self.drift = []
        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
            self.waypoint_distance.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)
            self.drift.append(fn.bound_angle_positive_negative_180(self.ac_hdg - wpt_qdr))

        return {
            **self.intruder_obs.observe(self.agent),
            **self.waypoint_obs.observe(self.agent, self.wpt_lat, self.wpt_lon),
        }
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean()
        }

    def _get_reward(self):

        # Always return done as false, as this is a non-ending scenario with 
        # new waypoints spawning continously

        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()

        total_reward = reach_reward + drift_reward + intrusion_reward
        self.total_reward += total_reward

        if 0 in self.wpt_reach:
            return total_reward, 0
        else:
            return total_reward, 1
        
    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.waypoint_distance:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        return reward

    def _check_drift(self):
        drift = abs(np.deg2rad(self.drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        
        return reward
    
    def _get_action(self, action):
        self.heading_action.execute(self.agent, action)

    def _render_frame(self):
        ac_idx = bs.traf.id2idx('KL001')
        self.projection.update_ref(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        canvas = self.pygame_canvas.begin_frame()

        draw_aircraft(canvas, *self.projection.center, bs.traf.hdg[ac_idx],
                      body_km=8, heading_km=50, projection=self.projection)

        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            x, y = self.projection.project(bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            int_dis = bs.tools.geo.kwikdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            draw_intruder(canvas, x, y, bs.traf.hdg[int_idx], self.projection,
                          body_km=3, heading_km=10,
                          safety_radius_km=INTRUSION_DISTANCE * NM2KM,
                          in_intrusion=int_dis < INTRUSION_DISTANCE)

        for lat, lon, reached in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            x, y = self.projection.project(lat, lon)
            draw_waypoint(canvas, x, y, DISTANCE_MARGIN, self.projection,
                          reached=bool(reached))

        self.pygame_canvas.end_frame(canvas)
        
    def close(self):
        bs.stack.stack('quit')