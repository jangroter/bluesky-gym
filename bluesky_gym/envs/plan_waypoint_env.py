import numpy as np
import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

from core.observations import WaypointObservation
from core.rendering import PygameCanvas, TopDownProjection, draw_aircraft, draw_waypoint
from core.actions import HeadingAction

DISTANCE_MARGIN = 5 # km
WAYPOINT_DISTANCE_MIN = 0
WAYPOINT_DISTANCE_MAX = 75

NUM_WAYPOINTS = 5

REACH_REWARD = 1
AC_SPD = 150

D_HEADING = 45

NM2KM = 1.852

ACTION_FREQUENCY = 10

class PlanWaypointEnv(gym.Env):
    """ 
    Dummy environment for horizontal control and rendering testing.
    Goal of the agent is to fly over the the waypoints and cross as many as possible
    to score points, similar to traveling salesman problem, but without explicit planning
    and with euler integration for the turn dynamics.

    For now only heading changes are possible.

    TODO:
    - More comments
    - Clean up rendering
    - More elegant observation function
    - Speed changes (?)
    - Run long training tests with the new observation function
    
    """

    # information regarding the possible rendering modes of the environment
    # for BlueSkyGym probably only implement 1 for now together with None, which is default
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.waypoint_obs = WaypointObservation(n=NUM_WAYPOINTS, distance_norm=WAYPOINT_DISTANCE_MAX, include_status=True)

        self.observation_space = spaces.Dict({
            **self.waypoint_obs.space(),
        })
       
        self.heading_action = HeadingAction(d_heading=D_HEADING)
        self.action_space = self.heading_action.space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # set correct sim speed
        bs.stack.stack('DT 1;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.waypoints_completed = 0

        self.pygame_canvas = PygameCanvas(self.window_width, self.window_height)
        self.projection = TopDownProjection(
            max_distance=200, ref_lat=0, ref_lon=0,
            window_size=(self.window_width, self.window_height),
        )


    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')
        self.ac_hdg = bs.traf.hdg[ac_idx]

        # raw waypoint values (generation order) retained for _check_waypoint, _render_frame (approach C)
        self.wpt_dis = []
        self.wpt_qdr = []
        self.drift = []
        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
            self.wpt_dis.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)
            self.drift.append(fn.bound_angle_positive_negative_180(self.ac_hdg - wpt_qdr))

        # waypoints are sorted by distance; waypoint_status is reordered to match (per-slot reach flag)
        return self.waypoint_obs.observe('KL001', self.wpt_lat, self.wpt_lon, reached_flags=self.wpt_reach)
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "total_reward": self.total_reward,
            "waypoints_completed": self.waypoints_completed
        }
    
    def _get_reward(self):

        # Always return done as false, as this is a non-ending scenario with 
        # new waypoints spawning continously

        reach_reward = self._check_waypoint()
        self.total_reward += reach_reward

        if 0 in self.wpt_reach:
            return reach_reward, 0
        else:
            return reach_reward, 1
        
    def _get_action(self, action):
        self.heading_action.execute(self.agent, action)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_reward = 0
        self.waypoints_completed = 0

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD)

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
    
    def render(self):
        pass

    def _generate_waypoint(self, acid = 'KL001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = np.random.randint(0, 359)

            ac_idx = bs.traf.id2idx(acid)

            wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)    
            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)

    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.wpt_dis:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.waypoints_completed += 1
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        return reward

    def _render_frame(self):
        ac_idx = bs.traf.id2idx('KL001')
        self.projection.update_ref(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        canvas = self.pygame_canvas.begin_frame()

        draw_aircraft(canvas, *self.projection.center, bs.traf.hdg[ac_idx],
                      body_km=8, heading_km=50, projection=self.projection)

        for lat, lon, reached in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            x, y = self.projection.project(lat, lon)
            draw_waypoint(canvas, x, y, DISTANCE_MARGIN, self.projection,
                          reached=bool(reached))

        self.pygame_canvas.end_frame(canvas)
        
    def close(self):
        bs.stack.stack('quit')