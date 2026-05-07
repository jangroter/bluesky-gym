import numpy as np
import pygame

import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 5 # km
WAYPOINT_DISTANCE_MIN = 0
WAYPOINT_DISTANCE_MAX = 75

NUM_WAYPOINTS = 5

REACH_REWARD = 1
AC_SPD = 150

D_HEADING = 45

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
    
    """

    # information regarding the possible rendering modes of the environment
    # for BlueSkyGym probably only implement 1 for now together with None, which is default
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.observation_space = spaces.Dict(
            {
                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "cos_difference": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "sin_difference": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "waypoint_reached": spaces.Box(0, 1, shape = (NUM_WAYPOINTS,), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

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
        Observation consists of distance to the waypoint and heading difference with respect to the waypoint
        in cosine and sine decomposition.

        """

        NM2KM = 1.852
        ac_idx = bs.traf.id2idx('KL001')

        self.wpt_dis = []
        self.wpt_qdr = []
        self.drift = []
        self.wpt_cos = []
        self.wpt_sin = []
        
        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            
            self.ac_hdg = bs.traf.hdg[ac_idx]
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
        
            self.wpt_dis.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)

            drift = self.ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            self.wpt_cos.append(np.cos(np.deg2rad(drift)))
            self.wpt_sin.append(np.sin(np.deg2rad(drift)))
            self.drift.append(drift)

        observation = {
                "waypoint_distance": (np.array(self.wpt_reach) -1)* -1 * np.array(self.wpt_dis)/WAYPOINT_DISTANCE_MAX,
                "cos_difference": (np.array(self.wpt_reach) -1)* -1 * np.array(self.wpt_cos),
                "sin_difference": (np.array(self.wpt_reach) -1)* -1 * np.array(self.wpt_sin),
                "waypoint_reached": np.array(self.wpt_reach)
            }
        
        return observation
    
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
        
    def _get_action(self,action):

        # Transform action to the change in heading
        # action = np.random.randint(-100,100)/100
        action = self.ac_hdg + action * D_HEADING

        bs.stack.stack(f"HDG KL001 {action[0]}")

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
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 1
        )

        # draw target waypoint
        for qdr, dis, reach in zip(self.wpt_qdr, self.wpt_dis, self.wpt_reach):

            circle_x = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
            circle_y = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width

            if reach:
                color = (155,155,155)
            else:
                color = (255,255,255)

            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        bs.stack.stack('quit')