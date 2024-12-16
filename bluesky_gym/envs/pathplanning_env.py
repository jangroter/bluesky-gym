import numpy as np
import pygame

import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.path import Path

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

from bluesky.traffic import Route

import gymnasium as gym
from gymnasium import spaces

POPULATION_WEIGHT = -0.5
PATH_LENGTH_WEIGHT = -0.00025

SCHIPHOL = [52.3068953,4.760783] # lat,lon coords of schiphol for reference to x_array and y_array
NM2KM = 1.852
NM2M = 1852.

RUNWAYS_SCHIPHOL_FAF = {
        "18C": {"lat": 52.301851, "lon": 4.737557, "track": 183},
        "36C": {"lat": 52.330937, "lon": 4.740026, "track": 3},
        "18L": {"lat": 52.291274, "lon": 4.777391, "track": 183},
        "36R": {"lat": 52.321199, "lon": 4.780119, "track": 3},
        "18R": {"lat": 52.329170, "lon": 4.708888, "track": 183},
        "36L": {"lat": 52.362334, "lon": 4.711910, "track": 3},
        "06":   {"lat": 52.304278, "lon": 4.776817, "track": 60},
        "24":   {"lat": 52.288020, "lon": 4.734463, "track": 240},
        "09":   {"lat": 52.318362, "lon": 4.796749, "track": 87},
        "27":   {"lat": 52.315940, "lon": 4.712981, "track": 267},
        "04":   {"lat": 52.313783, "lon": 4.802666, "track": 45},
        "22":   {"lat": 52.300518, "lon": 4.783853, "track": 225}
    }

FAF_DISTANCE = 10 #km
IAF_DISTANCE = 15 #km, from FAF
IAF_ANGLE = 60 #degrees, symmetrical around FAF

MIN_DISTANCE = FAF_DISTANCE + IAF_DISTANCE
MAX_DISTANCE = 300

MAX_DIS_NEXT_WPT = 30 #km
MIN_DIS_NEXT_WPT = 5 #km

# constants in this environment
SPEED = 125 # m/s
ALTITUDE = 3000 # m
SIM_DT = 5 # s

DISTANCE_MARGIN = 4.5 # km

class PathPlanningEnv(gym.Env):
    """ 
    Single agent path planning environment based on simple states ([x,y]).
    Penalized for path length and population exposed to the noise of the aircraft. 
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 1000}

    def __init__(self, render_mode=None, runway="27"):
        self.runway = runway
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "y": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack(f'DT {SIM_DT};FF')

        # load the data used for this environment
        self.pop_array = np.genfromtxt('bluesky_gym/envs/data/population_1km.csv', delimiter = ' ')
        self.x_array = np.genfromtxt('bluesky_gym/envs/data/x_array.csv', delimiter = ' ')
        self.y_array = np.genfromtxt('bluesky_gym/envs/data/y_array.csv', delimiter = ' ')
        self.x_max = np.max(self.x_array)
        self.y_max = np.max(self.y_array)
        self.cell_size = 1000 # distance per pixel in pop_array, in m
        self.projection_size = 30 # distance in km that noise is projected down, similar to kernel size in CNN

        # initialize values used for logging -> input in _get_info
        self.segment_reward = 0
        self.total_reward = 0

        self.segment_noise = 0
        self.total_noise = 0

        self.segment_length = 0
        self.total_length = 0

        self.population_weight = POPULATION_WEIGHT
        self.path_length_weight = PATH_LENGTH_WEIGHT

        self.average_noise = 0
        self.average_path = 0

        self.wpt_reach = False
        self.terminated = False
        self.truncated = False

        self.lat = 0
        self.lon = 0

        self.lat_list = []
        self.lon_list = []
        
        self._set_terminal_conditions()
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """
        resets environment after getting a done flag through terminal condition or time-out
        should return observation and contain spawning logic
        """
        super().reset(seed=seed)
        bs.traf.reset()

        self.average_noise = 0
        self.average_path = 0

        self.total_reward = 0
        self.segment_reward = 0

        self.terminated = False
        self.truncated = False
        self.wpt_reach = False

        spawn_lat, spawn_lon, spawn_heading = self._get_spawn()
        bs.traf.cre('kl001','a320',spawn_lat,spawn_lon,spawn_heading,ALTITUDE,SPEED)
        acrte = Route._routes.get('kl001')
        acrte.delrte(0)

        bs.traf.ap.setdest(0,'EHAM')
        bs.traf.ap.setLNAV(0, True)
        bs.traf.ap.route[0].addwptMode(0,'FLYOVER')

        self.lat = bs.traf.lat[0]
        self.lon = bs.traf.lon[0]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        this executes the action and progresses the environment till a new action is required
        main MDP logic is contained here
        """
        self.segment_reward = 0
        self._get_action(action)
        while not self.wpt_reach:
            bs.sim.step()
            self._update_wpt_reach()
            self._update_reward()
            terminated = self._get_terminated()
            truncated = self._get_truncated()
            if terminated:
                break
            if truncated:
                break
            if self.render_mode == "human":
                self._render_frame()

        self.wpt_reach = False
        observation = self._get_obs()
        reward = self._get_reward()
        self.total_reward += reward
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def _get_obs(self):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """

        brg, dis = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0])

        x = np.sin(np.radians(brg))*dis*NM2KM / MAX_DISTANCE
        y = np.cos(np.radians(brg))*dis*NM2KM / MAX_DISTANCE

        observation = {
            "x" : np.array([x]),
            "y" : np.array([y])
        }
        return observation
    
    def _get_info(self):
        """
        returns only the total reward of the episode for now, potentially extend later
        """
        return {
            "total_reward": self.total_reward,
            "average_path_rew": self.average_path,
            "average_noise_rew": self.average_noise,
            "population_weight": self.population_weight,
            "path_length_weight": self.path_length_weight
        }
    
    def _get_reward(self):
        """
        Reward is based on pathlength and population exposure and is a weighted bicriterion problem
        Additional rewards/penalties are included for satisfying terminal conditions
        see: _update_reward(), _get_terminated()

        Because the reward is build up of multiple components that are calculated each sim.step() call,
        this function just returns the self.segment reward, which gets updated in:

            _update_reward()
            _get_terminated()
            _get_truncated()
        """
        return self.segment_reward

    def _update_reward(self):
        """
        Updates the reward in accordance with the segment cost associated with population exposure
        and length of the path segment (as travelled through the air, to correct for wind)
        """
        path_length = self._get_path_length() * self.path_length_weight
        population_exposure = self._get_population_exposure() * self.population_weight 

        self.average_path += path_length
        self.average_noise += population_exposure

        self.segment_reward += path_length + population_exposure

    def _update_wpt_reach(self):
        """
        Performs a quick check to see if the last waypoint has been reaches,
        it does this by checking if the current active waypoint is the destination of the aircraft
        in which case a new waypoint should be set
        """
        acrte = Route._routes.get('kl001')
        if bs.traf.actwp.lat[0] == acrte.wplat[-1]:
            self.wpt_reach = True

    def _get_path_length(self):
        """
        Use deadreckoning on the current True Airspeed of the aircraft to approximate
        the distance travelled through the air.
        Note that this does not have to correspond to the groundspeed and hence the distance 
        covered over ground due to altitude and wind effects.

        By doing this we favour flying in regions with tailwind.
        """
        dist = bs.traf.tas[0] * SIM_DT / 1852.
        return dist

    def _get_population_exposure(self):
        """
        Calculates the population exposed to the aircraft, scaled with the inverse square 
        of the distance. Inverse square of the distance is based on the inverse square law
        of noise dissipation.

        This function assumes that 2 people exposed to 500 units of noise would be equivalent to 
        1 person being exposed to 1000 units of noise. Can be replaced by a more accurate noise
        cost function, but is deemed enough showcasing this environment. 
        """
        brg, dist = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0])

        x = np.sin(np.radians(brg))*dist*NM2M
        y = np.cos(np.radians(brg))*dist*NM2M
        z = bs.traf.alt[0]
        
        x_index_min = int(((x+self.x_max)/self.cell_size)-self.projection_size)
        x_index_max = int(((x+self.x_max)/self.cell_size)+self.projection_size)
        y_index_min = int(((self.y_max - y)/self.cell_size)-self.projection_size)
        y_index_max = int(((self.y_max - y)/self.cell_size)+self.projection_size)

        distance2 = (self.x_array[y_index_min:y_index_max,x_index_min:x_index_max]-x)**2 + (self.y_array[y_index_min:y_index_max,x_index_min:x_index_max]-y)**2 + z**2
        return np.sum(self.pop_array[y_index_min:y_index_max,x_index_min:x_index_max]/distance2)

    def _get_terminated(self):
        """
        Checks if the aircraft has passed the IAF beacon and can be routed to the FAF (SINK)
        or if it has missed approach by coming in with a too high turn radius requirements (RESTRICT)
        """
        self.terminated = False
        shapes = bs.tools.areafilter.basic_shapes
        line_ac = Path(np.array([[self.lat, self.lon],[bs.traf.lat[0], bs.traf.lon[0]]]))
        line_sink = Path(np.reshape(shapes['SINK'].coordinates, (len(shapes['SINK'].coordinates) // 2, 2)))
        line_restrict = Path(np.reshape(shapes['RESTRICT'].coordinates, (len(shapes['RESTRICT'].coordinates) // 2, 2)))

        if line_sink.intersects_path(line_ac):
            self.segment_reward += 10
            self.terminated = True

        if line_restrict.intersects_path(line_ac):
            self.segment_reward += -1
            self.terminated = True

    def _get_truncated(self):
        """
        Check to see if the aircraft is too far out of schiphol,
        penalizes if the distance is further than 5% and truncates if more than 15%
        """
        dis_origin = bs.tools.geo.kwikdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0])*NM2KM
        if dis_origin > MAX_DISTANCE*1.05:
            self.truncated = False
            self.segment_reward += -1
        if dis_origin > MAX_DISTANCE*1.15:
            self.truncated = True
        return self.truncated
