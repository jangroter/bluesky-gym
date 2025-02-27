
"""
This table enumerates the observation space for 5 neigbours in observation space,
if number of neighbours included is changed, indices will change aswell (of course):

| Index: [start, end) | Description                                                  |   Values    |
|:-----------------:|------------------------------------------------------------|:---------------:|
|          0          | x coordinate ownship                                         | [-inf, inf] |
|          1          | y coordinate ownship                                         | [-inf, inf] |
|         2-6         | x coordinates intruders                                      | [-inf, inf] |
|         7-11        | y coordinates intruders                                      | [-inf, inf] |


"""

import functools
from pettingzoo import ParallelEnv
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

POPULATION_WEIGHT = -1.0
PATH_LENGTH_WEIGHT = -0.0025

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

MAX_DIS_NEXT_WPT = 15 #km
MIN_DIS_NEXT_WPT = 15 #km

# constants in this environment
SPEED = 125 # m/s
ALTITUDE = 3000 # m
SIM_DT = 5 # s
ACTION_TIME = 120

ACTION_FREQUENCY = int(ACTION_TIME / SIM_DT)
NUM_AC_STATE = 5

DISTANCE_MARGIN = 4.5 # km

class PathPlanningEnv(ParallelEnv):
    """ 
    Single agent path planning environment based on simple states ([x,y]).
    Penalized for path length and population exposed to the noise of the aircraft. 
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 1000}

    def __init__(self, render_mode=None, n_agents=10, time_limit=500, runway="27"):
        self.runway = runway
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.observation_spaces = {agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2+2*NUM_AC_STATE,), dtype=np.float64) for agent in self.agents}
        self.action_spaces = {agent: gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64) for agent in self.agents}        

        self.n_agents = n_agents
        self.agents = self._get_agents(self.n_agents)
        self.time_limit = time_limit

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
        self.segment_reward = [0] * self.n_agents

        self.terminated = False
        self.truncated = False
        self.wpt_reach = False

        for agent, idx in zip(self.agents,np.arange(self.n_agents)):
            spawn_lat, spawn_lon, spawn_heading = self._get_spawn()
            bs.traf.cre(agent,'a320',spawn_lat,spawn_lon,spawn_heading,ALTITUDE,SPEED)
            acrte = Route._routes.get(agent)
            acrte.delrte(idx)

            bs.traf.ap.setdest(idx,'EHAM')
            bs.traf.ap.setLNAV(idx, True)
            bs.traf.ap.route[idx].addwptMode(idx,'FLYOVER')

        self.lat = bs.traf.lat.copy()
        self.lon = bs.traf.lon.copy()

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, actions):
        """
        this executes the action and progresses the environment till a new action is required
        main MDP logic is contained here
        """
        self.segment_reward = [0] * self.n_agents
        self._set_action(actions)

        #TODO: make terminated and truncated in MA format, quit when 1 ac terminates or all truncate1
        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            self._update_reward()
            terminated = self._get_terminated()
            truncated = self._get_truncated()
            if any(terminated.values()):
                # makes sure the episode stops when one agent finishes and start new episode
                truncs = [True] * self.n_agents
                truncated = {
                    a: t
                    for a,t in zip(self.agents,truncs)
                }
                break
            if any(truncated.values()):
                break
            if self.render_mode == "human":
                self._render_frame()
        observation = self._get_observation()
        reward = self._get_reward()
        self.total_reward += sum(reward.values())
        info = self._get_info()

        # important step to reset the agents, mandatory by pettingzoo API
        if any(terminated.values()) or all(truncated.values()):
            self.agents = []

        return observation, reward, terminated, truncated, info

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent] # have to define observation_spaces & action_spaces, probably in init

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def render(self):
        pass

    def close(self):
        pass

    def _get_observation(self):
        """
        Observation is the normalized x and y coordinate of the aircraft

        TODO: make this part of code more efficient, only generating the x and y list once instead of every loop
        """
        obs = []
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            brg, dis = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])

            x = np.sin(np.radians(brg))*dis*NM2KM / MAX_DISTANCE
            y = np.cos(np.radians(brg))*dis*NM2KM / MAX_DISTANCE

            distances = bs.tools.geo.kwikdist_matrix(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat,bs.traf.lon)
            ac_idx_by_dist = np.argsort(distances) # sort aircraft by distance to ownship

            x_int = np.array([])
            y_int = np.array([])
            for i in range(self.n_agents):
                int_idx = ac_idx_by_dist[i]
                if int_idx == ac_idx:
                    continue
                brg, dis = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
                x_int = np.append(x_int, np.sin(np.radians(brg))*dis*NM2KM / MAX_DISTANCE)
                y_int = np.append(y_int, np.cos(np.radians(brg))*dis*NM2KM / MAX_DISTANCE)
                
            observation = {
                "x" : np.array([x]),
                "y" : np.array([y]),
                "x_int" : x_int[:NUM_AC_STATE],
                "y_int" : y_int[:NUM_AC_STATE]
            }
            obs.append(np.concatenate(list(observation.values())))

        observations = {
            a: o
            for a, o in zip(self.agents, obs)
        }
        
        return observations
    
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
        rewards = {
            a: r
            for a, r in zip(self.agents, self.segment_reward)
        }
        return rewards

    def _update_reward(self):
        """
        Updates the reward in accordance with the segment cost associated with population exposure
        and length of the path segment (as travelled through the air, to correct for wind)

        TODO: add intrusion component for the multi-agent to have a true effect
        """
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            path_length = self._get_path_length(ac_idx) * self.path_length_weight
            population_exposure = self._get_population_exposure(ac_idx) * self.population_weight 

            self.average_path += path_length
            self.average_noise += population_exposure

            self.segment_reward[ac_idx] += path_length + population_exposure

    def _get_path_length(self, ac_idx):
        """
        Use deadreckoning on the current True Airspeed of the aircraft to approximate
        the distance travelled through the air.
        Note that this does not have to correspond to the groundspeed and hence the distance 
        covered over ground due to altitude and wind effects.

        By doing this we favour flying in regions with tailwind.
        """
        dist = bs.traf.tas[ac_idx] * SIM_DT / 1852.
        return dist

    def _get_population_exposure(self, ac_idx):
        """
        Calculates the population exposed to the aircraft, scaled with the inverse square 
        of the distance. Inverse square of the distance is based on the inverse square law
        of noise dissipation.

        This function assumes that 2 people exposed to 500 units of noise would be equivalent to 
        1 person being exposed to 1000 units of noise. Can be replaced by a more accurate noise
        cost function, but is deemed enough showcasing this environment. 
        """
        brg, dist = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])

        x = np.sin(np.radians(brg))*dist*NM2M
        y = np.cos(np.radians(brg))*dist*NM2M
        z = bs.traf.alt[ac_idx]
        
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

        shapes = bs.tools.areafilter.basic_shapes
        line_sink = Path(np.reshape(shapes['SINK'].coordinates, (len(shapes['SINK'].coordinates) // 2, 2)))
        line_restrict = Path(np.reshape(shapes['RESTRICT'].coordinates, (len(shapes['RESTRICT'].coordinates) // 2, 2)))

        done = []
        for agent in self.agents:
            acidx = bs.traf.id2idx(agent)
            line_ac = Path(np.array([[self.lat[acidx], self.lon[acidx]],[bs.traf.lat[acidx], bs.traf.lon[acidx]]]))

            if line_sink.intersects_path(line_ac):
                self.segment_reward[acidx] += 10
                done.append(True)

            elif line_restrict.intersects_path(line_ac):
                self.segment_reward[acidx] += -1
                done.append(True)
            
            else:
                done.append(False)

        dones = {
            a: d
            for a,d in zip(self.agents,done)
        }
        return dones

    def _get_truncated(self):
        """
        Check to see if the aircraft is too far out of schiphol,
        penalizes if the distance is further than 5%
        """
        truncs = []
        for agent in self.agents:
            acidx = bs.traf.id2idx(agent)
            dis_origin = bs.tools.geo.kwikdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[acidx], bs.traf.lon[acidx])*NM2KM
            if dis_origin > MAX_DISTANCE*1.05:
                self.segment_reward[acidx] += -1
                truncs.append(False)
            else:
                truncs.append(False)
        truncated = {
            a: t
            for a,t in zip(self.agents,truncs)
        }
        return truncated

    def _set_action(self,actions):
        """
        transforms action to a waypoint based on current position of the aircraft and specified action
        """
        for agent in self.agents:
            ac_idx = bs.traf.id2idx(agent)
            action = actions[agent]
            bearing = np.rad2deg(np.arctan2(action[0],action[1]))
            bs.traf.ap.selhdgcmd(0,bearing)
            

    def _set_terminal_conditions(self):
        """
        Creates the terminal conditions surrounding the FAF to ensure correct approach angle

        If render mode is set to 'human' also already creates the required elements for plotting 
        these terminal conditions in the rendering window
        """
        num_points = 36 # number of straight line segments that make up the circle

        faf_lat, faf_lon = fn.get_point_at_distance(RUNWAYS_SCHIPHOL_FAF[self.runway]['lat'],
                                                    RUNWAYS_SCHIPHOL_FAF[self.runway]['lon'],
                                                    FAF_DISTANCE,
                                                    RUNWAYS_SCHIPHOL_FAF[self.runway]['track']-180)
        
        # Compute bounds for the merge angles from FAF
        cw_bound = ((RUNWAYS_SCHIPHOL_FAF[self.runway]['track']-180+ 360)%360) + (IAF_ANGLE/2)
        ccw_bound = ((RUNWAYS_SCHIPHOL_FAF[self.runway]['track']-180+ 360)%360) - (IAF_ANGLE/2)

        angles = np.linspace(cw_bound,ccw_bound,num_points)
        lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)

        command = 'POLYLINE SINK'
        for i in range(0,len(lat_iaf)):
            command += ' '+str(lat_iaf[i])+' '
            command += str(lon_iaf[i])
        bs.stack.stack(command)
    
        bs.stack.stack(f'POLYLINE RESTRICT {lat_iaf[0]} {lon_iaf[0]} {faf_lat} {faf_lon} {lat_iaf[-1]} {lon_iaf[-1]}')
        bs.stack.stack('COLOR RESTRICT red')

        if self.render_mode == 'human':
            env_max_distance = np.sqrt((MAX_DISTANCE)**2 + (MAX_DISTANCE)**2) #km
            lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(SCHIPHOL[0], SCHIPHOL[1], 315, env_max_distance/NM2KM)
            self.screen_coords = [lat_ref_point,lon_ref_point]
            coordinates = np.empty(2 * 36, dtype=np.float32)  # Create empty array
            coordinates[0::2] = lat_iaf  # Fill array lat0,lon0,lat1,lon1....
            coordinates[1::2] = lon_iaf

            line_arc = np.reshape(coordinates, (len(coordinates) // 2, 2))
            line_restrict = np.array([[lat_iaf[0],lon_iaf[0]],[faf_lat, faf_lon],[lat_iaf[-1], lon_iaf[-1]]])

            # Convert all coordinates to Pygame window reference frame
            qdr, dis = bs.tools.geo.kwikqdrdist(self.screen_coords[0], self.screen_coords[1], line_arc[:,0], line_arc[:,1])
            dis = dis*NM2KM
            x_arc = ((np.sin(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
            y_arc = ((-np.cos(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
            line_arc_pg = list(zip(x_arc, y_arc))

            self.line_arc_pg = [(float(x), float(y)) for x, y in line_arc_pg]

            qdr, dis = bs.tools.geo.kwikqdrdist(self.screen_coords[0], self.screen_coords[1], line_restrict[:,0], line_restrict[:,1])
            dis = dis*NM2KM
            x_restrict = ((np.sin(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
            y_restrict = ((-np.cos(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
            line_restrict_pg = list(zip(x_restrict, y_restrict))

            self.line_restrict_pg = [(float(x), float(y)) for x, y in line_restrict_pg]
    
    def _get_agents(self, n_agents):
        return [f'kl00{i+1}'.upper() for i in range(n_agents)]

    def _get_spawn(self):
        """
        Determine the random spawn conditions for the aircraft during training,
        uses a random uniform spawn distance, because of this, area further away from the
        destination are explored less, due to area scaling quadratically with the radius/distance

        Could be interesting to build a toggle for different spawn functions.
        """
        spawn_bearing = np.random.uniform(0,360)
        spawn_distance = max(np.random.uniform(0,0.9)*MAX_DISTANCE,
                             MIN_DISTANCE)
        spawn_lat, spawn_lon = fn.get_point_at_distance(SCHIPHOL[0],SCHIPHOL[1],spawn_distance,spawn_bearing)
        spawn_heading = (spawn_bearing + 180 + 360)%360

        return spawn_lat, spawn_lon, spawn_heading

    def _render_frame(self):
        """
        handles rendering, needs to be fixed to only render the background surface once instead of on each frame.
        """

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # pygame.draw.line(canvas,
        #     (235, 52, 52),
        #     (x_actor, y_actor),
        #     (x_actor+heading_end_x, y_actor-heading_end_y),
        #     width = 5
        # )

        # Create background image from population data
        # pop_array = np.genfromtxt('bluesky_gym/envs/data/population_1km.csv', delimiter = ' ')
        # x_index_min = int(((500000)/1000)-(MAX_DISTANCE))
        # x_index_max = int(((500000)/1000)+(MAX_DISTANCE))
        # y_index_min = int(((500000)/1000)-(MAX_DISTANCE))
        # y_index_max = int(((500000)/1000)+(MAX_DISTANCE))

        # pop_array = pop_array[y_index_min:y_index_max,x_index_min:x_index_max]
        # c_map = cm.get_cmap("Blues").copy()
        # c_map.set_bad((0.8,0.8,0.9))
        # norm=LogNorm(vmin=100,vmax=100000)

        # normalized_array = norm(pop_array)

        # colored_array = c_map(normalized_array)[:, :, :3]  # Apply colormap and drop alpha channel
        # colored_array = (colored_array * 255).astype(np.uint8)  # Convert to 0-255 for Pygame

        # # Convert the array to a Pygame surface
        # surface = pygame.surfarray.make_surface(np.transpose(colored_array, (1, 0, 2)))
        surface = pygame.Surface(self.window_size)
        surface.fill((255,255,255))
        # Scale the surface to fit the screen
        surface = pygame.transform.scale(surface, (self.window_width, self.window_height))

        # Draw the FAF and IAF
        pygame.draw.lines(surface, (0, 0, 0), False, self.line_arc_pg, 3)
        pygame.draw.lines(surface, (255, 0, 0), False, self.line_restrict_pg, 2)

        ### PLOTTING OF THE AIRCRAFT AND WAYPOINTS ###
        ac_lat, ac_lon = bs.traf.lat[0], bs.traf.lon[0]

        qdr, dis = bs.tools.geo.kwikqdrdist(self.screen_coords[0], self.screen_coords[1], ac_lat, ac_lon)
        dis = dis*NM2KM
        x_ac = ((np.sin(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
        y_ac = ((-np.cos(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width

        pygame.draw.circle(surface, (0,0,0), (x_ac,y_ac),5)

        wpt_lat, wp_lon = bs.traf.actwp.lat[0], bs.traf.actwp.lon[0]
        qdr, dis = bs.tools.geo.kwikqdrdist(self.screen_coords[0], self.screen_coords[1], wpt_lat, wp_lon)
        dis = dis*NM2KM
        x_wpt = ((np.sin(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
        y_wpt = ((-np.cos(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width

        pygame.draw.circle(surface, (255,0,0), (x_wpt,y_wpt),5)

        self.window.blit(surface, (0,0))
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

