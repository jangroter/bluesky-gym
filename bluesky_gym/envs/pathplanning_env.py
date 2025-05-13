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

FAF_DISTANCE = 25 #km
IAF_DISTANCE = 30 #km, from FAF

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

DISTANCE_MARGIN = 4.5 # km


class PathPlanningEnv(gym.Env):
    """ 
    Single agent path planning environment based on simple states ([x,y]).
    Penalized for path length and population exposed to the noise of the aircraft. 
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 1000}

    def __init__(self, render_mode=None, runway=["27"], action_mode="wpt"):
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
        
        self.action_mode = action_mode
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
        self._set_action(action)

        if self.action_mode == 'wpt':
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
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        elif self.action_mode == 'hdg':
            for _ in range(ACTION_FREQUENCY):
                bs.sim.step()
                self._update_reward()
                terminated = self._get_terminated()
                truncated = self._get_truncated()
                if terminated:
                    break
                if truncated:
                    break
                if self.render_mode == "human":
                    self._render_frame()
            observation = self._get_obs()
            reward = self._get_reward()
            self.total_reward += reward
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
        for rwy in self.runway:
            shapes = bs.tools.areafilter.basic_shapes
            line_ac = Path(np.array([[self.lat, self.lon],[bs.traf.lat[0], bs.traf.lon[0]]]))
            line_sink = Path(np.reshape(shapes[f'SINK{rwy}'].coordinates, (len(shapes[f'SINK{rwy}'].coordinates) // 2, 2)))
            line_restrict = Path(np.reshape(shapes[f'RESTRICT{rwy}'].coordinates, (len(shapes[f'RESTRICT{rwy}'].coordinates) // 2, 2)))

            if line_sink.intersects_path(line_ac):
                self.segment_reward += 10
                self.terminated = True

            if line_restrict.intersects_path(line_ac):
                self.segment_reward += -1
                self.terminated = True
        
        self.lat = bs.traf.lat[0]
        self.lon = bs.traf.lon[0]

        return self.terminated

    def _get_truncated(self):
        """
        Check to see if the aircraft is too far out of schiphol,
        penalizes if the distance is further than 5%
        """
        dis_origin = bs.tools.geo.kwikdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0])*NM2KM
        if dis_origin > MAX_DISTANCE*1.05:
            self.truncated = False
            self.segment_reward += -1
        return self.truncated

    def _set_action(self,action):
        """
        transforms action to a waypoint based on current position of the aircraft and specified action
        """
        if self.action_mode == 'wpt':
            distance = max(max(abs(action[0]*MAX_DIS_NEXT_WPT), abs(action[1]*MAX_DIS_NEXT_WPT)), MIN_DIS_NEXT_WPT)
            bearing = np.rad2deg(np.arctan2(action[0],action[1]))

            ac_lat = bs.traf.lat[0]
            ac_lon = bs.traf.lon[0]

            new_lat, new_lon = fn.get_point_at_distance(ac_lat, ac_lon, distance, bearing)
            bs.traf.ap.route[0].addwptStack(0, f'{new_lat}, {new_lon}')
        elif self.action_mode == 'hdg':
            bearing = np.rad2deg(np.arctan2(action[0],action[1]))
            bs.traf.ap.selhdgcmd(0,bearing)
            
    def _set_terminal_conditions(self):
        """
        Creates the terminal conditions surrounding the FAF to ensure correct approach angle

        If render mode is set to 'human' also already creates the required elements for plotting 
        these terminal conditions in the rendering window
        """
        self.line_arc_pg = []
        self.line_restrict_pg = []

        for rwy in self.runway:
            num_points = 36 # number of straight line segments that make up the circle

            faf_lat, faf_lon = fn.get_point_at_distance(RUNWAYS_SCHIPHOL_FAF[rwy]['lat'],
                                                        RUNWAYS_SCHIPHOL_FAF[rwy]['lon'],
                                                        FAF_DISTANCE,
                                                        RUNWAYS_SCHIPHOL_FAF[rwy]['track']-180)
            
            # Compute bounds for the merge angles from FAF
            cw_bound = ((RUNWAYS_SCHIPHOL_FAF[rwy]['track']-180+ 360)%360) + (IAF_ANGLE/2)
            ccw_bound = ((RUNWAYS_SCHIPHOL_FAF[rwy]['track']-180+ 360)%360) - (IAF_ANGLE/2)

            angles = np.linspace(cw_bound,ccw_bound,num_points)
            lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)

            command = f'POLYLINE SINK{rwy}'
            for i in range(0,len(lat_iaf)):
                command += ' '+str(lat_iaf[i])+' '
                command += str(lon_iaf[i])
            bs.stack.stack(command)
    
            bs.stack.stack(f'POLYLINE RESTRICT{rwy} {lat_iaf[0]} {lon_iaf[0]} {faf_lat} {faf_lon} {lat_iaf[-1]} {lon_iaf[-1]}')
            bs.stack.stack(f'COLOR RESTRICT{rwy} red')

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

                self.line_arc_pg.append([(float(x), float(y)) for x, y in line_arc_pg])

                qdr, dis = bs.tools.geo.kwikqdrdist(self.screen_coords[0], self.screen_coords[1], line_restrict[:,0], line_restrict[:,1])
                dis = dis*NM2KM
                x_restrict = ((np.sin(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
                y_restrict = ((-np.cos(np.deg2rad(qdr))*dis)/(MAX_DISTANCE*2))*self.window_width
                line_restrict_pg = list(zip(x_restrict, y_restrict))

                self.line_restrict_pg.append([(float(x), float(y)) for x, y in line_restrict_pg])

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
        for line_arc, line_restrict in zip(self.line_arc_pg,self.line_restrict_pg):
            pygame.draw.lines(surface, (0, 0, 0), False, line_arc, 3)
            pygame.draw.lines(surface, (255, 0, 0), False, line_restrict, 2)

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

