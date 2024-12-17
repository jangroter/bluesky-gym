import numpy as np
import pygame

import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.path import Path

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import bluesky.traffic.autopilot 
from bluesky.traffic import Route

import gymnasium as gym
from gymnasium import spaces

"""
Environment created for doing research in PathPlanning with RL, for now trying
to implement the simplest version possible, but should also become a test bed
for providing custom observations and action space wrappers.

Additionally, this environment will need the functionality of using scenario-files
for testing.
"""

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

MIN_SPAWN_DISTANCE = FAF_DISTANCE + IAF_DISTANCE

MAX_DIS_NEXT_WPT = 30 #km
MIN_DIS_NEXT_WPT = 5 #km

# constants in this environment
SPEED = 125 # m/s
ALTITUDE = 3000 # m

DISTANCE_MARGIN = 4.5 # km

class PathPlanning2DEnv(gym.Env):
    """ 
    TODO:
    write this
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

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
        bs.stack.stack('DT 1;FF')

        # load the data used for this environment
        self.pop_array = np.genfromtxt('bluesky_gym/envs/data/population_1km.csv', delimiter = ' ')
        self.x_array = np.genfromtxt('bluesky_gym/envs/data/x_array.csv', delimiter = ' ')
        self.y_array = np.genfromtxt('bluesky_gym/envs/data/y_array.csv', delimiter = ' ')
        self.x_max = np.max(self.x_array)
        self.y_max = np.max(self.y_array)
        self.cell_size = 1000 # distance per pixel in pop_array, in meters
        self.projection_size = 20 # distance in km that noise is projected down, similar to kernel size in CNN

        # initialize values used for logging -> input in _get_info
        self.segment_reward = 0
        self.total_reward = 0
        self.segment_noise = 0
        self.total_noise = 0
        self.segment_length = 0
        self.total_length = 0

        self.population_weight = 0
        self.path_length_weight = -0.05

        self.average_noise = 0
        self.average_path = 0

        self.wpt_reach = False
        self.terminated = False
        self.truncated = False

        self.lat = 0
        self.lon = 0
        
        self.new_wpt_set = True

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


    def _get_obs(self):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """

        brg, dis = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0])

        x = np.sin(np.radians(brg))*dis*NM2M / np.max(self.x_array)
        y = np.cos(np.radians(brg))*dis*NM2M / np.max(self.y_array)

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
            "total_reward": self.total_reward
        }
    
    def _get_reward(self):
        """
        Reward is based on pathlength and population exposure and is a weighted bicriterion problem
        Additional rewards/penalties are included for satisfying terminal conditions
        see: _update_reward(), _get_terminated()
        """
        return self.segment_reward

    def _update_reward(self):
        path_length = self._get_path_length()
        population_exposure = self._get_population_exposure()

        self.average_path += path_length
        self.average_noise += population_exposure

        self.segment_reward += path_length * self.path_length_weight + population_exposure * self.population_weight
    
    def _update_wpt_reach(self):
        acrte = Route._routes.get('kl001')
        wpt_dis = bs.tools.geo.kwikdist(bs.traf.lat[0], bs.traf.lon[0], acrte.wplat[0], acrte.wplon[0])*NM2KM
        # import code
        # code.interact(local=locals())
        if wpt_dis < DISTANCE_MARGIN:
            self.wpt_reach = True
            acrte.delrte(0)
            # bs.stack.stack(f"DELWPT kl001 {bs.traf.ap.route.wpname[0]}")
            # import code
            # code.interact(local=locals())
        else:
            self.wpt_reach = False

    def _get_path_length(self):
        dist = bs.tools.geo.kwikdist(self.lat, self.lon, bs.traf.lat[0], bs.traf.lon[0])
        self.lat = bs.traf.lat[0]
        self.lon = bs.traf.lon[0]
        return dist

    def _get_population_exposure(self):
        brg, dist = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0])

        x = np.sin(np.radians(brg))*dist
        y = np.cos(np.radians(brg))*dist
        z = bs.traf.alt[0]
        
        x_index_min = int(((x+self.x_max)/self.cell_size)-self.projection_size)
        x_index_max = int(((x+self.x_max)/self.cell_size)+self.projection_size)
        y_index_min = int(((self.y_max - y)/self.cell_size)-self.projection_size)
        y_index_max = int(((self.y_max - y)/self.cell_size)+self.projection_size)

        distance2 = (self.x_array[y_index_min:y_index_max,x_index_min:x_index_max]-x)**2 + (self.y_array[y_index_min:y_index_max,x_index_min:x_index_max]-y)**2 + z**2

        return np.sum(self.pop_array[y_index_min:y_index_max,x_index_min:x_index_max]/distance2)

    def _get_terminated(self):
        # TODO: Make this interect path a function in areafilter with PR
        self.terminated = False
        shapes = bs.tools.areafilter.basic_shapes
        line_ac = Path(np.array([[self.lat, self.lon],[bs.traf.lat[0], bs.traf.lon[0]]]))
        line_sink = Path(np.reshape(shapes['SINK'].coordinates, (len(shapes['SINK'].coordinates) // 2, 2)))
        line_restrict = Path(np.reshape(shapes['RESTRICT'].coordinates, (len(shapes['RESTRICT'].coordinates) // 2, 2)))

        if line_sink.intersects_path(line_ac):
            self.segment_reward += 5
            self.terminated = True

        if line_restrict.intersects_path(line_ac):
            self.segment_reward += -1
            self.terminated = True

        return self.terminated

    def _get_truncated(self):
        dis_origin = bs.tools.geo.kwikdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[0], bs.traf.lon[0])*NM2KM
        if dis_origin > (np.max(self.x_array)/1000.)-self.projection_size:
            self.truncated = False
            self.segment_reward += -1
        return self.truncated

    def _get_action(self,action):
        """
        transforms action to a waypoint based on current position of the aircraft and specified action
        """
        acid = bs.traf.id[0]
        acrte = Route._routes.get(acid)

        distance = max(max(action[0]*MAX_DIS_NEXT_WPT, action[1]*MAX_DIS_NEXT_WPT), MIN_DIS_NEXT_WPT)
        bearing = np.deg2rad(np.arctan2(action[0],action[1]))

        ac_lat = bs.traf.lat[0]
        ac_lon = bs.traf.lon[0]

        new_lat, new_lon = fn.get_point_at_distance(ac_lat, ac_lon, distance, bearing)
        wptype  = Route.wplatlon

        acrte.addwpt_simple(0, 'waypoint', wptype, new_lat, new_lon)
        acrte.calcfp()
        acrte.direct(0, acrte.wpname[-1])
        # bs.stack.stack(f'ADDWPT {acid} {new_lat} {new_lon}')
        bs.stack.stack(f'LNAV {acid} ON')
        self.new_wpt_set = True

    def _set_terminal_conditions(self):
        """
        creates the terminal conditions surrounding the FAF to ensure correct approach angle
        """

        r_earth = 6371000.0 # radius of the Earth [m]
        num_points = 36 # number of straight line segments that make up the circrle

        faf_lat, faf_lon = fn.get_point_at_distance(RUNWAYS_SCHIPHOL_FAF[self.runway]['lat'],
                                                    RUNWAYS_SCHIPHOL_FAF[self.runway]['lon'],
                                                    FAF_DISTANCE,
                                                    RUNWAYS_SCHIPHOL_FAF[self.runway]['track']-180)
        
        # Compute flat Earth correction at the FAF
        coslatinv = 1.0 / np.cos(np.deg2rad(faf_lat))

        # Compute bounds for the merge angles from FAF
        cw_bound = ((RUNWAYS_SCHIPHOL_FAF[self.runway]['track']-180+ 360)%360) + (IAF_ANGLE/2)
        ccw_bound = ((RUNWAYS_SCHIPHOL_FAF[self.runway]['track']-180+ 360)%360) - (IAF_ANGLE/2)

        angles = np.linspace(cw_bound,ccw_bound,num_points)

        # Calculate the iaf coordinates in lat/lon degrees. Is an approximation of an arg starting routing to FAF
        # Use flat-earth approximation to convert from cartesian to lat/lon.
        # lat_iaf = faf_lat + np.rad2deg(IAF_DISTANCE * 1000 * np.sin(angles) / r_earth)  # [deg]
        # lon_iaf = faf_lon + np.rad2deg(IAF_DISTANCE * 1000 * np.cos(angles) * coslatinv / r_earth)  # [deg]
        lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)
        # make the data array in the format needed to plot circle
        # Coordinates = np.empty(2 * num_points, dtype=np.float32)  # Create empty array
        # Coordinates[0::2] = lat_iaf  # Fill array lat0,lon0,lat1,lon1....
        # Coordinates[1::2] = lon_iaf

        command = 'POLYLINE SINK'
        for i in range(0,len(lat_iaf)):
            command += ' '+str(lat_iaf[i])+' '
            command += str(lon_iaf[i])
        bs.stack.stack(command)
    
        bs.stack.stack(f'POLYLINE RESTRICT {lat_iaf[0]} {lon_iaf[0]} {faf_lat} {faf_lon} {lat_iaf[-1]} {lon_iaf[-1]}')
        bs.stack.stack('COLOR RESTRICT red')

        # IF RENDER HUMAN, MAYBE ADD TO RENDERING SURFACE HERE NOW ASWELL

    def _get_spawn(self):
        spawn_bearing = np.random.uniform(0,360)
        spawn_distance = max(np.random.uniform(0,0.1)*((np.max(self.x_array)/1000.)-self.projection_size),
                             MIN_SPAWN_DISTANCE)
        spawn_lat, spawn_lon = fn.get_point_at_distance(SCHIPHOL[0],SCHIPHOL[1],spawn_distance,spawn_bearing)
        spawn_heading = (spawn_bearing + 180 + 360)%360

        return spawn_lat, spawn_lon, spawn_heading

    def reset(self, seed=None, options=None):
        """
        resets environment after getting a done flag through terminal condition or time-out
        should return observation and contain spawning logic
        """
        super().reset(seed=seed)
        bs.traf.reset()

        # print('noise:', self.average_noise)
        # print('path:', self.average_path)

        self.average_noise = 0
        self.average_path = 0

        self.total_reward = 0

        self.terminated = False
        self.truncated = False

        spawn_lat, spawn_lon, spawn_heading = self._get_spawn()
        bs.traf.cre('kl001','a320',spawn_lat,spawn_lon,spawn_heading,ALTITUDE,SPEED)
        acrte = Route._routes.get('kl001')
        acrte.delrte(0)
        bs.stack.stack(f'LNAV kl001 ON')
        bs.stack.stack(f"DEST kl001 {RUNWAYS_SCHIPHOL_FAF[self.runway]['lat']} {RUNWAYS_SCHIPHOL_FAF[self.runway]['lon']}")
        bs.stack.stack(f'ADDWPTMODE kl001 FLYOVER')
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
        self._get_action(action)
        self._update_wpt_reach()
        self.segment_reward = 0
        while not self.wpt_reach and not self.terminated and not self.truncated:
            bs.sim.step()
            terminated = self._get_terminated()
            truncated = self._get_truncated()
            self._update_reward()
            self._update_wpt_reach()
            a = bs.traf.ap.route[0]
            b = bs.traf.ap
            # print(bs.traf.ap.route[0])
            # import code
            # code.interact(local=locals())
            if self.render_mode == "human":
                self._render_frame()
        
        observation = self._get_obs()
        reward = self._get_reward()
        self.total_reward += reward
        info = self._get_info()
        # if self.total_reward == 0:
        #     import code
        #     code.interact(local=locals())
        return observation, reward, terminated, truncated, info
    
    def check_past_wpt(self):
        acrte = Route._routes.get('kl001')
        wptlat, wptlon = acrte.wplat[0], acrte.wplon[0]
        dis = bs.tools.geo.kwikdist(bs.traf.lat[0], bs.traf.lon[0], wptlat, wptlon)
        if self.new_wpt_set:
            self.wptdist = dis
            self.wptdist_old = dis
            self.new_wpt_set = False
        else:
            if self.wptdist - dis < 0 and self.wptdist_old - self.wptdist > 0:
                self.wpt_reach = True
            if self.wptdist - dis < 0 and self.wptdist_old - self.wptdist < 0:
                wptlat = bs.traf.actwp.lat[0]
                wptlon = bs.traf.actwp.lon[0]
                wptype  = Route.wplatlon
                acrte.addwpt_simple(0, 'waypoint', wptype, wptlat, wptlon)
                acrte.calcfp()
                acrte.direct(0, acrte.wpname[-1])
                bs.stack.stack('LNAV kl001 ON')

            self.wptdist_old = self.wptdist
            self.wptdist = dis
            
        self.wpt_reach = False

    def render(self):
        pass

    def _render_frame(self):
        """
        handles rendering
        """

        # Set up the Pygame window
        screen = pygame.display.set_mode((self.window_width, self.window_height))

        # Set screen coordinates
        env_max_distance = np.sqrt((DISTANCE_ENV/2)**2 + (DISTANCE_ENV/2)**2) #km
        lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(SCHIPHOL[0], SCHIPHOL[1], 315, env_max_distance/NM2KM)
        screen_coords = [lat_ref_point,lon_ref_point]

        # Determine coordinates of FAF and IAF in latlon reference frame
        faf_lat, faf_lon = fn.get_point_at_distance(RUNWAY[0],
                                                    RUNWAY[1],
                                                    FAF_DISTANCE,
                                                    RUNWAY_TRACK-180)

        # Compute bounds for the merge angles from FAF
        cw_bound = ((RUNWAY_TRACK-180+ 360)%360) + (IAF_ANGLE/2)
        ccw_bound = ((RUNWAY_TRACK-180+ 360)%360) - (IAF_ANGLE/2)
        angles = np.linspace(cw_bound,ccw_bound,36)
        lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)

        coordinates = np.empty(2 * 36, dtype=np.float32)  # Create empty array
        coordinates[0::2] = lat_iaf  # Fill array lat0,lon0,lat1,lon1....
        coordinates[1::2] = lon_iaf

        line_arc = np.reshape(coordinates, (len(coordinates) // 2, 2))
        line_restrict = np.array([[lat_iaf[0],lon_iaf[0]],[faf_lat, faf_lon],[lat_iaf[-1], lon_iaf[-1]]])


        # Convert all coordinates to Pygame window reference frame
        qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], line_arc[:,0], line_arc[:,1])
        dis = dis*NM2KM
        x_arc = ((np.sin(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*self.window_width
        y_arc = ((-np.cos(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*self.window_width

        line_arc_pg = list(zip(x_arc, y_arc))
        line_arc_pg = [(float(x), float(y)) for x, y in line_arc_pg]

        qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], line_restrict[:,0], line_restrict[:,1])
        dis = dis*NM2KM
        x_restrict = ((np.sin(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*self.window_width
        y_restrict = ((-np.cos(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*self.window_width

        line_restrict_pg = list(zip(x_restrict, y_restrict))
        line_restrict_pg = [(float(x), float(y)) for x, y in line_restrict_pg]
        import code
        code.interact(local=locals())

        # pygame.draw.line(canvas,
        #     (235, 52, 52),
        #     (x_actor, y_actor),
        #     (x_actor+heading_end_x, y_actor-heading_end_y),
        #     width = 5
        # )

        # Create background image from population data
        pop_array = np.genfromtxt('bluesky_gym/envs/data/population_1km.csv', delimiter = ' ')
        x_index_min = int(((500000)/1000)-(DISTANCE_ENV/2))
        x_index_max = int(((500000)/1000)+(DISTANCE_ENV/2))
        y_index_min = int(((500000)/1000)-(DISTANCE_ENV/2))
        y_index_max = int(((500000)/1000)+(DISTANCE_ENV/2))

        x_array = np.genfromtxt('bluesky_gym/envs/data/x_array.csv',delimiter=' ')
        y_array = np.genfromtxt('bluesky_gym/envs/data/y_array.csv',delimiter=' ')

        pop_array = pop_array[y_index_min:y_index_max,x_index_min:x_index_max]
        c_map = cm.get_cmap("Blues").copy()
        c_map.set_bad((0.8,0.8,0.9))
        norm=LogNorm(vmin=100,vmax=100000)

        normalized_array = norm(pop_array)

        colored_array = c_map(normalized_array)[:, :, :3]  # Apply colormap and drop alpha channel
        colored_array = (colored_array * 255).astype(np.uint8)  # Convert to 0-255 for Pygame

        # Convert the array to a Pygame surface
        surface = pygame.surfarray.make_surface(np.transpose(colored_array, (1, 0, 2)))

        # Scale the surface to fit the screen
        surface = pygame.transform.scale(surface, (self.window_width, self.window_height))

        # Draw the FAF and IAF
        pygame.draw.lines(surface, (0, 0, 0), False, line_arc_pg, 3)
        pygame.draw.lines(surface, (255, 0, 0), False, line_restrict_pg, 2)

        screen.blit(surface, (0,0))
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

        import code
        code.interact(local=locals())
        pass

    def close(self):
        pass