import numpy as np
import pygame
import bluesky as bs
import bluesky_gym.envs.common.functions as fn
import bluesky_gym.envs.common.deterministic_path_planning as path_plan
from bluesky.tools.aero import kts

import gymnasium as gym
from gymnasium import spaces

from core.tools import kwikqdrdist

from shapely.geometry import Polygon

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1 # reach set waypoint

DRIFT_PENALTY = -0.01
AC_INTRUSION_PENALTY = -5
RESTRICTED_AREA_INTRUSION_PENALTY = -5
SECTOR_EXIT_PENALTY = -4

WAYPOINT_DISTANCE_MIN = 180 # KM
WAYPOINT_DISTANCE_MAX = 400 # KM

OBSTACLE_DISTANCE_MIN = 20 # KM
OBSTACLE_DISTANCE_MAX = 150 # KM

OTHER_AC_DISTANCE_MIN = 50 # KM
OTHER_AC_DISTANCE_MAX = 170 # KM

D_HEADING = 45 #degrees
D_SPEED = 20/3 # m/s

AC_SPD = 150 # m/s (CAS - typical commercial airliner cruise value)
ALTITUDE = 350 # In FL

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10
## for obstacles generation
NUM_OBSTACLES = 5 #np.random.randint(1,5)
NUM_INTRUDERS = 5
INTRUSION_DISTANCE = 5 # NM

## number of waypoints coincides with the number of destinations for each aircraft (actor and all other aircraft)
NUM_WAYPOINTS = NUM_INTRUDERS + 1

OBSTACLE_AREA_RANGE = (50, 1000) # In NM^2
# Sector polygon area range
SECTOR_AREA_RANGE = (15000, 23001) # In NM^2

# CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates
CENTER = (52., 4.) # TU Delft AE Faculty coordinates

MAX_DISTANCE = 350 # width of screen in km

TOTAL_OBSERVATION_POINTS = 50 # Number of points to be observed along the sector polygon edges

class CompetitionEnv(gym.Env):
    """ 
    Static Obstacle Conflict Resolution Environment
    The environment simulates a 2D airspace where an ownship (the agent) must navigate to a series of waypoints while avoiding collisions with other aircraft and static obstacles.
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512 # pixels
        self.window_height = 512 # pixels
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        # Observation space should include ownship and intruder info, end destination info for ownship, relative position of obstcles in reference to ownships
        # Maybe later also have an option for CNN based intruder and obstacle info, could be interesting
        self.observation_space = spaces.Dict(
            {   
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "destination_waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_cos_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_sin_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "restricted_area_radius": spaces.Box(0, 1, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "restricted_area_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES, ), dtype=np.float64),
                "cos_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "sin_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "sector_points_distance": spaces.Box(-np.inf, np.inf, shape = (TOTAL_OBSERVATION_POINTS,), dtype=np.float64),
                "sector_points_cos_drift": spaces.Box(-np.inf, np.inf, shape = (TOTAL_OBSERVATION_POINTS,), dtype=np.float64),
                "sector_points_sin_drift": spaces.Box(-np.inf, np.inf, shape = (TOTAL_OBSERVATION_POINTS,), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.stack.stack('DT 1;FF')
        
        # variables for logging
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.exited_sector = 0
        self.average_drift = np.array([])

        self._generate_sector() # Create airspace polygon
        self.obstacle_names = []

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()
        self.counter = 0

        # reset logging variables 
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.exited_sector = 0
        self.average_drift = np.array([])
        self.impossible_route_counter = 0

        # bs.tools.areafilter.deleteArea(self.poly_name)

        # defining screen coordinates
        # defining the reference point as the top left corner of the SQUARE screen
        # from the initial position of the aircraft which is set to be the centre of the screen
        d = np.sqrt(2*(MAX_DISTANCE/2)**2) #KM
        lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(CENTER[0], CENTER[1], 315, d/NM2KM)
        
        self.screen_coords = [lat_ref_point,lon_ref_point]#[52.9, 2.6]

        bs.traf.cre('KL001',actype="A320", aclat = CENTER[0], aclon = CENTER[1], acspd=AC_SPD, acalt=ALTITUDE)
        ac_idx = bs.traf.id2idx('KL001')
        
        self.sample_entire_scenario = True
        while self.sample_entire_scenario:
            # generate obstacles and other aircraft until a valid (not more than two sectors overlap) scenario is found
            self.sample_obstacle = True
            while self.sample_obstacle:
                self._generate_obstacles()

            self._generate_other_aircraft()

            self._generate_waypoint()

            ac_idx = bs.traf.id2idx('KL001')
            self.initial_wpt_qdr, _ = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
            bs.traf.hdg[ac_idx] = self.initial_wpt_qdr
            bs.traf.ap.trk[ac_idx] = self.initial_wpt_qdr

            try:
                self._path_planning()
                self.sample_entire_scenario = False  # Success
            except Exception as e:
                if str(e) == "Impossible to find a route":
                    print("Impossible to find a route, resampling the scenario")
                    # import pickle
                    
                    # with open(f'de-bugging_obstacles/objs_impossible_route_automatic_saving_{self.impossible_route_counter}.pkl', 'wb') as f:
                    #     obj0 = self.other_aircraft_names
                    #     obj1 = bs.traf.lat
                    #     obj2 = bs.traf.lon
                    #     obj3 = bs.traf.alt
                    #     obj4 = bs.traf.tas
                    #     obj5 = self.wpt_lat
                    #     obj6 = self.wpt_lon
                    #     obj7 = self.obstacle_vertices
                    #     pickle.dump([obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7], f)
                    
                    for name in self.other_aircraft_names:
                        ac_idx = bs.traf.id2idx(name)
                        bs.traf.delete(ac_idx)
                    self.impossible_route_counter += 1
                    continue
                else:
                    raise  

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self.counter += 1
        self._get_action(action)

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
            reward, done, terminated = self._get_reward()
            if terminated:
                observation = self._get_obs()
                info = self._get_info()
                self.total_reward += reward
                return observation, reward, done, terminated, info

        observation = self._get_obs()
        reward, done, terminated = self._get_reward()
        self.total_reward += reward
        info = self._get_info()

        return observation, reward, done, terminated, info

    def _generate_sector(self):
        R = np.sqrt(SECTOR_AREA_RANGE[1] / np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        
        while p_area < SECTOR_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        self.poly_area = p_area

        self.poly_points = np.array(p) # Polygon vertices are saved in terms of NM

        p = [fn.nm_to_latlong(CENTER, point) for point in p] # Convert to lat/long coordinates
        self.poly_points_lat_lon = np.array(p) # Polygon vertices are saved in lat/lon coordinates

        points = [coord for point in p for coord in point] # Flatten the list of points
        # red(f'Polygon points: {p}')
        bs.tools.areafilter.defineArea('sector', 'POLY', points)

    def _get_observation_polygon_edges(self, vertices, total_points=20):
        """
        Interpolates evenly along polygon edges to generate exactly `total_points` total points,
        including original vertices.

        Args:
            vertices (np.ndarray): Polygon vertex coordinates, structures as [lat, lon].
            total_points (int): Desired total number of points (vertices + interpolated).

        Returns:
            all_points (np.ndarray): All interpolated + original points, structures as [lat, lon].
        """

        num_edges = len(vertices)
        points_to_interpolate = total_points - num_edges

        # How many points per edge?
        base_points = points_to_interpolate // num_edges
        extra_points = points_to_interpolate % num_edges # remainder

        all_points = []

        for i in range(num_edges):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % num_edges] # wrap around to first vertex

            # Distribute remainder: first `extra_points` edges get one more
            n_interp = base_points + (1 if i < extra_points else 0)

            # Include original vertex
            all_points.append(p1)

            # Interpolate (exclude p2 so that next edge includes it)
            for t in np.linspace(0, 1, n_interp + 2)[1:-1]:  # exclude 0 and 1
                lat = p1[0] + t * (p2[0] - p1[0])
                lon = p1[1] + t * (p2[1] - p1[1])
                all_points.append([lat, lon])

        return np.array(all_points)

    def _generate_other_aircraft(self, acid_actor = 'KL001', num_other_aircraft = NUM_INTRUDERS):
        self.other_aircraft_names = []
        for i in range(num_other_aircraft): 
            other_aircraft_name = 'AC' + str(i+1)
            self.other_aircraft_names.append(other_aircraft_name)
            ac_idx_actor = bs.traf.id2idx(acid_actor)

            check_if_inside_obs = True
            loop_counter = 0
            # check if aircraft is is created inside obstacle
            while check_if_inside_obs:
                loop_counter+= 1

                other_aircraft_dis_from_reference = np.random.randint(OTHER_AC_DISTANCE_MIN, OTHER_AC_DISTANCE_MAX)
                other_aircraft_hdg_from_reference = np.random.randint(0, 360)
                
                other_aircraft_lat, other_aircraft_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx_actor], bs.traf.lon[ac_idx_actor], other_aircraft_dis_from_reference, other_aircraft_hdg_from_reference)
                
                bs.traf.cre(acid=other_aircraft_name,actype="A320",aclat=other_aircraft_lat, aclon=other_aircraft_lon, acspd=AC_SPD)
                
                ac_idx = bs.traf.id2idx(other_aircraft_name)

                inside_temp = []
                for j in range(NUM_OBSTACLES):
                    inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], bs.traf.lat, bs.traf.lon, bs.traf.alt)[-1])

                check_if_inside_obs = any(x == True for x in inside_temp)
                if check_if_inside_obs:
                    bs.traf.delete(ac_idx)

                if loop_counter > 50:
                    raise Exception("No aircraft can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

    def _generate_polygon(self, centre):
        poly_area = np.random.randint(OBSTACLE_AREA_RANGE[0]*2, OBSTACLE_AREA_RANGE[1])
        R = np.sqrt(poly_area/ np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        
        while p_area < OBSTACLE_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        p = [fn.nm_to_latlong(centre, point) for point in p] # Convert to lat/long coordinateS
        
        return p_area, p, R
    
    def _generate_obstacles(self, acid = 'KL001'):
        # delete existing obstacles from previous episode in BlueSky
        for name in self.obstacle_names:
            bs.tools.areafilter.deleteArea(name)

        obstacle_names = []
        obstacle_vertices = []
        obstacle_radius = []

        self.obstacle_names = []
        self.obstacle_vertices = []
        self.obstacle_radius = []

        self._generate_coordinates_centre_obstacles(num_obstacles = NUM_OBSTACLES)

        obstacle_dict = {}  # Initialize the dictionary to store obstacles for overlap checking
        ac_idx = bs.traf.id2idx(acid)

        for i in range(NUM_OBSTACLES):

            centre_obst = (self.obstacle_centre_lat[i], self.obstacle_centre_lon[i])
            _, p, R = self._generate_polygon(centre_obst)
            
            points = [coord for point in p for coord in point] # Flatten the list of points
            poly_name = 'restricted_area_' + str(i+1)
            bs.tools.areafilter.defineArea(poly_name, 'POLY', points)

            obstacle_vertices_coordinates = []
            for k in range(0,len(points),2):
                obstacle_vertices_coordinates.append([points[k], points[k+1]])
            
            overlap = bs.tools.areafilter.checkInside(poly_name, np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]]))[0]
            if overlap:
                self.sample_obstacle = True
                return

            obstacle_names.append(poly_name)
            obstacle_vertices.append(obstacle_vertices_coordinates)
            obstacle_radius.append(R)

        for i in range(NUM_OBSTACLES):
            overlap_list = []  # List to store overlap information
            # Check for overlaps with existing obstacles
            for j in range(NUM_OBSTACLES):
                if i == j:
                    continue  # Skip checking the same obstacle

                found_overlap = False
                for k in range(0, len(obstacle_vertices[j])):
                    # check if the vertices of the obstacle are inside the other obstacles
                    overlap = bs.tools.areafilter.checkInside(obstacle_names[i], np.array(obstacle_vertices[j][k][0]), np.array(obstacle_vertices[j][k][1]), np.array([bs.traf.alt[ac_idx]]))[0]
                    if overlap:
                        overlap_list.append(obstacle_names[j])
                        break #break vertex loop
                    # check if points along the edges of the obstacle are inside the other obstacles
                    if k == len(obstacle_vertices[j]) -1:
                        interpolated_points = self._interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][0])
                    else:
                        interpolated_points = self._interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][k+1])
                    for point in interpolated_points:
                        overlap = bs.tools.areafilter.checkInside(obstacle_names[i], np.array(point[0]), np.array(point[1]), np.array([bs.traf.alt[ac_idx]]))[0]
                        if overlap:
                            overlap_list.append(obstacle_names[j])
                            found_overlap = True
                            break # break interpolation loop
                    if found_overlap:
                        break
        
            obstacle_dict[obstacle_names[i]] = overlap_list
        
        max_overlaps_allowed = 1

        too_many_overlapping_obstacles = any(len(overlaps) > max_overlaps_allowed for overlaps in obstacle_dict.values())

        if too_many_overlapping_obstacles:
            self.sample_obstacle = True
        else:
            self.sample_obstacle = False

        # Store the generated obstacles in the environment
        self.obstacle_names = obstacle_names
        self.obstacle_vertices = obstacle_vertices
        self.obstacle_radius = obstacle_radius

    def _interpolate_along_obstacle_vertices(self, vertex_1, vertex_2, n=15):
        """Interpolate n points between vertex_1 and vertex_2."""
        lats = np.linspace(vertex_1[0], vertex_2[0], n)
        lons = np.linspace(vertex_1[1], vertex_2[1], n)
        return list(zip(lats, lons))

    def _generate_waypoint(self, acid = 'KL001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):

            ac_idx = bs.traf.id2idx(acid)
            
            check_inside_var = True
            loop_counter = 0
            while check_inside_var:
                loop_counter += 1
                if i == 0:
                    wpt_dis_init = np.random.randint(100, 170)
                else:
                    wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
                wpt_hdg_init = np.random.randint(0, 360)
                wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)

                # working around the bug in bluesky that gives a ValueError when checkInside is used to check a single element
                wpt_lat_array = np.array([wpt_lat, wpt_lat])
                wpt_lon_array = np.array([wpt_lon, wpt_lon])
                ac_idx_alt_array = np.array([bs.traf.alt[ac_idx], bs.traf.alt[ac_idx]])
                inside_temp = []

                # check if waypoint is inside the sector. If not, a new waypoint should be generated
                if i == 0:
                    if bs.tools.areafilter.checkInside('sector', wpt_lat_array, wpt_lon_array, ac_idx_alt_array)[0]:
                        inside_temp.append(False)
                    else:
                        inside_temp.append(True)

                for j in range(NUM_OBSTACLES):
                    # shapetemp = bs.tools.areafilter.basic_shapes[self.obstacle_names[j]]
                    inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], wpt_lat_array, wpt_lon_array, ac_idx_alt_array)[0])

                # if any inside_temp is True, then the waypoint is inside an obstacle (or outside the sector for the target waypoint) and a new waypoint should be generated
                check_inside_var = any(x == True for x in inside_temp)                    
                
                if loop_counter > 1000:
                    raise Exception("No waypoints can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")


            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)
            
            if i > 0:
                ac_idx_other_aircraft = bs.traf.id2idx(self.other_aircraft_names[i-1])
                initial_wpt_qdr, _ = kwikqdrdist(bs.traf.lat[ac_idx_other_aircraft], bs.traf.lon[ac_idx_other_aircraft], self.wpt_lat[i], self.wpt_lon[i])
                bs.traf.hdg[ac_idx_other_aircraft] = initial_wpt_qdr
                bs.traf.ap.trk[ac_idx_other_aircraft] = initial_wpt_qdr


    def _generate_coordinates_centre_obstacles(self, acid = 'KL001', num_obstacles = NUM_OBSTACLES):
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        
        for i in range(num_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(0, 360)
            ac_idx = bs.traf.id2idx(acid)

            obstacle_centre_lat, obstacle_centre_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_centre_lat.append(obstacle_centre_lat)
            self.obstacle_centre_lon.append(obstacle_centre_lon)

    def _path_planning(self, num_other_aircraft = NUM_INTRUDERS):
        '''used for debugging'''
        # import pickle

        # # Saving the objects:
        # with open('de-bugging_obstacles/objs.pkl', 'wb') as f:
        #     obj0 = self.other_aircraft_names
        #     obj1 = bs.traf.lat
        #     obj2 = bs.traf.lon
        #     obj3 = bs.traf.alt
        #     obj4 = bs.traf.tas
        #     obj5 = self.wpt_lat
        #     obj6 = self.wpt_lon
        #     obj7 = self.obstacle_vertices
        #     pickle.dump([obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7], f)

        # # Getting back the objects:
        # with open('de-bugging_obstacles/objs_impossible_route_0.pkl', 'rb') as f:
            # obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7 = pickle.load(f)
            # obj7 = self._merge_overlapping_obstacles(obj7)
        '''END used for debugging'''

        merged_obstacles_vertices = self._merge_overlapping_obstacles(self.obstacle_vertices)
        self.planned_path_other_aircraft = []

        for i in range(num_other_aircraft): 
            ac_idx = bs.traf.id2idx(self.other_aircraft_names[i])
            planned_path_other_aircraft = path_plan.det_path_planning(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.alt[ac_idx], bs.traf.tas[ac_idx]/kts, self.wpt_lat[i+1], self.wpt_lon[i+1], merged_obstacles_vertices)

            '''used for debugging'''
            # ac_idx = bs.traf.id2idx(obj0[i])
            # planned_path_other_aircraft = path_plan.det_path_planning(obj1[ac_idx], obj2[ac_idx], obj3[ac_idx], obj4[ac_idx]/kts, obj5[i+1], obj6[i+1], obj7)
            '''END used for debugging'''

            self.planned_path_other_aircraft.append(planned_path_other_aircraft)

            for element in planned_path_other_aircraft:
                bs.stack.stack(f"ADDWPT {self.other_aircraft_names[i]} {element[0]} {element[1]}")

    def _merge_overlapping_obstacles(self, inputObs):
        polygons = [Polygon(obs) for obs in inputObs]
        merged = []

        while polygons:
            base = polygons.pop(0)
            group = [base]

            i = 0
            while i < len(polygons):
                if base.intersects(polygons[i]) or base.contains(polygons[i]) or polygons[i].contains(base):
                    base = base.union(polygons[i])
                    group.append(polygons[i])
                    polygons.pop(i)
                    i = 0  # restart
                else:
                    i += 1

            merged.append(base)

        return [list(poly.exterior.coords[:-1]) for poly in merged]
    
    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.intruder_distance = []
        self.intruder_cos_bearing = []
        self.intruder_sin_bearing = []
        self.intruder_x_difference_speed = []
        self.intruder_y_difference_speed = []

        self.destination_waypoint_distance = []
        self.wpt_qdr = []
        self.destination_waypoint_cos_drift = []
        self.destination_waypoint_sin_drift = []
        self.destination_waypoint_drift = []

        # obstacles 
        self.obstacle_centre_distance = []
        self.obstacle_centre_cos_bearing = []
        self.obstacle_centre_sin_bearing = []
        
        # sector polygon edges observation
        self.sector_points_distance = []
        self.sector_points_cos_drift = []
        self.sector_points_sin_drift = []        
        
        # other aircraft destination waypoints
        self.other_ac_destination_waypoint_distance = []

        # destination waypoint observation
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_tas = bs.traf.tas[ac_idx]
        wpt_qdr, wpt_dis = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])

        # intruders observation
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_qdr, int_dis = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
            self.intruder_distance.append(int_dis * NM2KM)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.intruder_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.intruder_sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            # relative velocity in x and y direction in reference to the ownship heading (x is in the direction of the ownship heading and y is perpendicular to it)
            x_dif = - np.sin(-np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.cos(-np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.intruder_x_difference_speed.append(x_dif)
            self.intruder_y_difference_speed.append(y_dif)

    
        self.destination_waypoint_distance.append(wpt_dis * NM2KM)
        self.wpt_qdr.append(wpt_qdr)

        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.destination_waypoint_drift.append(drift)
        self.destination_waypoint_cos_drift.append(np.cos(np.deg2rad(drift)))
        self.destination_waypoint_sin_drift.append(np.sin(np.deg2rad(drift)))

        # other aircraft destination waypoints
        for i in range(NUM_INTRUDERS):
            other_ac_idx = i+1
            wpt_qdr, wpt_dis = kwikqdrdist(bs.traf.lat[other_ac_idx], bs.traf.lon[other_ac_idx], self.wpt_lat[other_ac_idx], self.wpt_lon[other_ac_idx])
            self.other_ac_destination_waypoint_distance.append(wpt_dis * NM2KM)

        # obstacles observations
        for obs_idx in range(NUM_OBSTACLES):
            obs_centre_qdr, obs_centre_dis = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * NM2KM #KM        

            bearing = self.ac_hdg - obs_centre_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.obstacle_centre_distance.append(obs_centre_dis)
            self.obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))
        
        # Get vertices and points along the edges of the sector
        sector_points = self._get_observation_polygon_edges(self.poly_points_lat_lon, TOTAL_OBSERVATION_POINTS)

        sector_points_qdr, sector_points_dis = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], sector_points[:,0], sector_points[:,1])
        
        # Convert distances to kilometers
        for sector_point in sector_points_dis:
            self.sector_points_distance.append(sector_point * NM2KM)
            
        # Calculate drift for sector points
        drift = self.ac_hdg - sector_points_qdr
        drift_temp = []
        for drift_angle in drift:
            drift_angle = fn.bound_angle_positive_negative_180(drift_angle)
            drift_temp.append(drift_angle)
        drift = np.array(drift_temp)

        # Calculate cosine and sine of the drift angles
        for drift_from_point in drift:
            self.sector_points_cos_drift.append(np.cos(np.deg2rad(drift_from_point)))
            self.sector_points_sin_drift.append(np.sin(np.deg2rad(drift_from_point)))
        
        observation = {
                "intruder_distance": np.array(self.intruder_distance)/WAYPOINT_DISTANCE_MAX,
                "intruder_cos_difference_pos": np.array(self.intruder_cos_bearing),
                "intruder_sin_difference_pos": np.array(self.intruder_sin_bearing),
                "intruder_x_difference_speed": np.array(self.intruder_x_difference_speed)/AC_SPD,
                "intruder_y_difference_speed": np.array(self.intruder_y_difference_speed)/AC_SPD,
                "destination_waypoint_distance": np.array(self.destination_waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(self.destination_waypoint_cos_drift),
                "destination_waypoint_sin_drift": np.array(self.destination_waypoint_sin_drift),
                # observations on obstacles
                "restricted_area_radius": np.array(self.obstacle_radius)/(OBSTACLE_AREA_RANGE[0]),
                "restricted_area_distance": np.array(self.obstacle_centre_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_restricted_area_pos": np.array(self.obstacle_centre_cos_bearing),
                "sin_difference_restricted_area_pos": np.array(self.obstacle_centre_sin_bearing),
                "sector_points_distance": np.array(self.sector_points_distance)/WAYPOINT_DISTANCE_MAX,
                "sector_points_cos_drift": np.array(self.sector_points_cos_drift),
                "sector_points_sin_drift": np.array(self.sector_points_sin_drift),
            }

        return observation
    
    def _get_info(self):
        # Any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        return {
            'total_reward': self.total_reward,
            'waypoint_reached': self.waypoint_reached,
            'crashed': self.crashed,
            'exited_sector': self.exited_sector,
            'average_drift': self.average_drift.mean()
        }

    def _get_reward(self):
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_other_ac_reward = self._check_intrusion_other_ac()
        intrusion_reward, intrusion_terminate = self._check_intrusion()
        exit_reward, exit_terminate = self._check_sector_exit()

        total_reward = reach_reward + drift_reward + intrusion_other_ac_reward + intrusion_reward + exit_reward

        done = 0
        if self.wpt_reach[0] == 1:
            done = 1
        elif intrusion_terminate or exit_terminate:
            done = 1

        return total_reward, done, False
    
    def _check_waypoint(self):
        reward = 0
        if self.destination_waypoint_distance[0] < DISTANCE_MARGIN and self.wpt_reach[0] != 1:
            self.wpt_reach[0] = 1
            self.waypoint_reached = 1
            reward += REACH_REWARD
        else:
            reward += 0
        
        for other_ac_wpt_reach_idx in range(len(self.wpt_reach)-1):
            if self.other_ac_destination_waypoint_distance[other_ac_wpt_reach_idx] < DISTANCE_MARGIN and self.wpt_reach[other_ac_wpt_reach_idx+1] != 1:
                self.wpt_reach[other_ac_wpt_reach_idx+1] = 1

        return reward

    def _check_drift(self):
        drift = abs(np.deg2rad(self.destination_waypoint_drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY
    
    def _check_intrusion_other_ac(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            _, int_dis = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                reward += AC_INTRUSION_PENALTY
        return reward

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        terminate = 0
        for obs_idx in range(NUM_OBSTACLES):
            if bs.tools.areafilter.checkInside(self.obstacle_names[obs_idx], np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]])):
                reward += RESTRICTED_AREA_INTRUSION_PENALTY
                self.crashed = 1
                terminate = 1
        return reward, terminate
    
    def _check_sector_exit(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        terminate = 0
        if bs.tools.areafilter.checkInside('sector', np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]])) == False:
            reward += SECTOR_EXIT_PENALTY
            self.exited_sector = 1
            terminate = 1
        return reward, terminate

    def _get_action(self,action):
        dh = action[0] * D_HEADING
        dv = action[1] * D_SPEED
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx('KL001')] + dh)
        speed_new = (bs.traf.cas[bs.traf.id2idx('KL001')] + dv) * MpS2Kt

        bs.stack.stack(f"HDG {'KL001'} {heading_new}")
        bs.stack.stack(f"SPD {'KL001'} {speed_new}")

    def _render_frame(self):
        # options for rendering
        hide_other_target_waypoints = True

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        screen_coords = self.screen_coords

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        px_per_km = self.window_width/MAX_DISTANCE

        # Draw the sector polygon
        airspace_color = (0, 255, 0)    
        points = []
        for coord in self.poly_points_lat_lon:
            lat_ref = coord[0]
            lon_ref = coord[1]
            qdr, dis = kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
            dis = dis*NM2KM
            x_ref = ((np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_width
            y_ref = ((-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_height

            points.append((x_ref, y_ref))
        pygame.draw.polygon(canvas, airspace_color, points, width=2)

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(self.ac_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(self.ac_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width

        qdr, dis = kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        dis = dis*NM2KM
        x_actor = ((np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_width
        y_actor = ((-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_width

        pygame.draw.line(canvas,
            (235, 52, 52),
            (x_actor, y_actor),
            (x_actor+heading_end_x, y_actor-heading_end_y),
            width = 5
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.sin(np.deg2rad(self.ac_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(self.ac_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (x_actor,y_actor),
            (x_actor+heading_end_x, y_actor-heading_end_y),
            width = 1
        )

        # draw obstacles
        for vertices in self.obstacle_vertices:
            points = []
            for coord in vertices:
                lat_ref = coord[0]
                lon_ref = coord[1]
                qdr, dis = kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis*NM2KM
                x_ref = (np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                y_ref = (-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                points.append((x_ref, y_ref))
            pygame.draw.polygon(canvas,
                (0,0,0), points
            )

        # draw intruders
        ac_length = 3

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width

            int_qdr, int_dis = kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine if intruder is within the minimum separation distance
            _, int_dis_ownship = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            
            if int_dis_ownship < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/MAX_DISTANCE)*self.window_width
            y_pos = -(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/MAX_DISTANCE)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width

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
                radius = (INTRUSION_DISTANCE*NM2KM/MAX_DISTANCE)*self.window_width,
                width = 2
            )

        # draw target waypoint
        indx = 0
        for lat, lon, reach in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            
            indx += 1
            qdr, dis = kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width
            circle_y = (-(np.cos(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width

            if reach:
                if hide_other_target_waypoints:
                    color = (135,206,235)
                else:
                    color = (155,155,155)
                color_actor_target = (5, 128, 9)
            else:
                if hide_other_target_waypoints:
                    color = (135,206,235)
                else:
                    color = (255,255,255)
                color_actor_target = (235, 52, 52)
            
            if indx == 1:
                color = color_actor_target

            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = (DISTANCE_MARGIN/MAX_DISTANCE)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        
        pygame.display.update()
        
        self.clock.tick(self.metadata["render_fps"])
        # pygame.time.wait(10**5)

        # Event-loop for MacOS to recognise window as active
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        if self.counter == 1:
            pygame.time.wait(100)

    def close(self):
        bs.stack.stack("quit")