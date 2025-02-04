import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
from bluesky.tools.aero import kts

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 5 # km

REACH_REWARD = 1 # reach set waypoint
DRIFT_PENALTY = -0.01
RESTRICTED_AREA_INTRUSION_PENALTY = -5

INTRUSION_DISTANCE = 5 # NM

WAYPOINT_DISTANCE_MIN = 100 # KM
WAYPOINT_DISTANCE_MAX = 170 # KM

OBSTACLE_DISTANCE_MIN = 20 # KM
OBSTACLE_DISTANCE_MAX = 150 # KM

D_HEADING = 45 #degrees
D_SPEED = 20/3 # kts (check)

AC_SPD = 150 # kts
ALTITUDE = 350 # In FL

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10

NUM_OBSTACLES = 10
NUM_WAYPOINTS = 1

OBSTACLE_AREA_RANGE = (50, 1000) # In NM^2
CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates

MAX_DISTANCE = 350 # width of screen in km

class StaticObstacleEnv(gym.Env):
    """ 
    Static Obstacle Conflict Resolution Environment

    TODO:
    - Investigate CNN and Lidar based observation
    - Change rendering such that none-square screens are also possible
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512 # pixels
        self.window_height = 512 # pixels
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.observation_space = spaces.Dict(
            {   
                "destination_waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_cos_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_sin_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "restricted_area_radius": spaces.Box(0, 1, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "restricted_area_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES, ), dtype=np.float64),
                "cos_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "sin_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64)

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
        
        # variables for logging
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])

        self.obstacle_names = []

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        bs.traf.reset()

        # reset logging variables 
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD, acalt=ALTITUDE)

        # defining screen coordinates
        # defining the reference point as the top left corner of the SQUARE screen
        # from the initial position of the aircraft which is set to be the centre of the screen
        ac_idx = bs.traf.id2idx('KL001')
        d = np.sqrt(2*(MAX_DISTANCE/2)**2) #KM
        lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 315, d/NM2KM)
        
        self.screen_coords = [lat_ref_point,lon_ref_point]#[52.9, 2.6]

        self._generate_obstacles()
        self._generate_waypoint()

        ac_idx = bs.traf.id2idx('KL001')
        self.initial_wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
        bs.traf.hdg[ac_idx] = self.initial_wpt_qdr
        bs.traf.ap.trk[ac_idx] = self.initial_wpt_qdr

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._get_action(action)

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            reward, done, terminated = self._get_reward()
            if self.render_mode == "human":
                self._render_frame()
            if terminated or done:
                observation = self._get_obs()
                self.total_reward += reward
                info = self._get_info()
                return observation, reward, done, terminated, info

        observation = self._get_obs()
        self.total_reward += reward
        info = self._get_info()

        return observation, reward, done, terminated, info

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
    
    def _generate_obstacles(self):
        # delete existing obstacles from previous episode in BlueSky
        for name in self.obstacle_names:
            bs.tools.areafilter.deleteArea(name)

        self.obstacle_names = []
        self.obstacle_vertices = []
        self.obstacle_radius = []

        self._generate_coordinates_centre_obstacles(num_obstacles = NUM_OBSTACLES)

        for i in range(NUM_OBSTACLES):
            centre_obst = (self.obstacle_centre_lat[i], self.obstacle_centre_lon[i])
            _, p, R = self._generate_polygon(centre_obst)
            
            points = [coord for point in p for coord in point] # Flatten the list of points
            poly_name = 'restricted_area_' + str(i+1)
            bs.tools.areafilter.defineArea(poly_name, 'POLY', points)
            self.obstacle_names.append(poly_name)

            obstacle_vertices_coordinates = []
            for k in range(0,len(points),2):
                obstacle_vertices_coordinates.append([points[k], points[k+1]])
            
            self.obstacle_vertices.append(obstacle_vertices_coordinates)
            self.obstacle_radius.append(R)

    def _generate_waypoint(self, acid = 'KL001'):
        # original _generate_waypoints function from horizotal_cr_env
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []

        ac_idx = bs.traf.id2idx(acid)
        check_inside_var = True
        loop_counter = 0
        while check_inside_var:
            loop_counter += 1
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = np.random.randint(0, 360)
            wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)

            inside_temp = []
            for j in range(NUM_OBSTACLES):
                inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], np.array([wpt_lat]), np.array([wpt_lon]), np.array([bs.traf.alt[ac_idx]]))[0])
            check_inside_var = any(x == True for x in inside_temp)
      
            if loop_counter > 1000:
                raise Exception("No waypoints can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

        self.wpt_lat.append(wpt_lat)
        self.wpt_lon.append(wpt_lon)
        self.wpt_reach.append(0)

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

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.destination_waypoint_distance = []
        self.wpt_qdr = []
        self.destination_waypoint_cos_drift = []
        self.destination_waypoint_sin_drift = []
        self.destination_waypoint_drift = []

        self.obstacle_centre_distance = []
        self.obstacle_centre_cos_bearing = []
        self.obstacle_centre_sin_bearing = []
            
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_tas = bs.traf.tas[ac_idx]

        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
    
        self.destination_waypoint_distance.append(wpt_dis * NM2KM)
        self.wpt_qdr.append(wpt_qdr)

        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.destination_waypoint_drift.append(drift)
        self.destination_waypoint_cos_drift.append(np.cos(np.deg2rad(drift)))
        self.destination_waypoint_sin_drift.append(np.sin(np.deg2rad(drift)))
        
        for obs_idx in range(NUM_OBSTACLES):
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * NM2KM #KM        

            bearing = self.ac_hdg - obs_centre_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.obstacle_centre_distance.append(obs_centre_dis)
            self.obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        observation = {
                "destination_waypoint_distance": np.array(self.destination_waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(self.destination_waypoint_cos_drift),
                "destination_waypoint_sin_drift": np.array(self.destination_waypoint_sin_drift),
                "restricted_area_radius": np.array(self.obstacle_radius)/(OBSTACLE_AREA_RANGE[0]),
                "restricted_area_distance": np.array(self.obstacle_centre_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_restricted_area_pos": np.array(self.obstacle_centre_cos_bearing),
                "sin_difference_restricted_area_pos": np.array(self.obstacle_centre_sin_bearing),
            }

        return observation
    
    def _get_info(self):
        return {
            'total_reward': self.total_reward,
            'waypoint_reached': self.waypoint_reached,
            'crashed': self.crashed,
            'average_drift': self.average_drift.mean()
        }

    def _get_reward(self):
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward, intrusion_terminate = self._check_intrusion()
        
        total_reward = reach_reward + drift_reward + intrusion_reward
        
        done = 0
        if self.wpt_reach[0] == 1:
            done = 1
        elif intrusion_terminate:
            done = 1

        # Always return truncated as False, as timelimit is managed outside
        return total_reward, done, False
    
    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.destination_waypoint_distance:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.waypoint_reached = 1
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        return reward

    def _check_drift(self):
        drift = abs(np.deg2rad(self.destination_waypoint_drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY

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

    def _get_action(self,action):
        dh = action[0] * D_HEADING
        dv = action[1] * D_SPEED
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx('KL001')] + dh)
        speed_new = (bs.traf.cas[bs.traf.id2idx('KL001')] + dv) * MpS2Kt

        bs.stack.stack(f"HDG {'KL001'} {heading_new}")
        bs.stack.stack(f"SPD {'KL001'} {speed_new}")

    def _render_frame(self):
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

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/MAX_DISTANCE)*self.window_width

        qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
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
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/MAX_DISTANCE)*self.window_width

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
                qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis*NM2KM
                x_ref = (np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                y_ref = (-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                points.append((x_ref, y_ref))
            pygame.draw.polygon(canvas,
                (0,0,0), points
            )

        # draw target waypoint
        indx = 0
        for lat, lon, reach in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            
            indx += 1
            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width
            circle_y = (-(np.cos(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width

            color = (255,255,255)

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

    def close(self):
        pass
