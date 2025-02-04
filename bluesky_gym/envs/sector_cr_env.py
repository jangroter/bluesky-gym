import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

AC_DENSITY_RANGE = (0.003, 0.007) # In AC/NM^2
AC_DENSITY_MU = 0.005 # In AC/NM^2
AC_DENSITY_SIGMA = 0.001 # In AC/NM^2

POLY_AREA_RANGE = (2400, 3750) # In NM^2
CENTER = np.array([51.990426702297746, 4.376124857109851]) # TU Delft AE Faculty coordinates
ALTITUDE = 350 # In FL

# Aircraft parameters
AC_SPD = 150
AC_TYPE = "A320"
ACTOR = "KL001"

# Conversion factors
NM2KM = 1.852
MpS2Kt = 1.94384
FL2M = 30.48

INTRUSION_DISTANCE = 5 # NM

# Model parameters
ACTION_FREQUENCY = 5
NUM_AC_STATE = 4
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1
D_HEADING = 22.5 # deg
D_VELOCITY = 20/3 # kts

class SectorCREnv(gym.Env):
    """ 
    Sector Conflict Resolution Environment
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}
    
    def __init__(self, render_mode=None, ac_density_mode="normal"):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment
        self.density_mode = ac_density_mode
        self.poly_name = 'airspace'
        # Feel free to add more observation spaces
        self.observation_space = spaces.Dict(
            {
                "cos(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "sin(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "airspeed": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "x_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "y_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vx_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vy_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "cos(track)": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "sin(track)": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "distances": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64)
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

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        self.window = None
        self.clock = None
    
    def reset(self, seed=None, options=None):
        bs.traf.reset()
        bs.tools.areafilter.deleteArea(self.poly_name)
        super().reset(seed=seed)

        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
       
        self._generate_polygon() # Create airspace polygon
        
        if self.density_mode == "normal":
            rand_density = np.random.normal(AC_DENSITY_MU, AC_DENSITY_SIGMA)
            self.num_ac = int(max(np.ceil(rand_density * self.poly_area), NUM_AC_STATE+1)) # Get total number of AC in the airspace including agent (min = 3)
        else:
            rand_density = np.random.uniform(*AC_DENSITY_RANGE)
            self.num_ac = int(max(np.ceil(rand_density * self.poly_area), NUM_AC_STATE+1)) # Get total number of AC in the airspace including agent (min = 3)
        
        self._generate_waypoints() # Create waypoints for aircraft
        self._generate_ac() # Create aircraft in the airspace

        observation = self._get_observation()

        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._get_action(action)
        action_frequency = ACTION_FREQUENCY
        for _ in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":              
                self._render_frame()
        
        observation = self._get_observation()        
        reward = self._get_reward()
        info = self._get_info()

        # truncate instead of terminate to avoid aircraft learning to exit sector fast
        truncate = self._check_inside_airspace()

        return observation, reward, False, truncate, info
    
    def _check_inside_airspace(self):
        ac_idx = bs.traf.id2idx(ACTOR)
        if bs.tools.areafilter.checkInside(self.poly_name, np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([ALTITUDE*FL2M])):
            return False
        else:
            return True

    def _generate_polygon(self):
        
        R = np.sqrt(POLY_AREA_RANGE[1] / np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        
        while p_area < POLY_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        self.poly_area = p_area
        
        self.poly_points = np.array(p) # Polygon vertices are saved in terms of NM
        
        p = [fn.nm_to_latlong(CENTER, point) for point in p] # Convert to lat/long coordinateS
        
        points = [coord for point in p for coord in point] # Flatten the list of points
        bs.tools.areafilter.defineArea(self.poly_name, 'POLY', points)
    
    def _generate_waypoints(self):
        
        edges = []
        perim_tot = 0
        
        for i in range(len(self.poly_points)):
            p1 = np.array(self.poly_points[i])
            p2 = np.array(self.poly_points[(i+1) % len(self.poly_points)]) # Ensure wrap-around
            len_edge = fn.euclidean_distance(p1, p2)
            edges.append((p1, p2, len_edge))
            perim_tot += len_edge
        
        d_list = [np.random.uniform(0, perim_tot) for _ in range(self.num_ac)] # Each ac including agent is given a waypoint
        d_list.sort()
        
        self.wpts = [] # In terms of NM
        current_d = 0
        
        for d in d_list:
            while d > current_d + edges[0][2]:
                current_d += edges[0][2]
                edges.pop(0)
            
            edge = edges[0]
            frac = (d - current_d) / edge[2]
            p = edge[0] + frac * (edge[1] - edge[0])
            self.wpts.append(p)
        
    def _generate_ac(self) -> None:
        
        # Determine bounding box of airspace
        min_x = min(self.poly_points[:, 0])
        min_y = min(self.poly_points[:, 1])
        max_x = max(self.poly_points[:, 0])
        max_y = max(self.poly_points[:, 1])
        
        init_p_latlong = []
        
        while len(init_p_latlong) < self.num_ac:
            p = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            p = fn.nm_to_latlong(CENTER, p)
            if bs.tools.areafilter.checkInside(self.poly_name, np.array([p[0]]), np.array([p[1]]), np.array([ALTITUDE*FL2M])):
                init_p_latlong.append(p)
        
        wpt_agent = fn.nm_to_latlong(CENTER, self.wpts[0])
        init_pos_agent = init_p_latlong[0]
        hdg_agent = fn.get_hdg(init_pos_agent, wpt_agent)
        
        # Actor AC is the only one that has ACTOR as acid
        bs.traf.cre(ACTOR, actype=AC_TYPE, aclat=init_pos_agent[0], aclon=init_pos_agent[1], achdg=hdg_agent, acspd=AC_SPD, acalt=ALTITUDE)
        
        for i in range(1, len(init_p_latlong)):
            wpt = fn.nm_to_latlong(CENTER, self.wpts[i])
            init_pos = init_p_latlong[i]
            hdg = fn.get_hdg(init_pos, wpt)
            bs.traf.cre(acid=str(i), actype=AC_TYPE, aclat=init_pos[0], aclon=init_pos[1], achdg=hdg, acspd=AC_SPD, acalt=ALTITUDE)
    
    def _get_info(self):
        # Here you implement any additional info that you want to log after an episode
        return {
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean()
        }
    
    def _get_reward(self):
        
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()

        total_reward = drift_reward + intrusion_reward
        self.total_reward += total_reward

        return total_reward
    
    def _get_observation(self):

        ac_idx = bs.traf.id2idx(ACTOR)

        # Observation vector shape and components
        self.cos_drift = np.array([])
        self.sin_drift = np.array([])
        self.airspeed = np.array([])
        self.x_r = np.array([])
        self.y_r = np.array([])
        self.vx_r = np.array([])
        self.vy_r = np.array([])
        self.cos_track = np.array([])
        self.sin_track = np.array([])
        self.distances = np.array([])

        # Drift of agent aircraft for reward calculation
        drift = 0

        ac_hdg = bs.traf.hdg[ac_idx]
        
        # Get and decompose agent aircaft drift
        wpts = fn.nm_to_latlong(CENTER, self.wpts[ac_idx])
        wpt_qdr, _  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpts[0], wpts[1])

        drift = ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)
        self.drift = drift
        self.cos_drift = np.append(self.cos_drift, np.cos(np.deg2rad(drift)))
        self.sin_drift = np.append(self.sin_drift, np.sin(np.deg2rad(drift)))

        # Get agent aircraft airspeed, m/s
        self.airspeed = np.append(self.airspeed, bs.traf.tas[ac_idx])

        vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
        vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]

        ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
        distances = [fn.euclidean_distance(ac_loc, fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000) for i in range(1, self.num_ac)]
        ac_idx_by_dist = np.argsort(distances)

        for i in range(self.num_ac-1):
            ac_idx = ac_idx_by_dist[i]+1
            int_hdg = bs.traf.hdg[ac_idx]
            
            # Intruder AC relative position, m
            int_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000
            self.x_r = np.append(self.x_r, int_loc[0] - ac_loc[0])
            self.y_r = np.append(self.y_r, int_loc[1] - ac_loc[1])
            
            # Intruder AC relative velocity, m/s
            vx_int = np.cos(np.deg2rad(int_hdg)) * bs.traf.tas[ac_idx]
            vy_int = np.sin(np.deg2rad(int_hdg)) * bs.traf.tas[ac_idx]
            self.vx_r = np.append(self.vx_r, vx_int - vx)
            self.vy_r = np.append(self.vy_r, vy_int - vy)

            # Intruder AC relative track, rad
            track = np.arctan2(vy_int - vy, vx_int - vx)
            self.cos_track = np.append(self.cos_track, np.cos(track))
            self.sin_track = np.append(self.sin_track, np.sin(track))

            self.distances = np.append(self.distances, distances[ac_idx-1])

        observation = {
            "cos(drift)": self.cos_drift,
            "sin(drift)": self.sin_drift,
            "airspeed": (self.airspeed-150)/6,
            "x_r": self.x_r[:NUM_AC_STATE]/13000,
            "y_r": self.y_r[:NUM_AC_STATE]/13000,
            "vx_r": self.vx_r[:NUM_AC_STATE]/32,
            "vy_r": self.vy_r[:NUM_AC_STATE]/66,
            "cos(track)": self.cos_track[:NUM_AC_STATE],
            "sin(track)": self.sin_track[:NUM_AC_STATE],
            "distances": (self.distances[:NUM_AC_STATE]-50000.)/15000.
        }

        return observation
    
    def _get_action(self, action):
        dh = action[0] * D_HEADING
        dv = action[1] * D_VELOCITY
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(ACTOR)] + dh)
        speed_new = (bs.traf.cas[bs.traf.id2idx(ACTOR)] + dv) * MpS2Kt

        bs.stack.stack(f"HDG {ACTOR} {heading_new}")
        bs.stack.stack(f"SPD {ACTOR} {speed_new}")

    def _check_drift(self):
        drift = abs(np.deg2rad(self.drift))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY
    
    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx(ACTOR)
        reward = 0
        for i in range(self.num_ac-1):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        
        return reward
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = max(np.linalg.norm(point1 - point2) for point1 in self.poly_points for point2 in self.poly_points)*NM2KM
        
        px_per_km = self.window_width/max_distance

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))
        
        # Draw airspace
        airspace_color = (255, 0, 0)
        coords = [((self.window_width/2)+point[0]*NM2KM*px_per_km, (self.window_height/2)-point[1]*NM2KM*px_per_km) for point in self.poly_points]
        pygame.draw.polygon(canvas, airspace_color, coords, width=2)

        # Draw ownship
        ac_idx = bs.traf.id2idx(ACTOR)
        ac_length = 10
        ac_hdg = bs.traf.hdg[ac_idx]
        heading_end_x = np.cos(np.deg2rad(ac_hdg)) * ac_length
        heading_end_y = np.sin(np.deg2rad(ac_hdg)) * ac_length
        ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])

        x_pos = (self.window_width/2)+(np.cos(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*px_per_km)
        y_pos = (self.window_height/2)-(np.sin(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*px_per_km)

        pygame.draw.line(canvas,
            (0,0,0),
            (x_pos,y_pos),
            ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
            width = 4
        )

        # Draw heading line
        heading_length = 20
        heading_end_x = np.cos(np.deg2rad(ac_hdg)) * heading_length
        heading_end_y = np.sin(np.deg2rad(ac_hdg)) * heading_length

        pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
        )

        # Draw intruders
        ac_length = 3

        for i in range(self.num_ac-1):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = np.cos(np.deg2rad(int_hdg)) * ac_length
            heading_end_y = np.sin(np.deg2rad(int_hdg)) * ac_length

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            separation = bs.tools.geo.kwikdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # Determine color
            if separation < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.window_width/2)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)
            y_pos = (self.window_height/2)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # Draw heading line
            heading_length = 20
            heading_end_x = np.cos(np.deg2rad(int_hdg)) * heading_length
            heading_end_y = np.sin(np.deg2rad(int_hdg)) * heading_length

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
                radius = INTRUSION_DISTANCE*NM2KM*px_per_km,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        pass