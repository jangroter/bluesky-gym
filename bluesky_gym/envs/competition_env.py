"""
Competition environment: 2D navigation with conflict resolution and static
obstacle avoidance.

The agent must fly to a goal waypoint by changing heading and speed while
avoiding intruder aircraft (which follow fixed pre-planned routes), static
obstacles (restricted areas / weather cells), and while staying inside the
sector. Each episode is a ``core.scenario.Scenario``: pass one to the
constructor (``scenario=``) for a fixed evaluation episode, or leave it out to
get seeded random generation for training.

Customizing the MDP
-------------------
This env ships a deliberately simple base MDP. Competitors design their own by
overriding the three MDP hooks (all take an aircraft id, so the same code is
reusable in a future multi-agent version):

    * ``_get_obs(ac_id)``    — return the observation dict. If you change it,
      also rebuild ``self.observation_space`` in your subclass ``__init__``.
    * ``_get_reward(ac_id)`` — return the scalar reward. Does NOT affect the
      scored metrics or termination.
    * ``_get_action(ac_id, action)`` — map the action vector to BlueSky commands.

Example (subclass)::

    class MyEnv(CompetitionEnv):
        def _get_reward(self, ac_id):
            return -0.001  # your reward shaping

Or via a standard gymnasium wrapper::

    env = gym.wrappers.TransformReward(gym.make("CompetitionEnv-v0"), lambda r: r)

The scored metrics live in ``info`` and are produced by ``_update_metrics`` /
``_get_info``. Do NOT override those — they are the fixed, objective competition
score. Constructor keyword arguments (``d_heading``, ``d_speed``,
``action_frequency``, the reward weights) are the no-code tuning path.

Episode length is measured in *simulated time*, not env steps: the env
truncates when ``flight_time`` reaches ``episode_time_limit`` seconds. Metric
sampling is likewise fixed to the 1-second simulation step. Together this means
a competitor can change ``action_frequency`` freely without changing either the
scoring resolution or the simulated-time budget of an episode — so there is no
``max_episode_steps`` TimeLimit wrapper on this env.
"""

import numpy as np
import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

from core.tools import kwikqdrdist
from core.scenario import ScenarioGenerator
from core.observations import (
    WaypointObservation, OwnAirspeedObservation, IntruderObservation,
    ObstacleObservation, SectorBoundaryObservation,
)
from core.actions import HeadingAction, SpeedAction, combine_action_spaces
from core.rendering import (
    PygameCanvas, TopDownProjection,
    draw_aircraft, draw_intruder, draw_polygon, draw_waypoint, draw_line,
    BRIGHT_GREEN, RED_ORANGE, SLATE_GRAY, LIGHT_GRAY,
)

NM2KM = 1.852


class CompetitionEnv(gym.Env):
    """RL competition environment (see module docstring)."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    # Render option: draw the intruder route polylines.
    DRAW_ROUTES = True

    def __init__(self, render_mode=None, scenario=None,
                 n_intruders=10, n_obstacles=5,
                 # MDP tuning (competitor-facing)
                 d_heading=45.0, d_speed=20 / 3, action_frequency=10,
                 reach_reward=1.0, drift_penalty=-0.01, intrusion_penalty=-1.0,
                 restricted_area_penalty=-1.0, sector_exit_penalty=-1.0,
                 # scenario / scoring parameters (fixed for the competition)
                 episode_time_limit=3000.0,    # simulated seconds
                 intrusion_distance=5.0,       # NM
                 distance_margin=5.0,          # km, waypoint capture radius
                 ac_spd=150.0, altitude=350, center=(52.0, 4.0)):
        assert action_frequency >= 1, "action_frequency must be a positive integer"

        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        self.n_intruders = n_intruders
        self.n_obstacles = n_obstacles

        self.d_heading = d_heading
        self.d_speed = d_speed
        self.action_frequency = action_frequency

        self.reach_reward = reach_reward
        self.drift_penalty = drift_penalty
        self.intrusion_penalty = intrusion_penalty
        self.restricted_area_penalty = restricted_area_penalty
        self.sector_exit_penalty = sector_exit_penalty

        self.episode_time_limit = episode_time_limit
        self.intrusion_distance = intrusion_distance
        self.distance_margin = distance_margin
        self.ac_spd = ac_spd
        self.altitude = altitude   # NOTE: passed to acalt (metres) despite the FL label,
        self.center = center       #       kept for consistency with the other envs.

        self.agent = "KL001"

        # --- observation components (simple base MDP) ---
        self.waypoint_obs = WaypointObservation(n=1, distance_norm=170.0)
        self.airspeed_obs = OwnAirspeedObservation(spd_mean=ac_spd, spd_std=20.0)
        self.intruder_obs = IntruderObservation(n=n_intruders, sort_by="distance", frame="body")
        self.obstacle_obs = ObstacleObservation(
            n=n_obstacles, distance_norm=170.0, radius_norm=float(np.sqrt(1000.0 / np.pi)))
        self.sector_obs = SectorBoundaryObservation(n=12, distance_norm=170.0)

        self.observation_space = spaces.Dict({
            **self.waypoint_obs.space(),
            **self.airspeed_obs.space(),
            **self.intruder_obs.space(),
            **self.obstacle_obs.space(),
            **self.sector_obs.space(),
        })

        # --- action (heading + speed) ---
        self.heading_action = HeadingAction(d_heading=d_heading)
        self.speed_action = SpeedAction(d_speed=d_speed)
        self.action_space = combine_action_spaces([self.heading_action, self.speed_action])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as a detached (non-networked) simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)
        # 1-second timestep, fast-forward. sim_dt is the fixed metric-sampling interval.
        bs.stack.stack('DT 1;FF')
        self.sim_dt = 1

        self._fixed_scenario = scenario
        self.scenario_generator = ScenarioGenerator(
            n_obstacles=n_obstacles, n_intruders=n_intruders,
            agent_speed=ac_spd, center=center)

        self._area_names = []
        self.metrics = {}

        self.pygame_canvas = PygameCanvas(self.window_width, self.window_height,
                                          mode=render_mode)
        self.projection = TopDownProjection(
            max_distance=350, ref_lat=center[0], ref_lon=center[1],
            window_size=self.window_size)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        bs.traf.reset()

        # remove the previous episode's areas
        for name in self._area_names:
            bs.tools.areafilter.deleteArea(name)
        self._area_names = []
        self._obstacle_area_names = []

        # fixed scenario if one was supplied to the constructor, else seeded random
        scenario = self._fixed_scenario or self.scenario_generator.generate(self.np_random)
        self.scenario = scenario
        center_arr = np.array(scenario.center)

        # register sector + obstacle areas
        bs.tools.areafilter.defineArea(
            'sector', 'POLY', [c for v in scenario.sector for c in v])
        self._area_names.append('sector')
        for i, obs in enumerate(scenario.obstacles):
            name = f'obstacle_{i}'
            bs.tools.areafilter.defineArea(name, 'POLY', [c for v in obs.vertices for c in v])
            self._area_names.append(name)
            self._obstacle_area_names.append(name)

        # per-episode static geometry for ObstacleObservation (centroid + enclosing radius)
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        self.obstacle_radius = []
        for obs in scenario.obstacles:
            verts_nm = np.array([fn.latlong_to_nm(center_arr, np.array(v)) for v in obs.vertices])
            c_nm = verts_nm.mean(axis=0)
            radius = float(np.max(np.hypot(verts_nm[:, 0] - c_nm[0], verts_nm[:, 1] - c_nm[1])))
            c_ll = fn.nm_to_latlong(center_arr, c_nm)
            self.obstacle_centre_lat.append(c_ll[0])
            self.obstacle_centre_lon.append(c_ll[1])
            self.obstacle_radius.append(radius)

        # create the agent
        spec = scenario.agents[0]
        self.agent = spec.ac_id
        self.goal_lat, self.goal_lon = spec.goal
        bs.traf.cre(self.agent, actype="A320",
                    aclat=spec.start[0], aclon=spec.start[1],
                    achdg=spec.heading, acspd=spec.speed, acalt=self.altitude)

        # create the intruders and push their routes
        self.intruder_ids = []
        for route in scenario.intruder_routes:
            spawn_lat, spawn_lon = route.waypoints[0]
            nxt_lat, nxt_lon = route.waypoints[1]
            hdg = fn.get_hdg(np.array([spawn_lat, spawn_lon]), np.array([nxt_lat, nxt_lon]))
            bs.traf.cre(route.ac_id, actype="A320",
                        aclat=spawn_lat, aclon=spawn_lon,
                        achdg=hdg, acspd=route.speed, acalt=self.altitude)
            for lat, lon in route.waypoints[1:]:
                bs.stack.stack(f"{route.ac_id} ADDWPT {lat} {lon}")
            self.intruder_ids.append(route.ac_id)
        bs.stack.stack('RESO OFF')   # intruders do not run BlueSky conflict resolution

        # fit the render projection to the sector
        sector_nm = np.array([fn.latlong_to_nm(center_arr, np.array(v)) for v in scenario.sector])
        span = np.hypot(np.ptp(sector_nm[:, 0]), np.ptp(sector_nm[:, 1]))
        self.projection = TopDownProjection(
            max_distance=span * NM2KM * 1.1,
            ref_lat=scenario.center[0], ref_lon=scenario.center[1],
            window_size=self.window_size)

        # metrics + edge-detection state
        self._init_metrics(self.agent)
        self._update_metrics(self.agent, dt=0.0)

        observation = self._get_obs(self.agent)
        info = self._get_info(self.agent)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # ------------------------------------------------------------------- step
    def step(self, action):
        self._get_action(self.agent, action)
        self._reward_baseline = dict(self.metrics[self.agent])

        for _ in range(self.action_frequency):
            bs.sim.step()
            self._update_metrics(self.agent)   # every sim step (fixed 1 s DT)
            if self.render_mode == "human":
                self._render_frame()
            if self._waypoint_reached:
                break

        observation = self._get_obs(self.agent)
        reward = self._get_reward(self.agent)
        self.metrics[self.agent]["total_reward"] += reward
        terminated = self._waypoint_reached
        truncated = self.metrics[self.agent]["flight_time"] >= self.episode_time_limit
        info = self._get_info(self.agent)

        return observation, reward, terminated, truncated, info

    # -------------------------------------------------------------- MDP hooks
    def _get_obs(self, ac_id):
        inside = not self._outside_sector
        return {
            **self.waypoint_obs.observe(ac_id, [self.goal_lat], [self.goal_lon]),
            **self.airspeed_obs.observe(ac_id),
            **self.intruder_obs.observe(ac_id),
            **self.obstacle_obs.observe(ac_id, self.obstacle_centre_lat,
                                        self.obstacle_centre_lon, self.obstacle_radius),
            **self.sector_obs.observe(ac_id, self.scenario.sector, inside),
        }

    def _get_action(self, ac_id, action):
        self.heading_action.execute(ac_id, action[0])
        self.speed_action.execute(ac_id, action[1])

    def _get_reward(self, ac_id):
        """Simple base reward, computed from the metric deltas over this env step
        (so it is action-frequency invariant) plus the current drift to the goal.
        Reads only committed sim/metric state, never state produced by _get_obs."""
        m = self.metrics[ac_id]
        base = self._reward_baseline
        ac_idx = bs.traf.id2idx(ac_id)

        qdr, _ = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.goal_lat, self.goal_lon)
        drift = abs(np.deg2rad(fn.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - qdr)))

        reached = m["waypoint_reached"] - base["waypoint_reached"]
        d_intrusion = m["intrusion_time"] - base["intrusion_time"]
        d_restricted = m["time_in_restricted_area"] - base["time_in_restricted_area"]
        d_outside = m["time_outside_sector"] - base["time_outside_sector"]

        return (self.reach_reward * reached
                + self.drift_penalty * drift
                + self.intrusion_penalty * d_intrusion
                + self.restricted_area_penalty * d_restricted
                + self.sector_exit_penalty * d_outside)

    # ---------------------------------------------------------- scoring layer
    def _init_metrics(self, ac_id):
        self.metrics[ac_id] = {
            "total_reward": 0.0,
            "flight_time": 0.0,
            "waypoint_reached": 0,
            "intrusion_events": 0,
            "intrusion_time": 0.0,
            "sector_exit_events": 0,
            "time_outside_sector": 0.0,
            "restricted_area_events": 0,
            "time_in_restricted_area": 0.0,
        }
        self._intruders_in_intrusion = set()
        self._obstacles_occupied = set()
        self._outside_sector = False
        self._waypoint_reached = False

    def _update_metrics(self, ac_id, dt=None):
        """Objective, competition-scored metrics. Called on every simulation
        step (fixed 1 s sampling, independent of action_frequency); never called
        from the reward. Uses the deterministic core.tools.kwikqdrdist copy."""
        if dt is None:
            dt = self.sim_dt
        idx = bs.traf.id2idx(ac_id)
        lat, lon, alt = bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx]
        m = self.metrics[ac_id]
        m["flight_time"] += dt

        # intrusions with other aircraft (rising-edge events + occupancy time)
        now_intr = set()
        for int_id in self.intruder_ids:
            j = bs.traf.id2idx(int_id)
            if j < 0:
                continue
            _, d_nm = kwikqdrdist(lat, lon, bs.traf.lat[j], bs.traf.lon[j])
            if d_nm < self.intrusion_distance:
                now_intr.add(int_id)
        m["intrusion_events"] += len(now_intr - self._intruders_in_intrusion)
        m["intrusion_time"] += len(now_intr) * dt
        self._intruders_in_intrusion = now_intr

        # restricted-area entries
        now_obs = set()
        for name in self._obstacle_area_names:
            if self._check_inside(name, lat, lon, alt):
                now_obs.add(name)
        m["restricted_area_events"] += len(now_obs - self._obstacles_occupied)
        if now_obs:
            m["time_in_restricted_area"] += dt
        self._obstacles_occupied = now_obs

        # sector exit
        outside = not self._check_inside("sector", lat, lon, alt)
        if outside and not self._outside_sector:
            m["sector_exit_events"] += 1
        if outside:
            m["time_outside_sector"] += dt
        self._outside_sector = outside

        # waypoint capture (the only terminal condition)
        _, d_goal_nm = kwikqdrdist(lat, lon, self.goal_lat, self.goal_lon)
        if d_goal_nm * NM2KM < self.distance_margin and not self._waypoint_reached:
            self._waypoint_reached = True
            m["waypoint_reached"] = 1

    def _get_info(self, ac_id):
        return dict(self.metrics[ac_id])

    def _check_inside(self, name, lat, lon, alt):
        # duplicate into 2-element arrays: BlueSky's checkInside raises on 0-d input
        lat_a = np.array([lat, lat])
        lon_a = np.array([lon, lon])
        alt_a = np.array([alt, alt])
        return bool(bs.tools.areafilter.checkInside(name, lat_a, lon_a, alt_a)[0])

    # ---------------------------------------------------------------- render
    def render(self):
        # rgb_array returns a frame; human rendering happens inside step()/reset()
        if self.render_mode == "rgb_array":
            canvas = self.pygame_canvas.begin_frame()
            self._draw_world(canvas)
            return self.pygame_canvas.end_frame(canvas)
        return None

    def _render_frame(self):
        canvas = self.pygame_canvas.begin_frame()
        self._draw_world(canvas)
        self.pygame_canvas.end_frame(canvas)

    def _draw_world(self, canvas):
        # sector outline
        sector_px = [self.projection.project(lat, lon) for lat, lon in self.scenario.sector]
        draw_polygon(canvas, sector_px, color=BRIGHT_GREEN, filled=False, width=2)

        # obstacles (filled, colored by kind)
        for obs in self.scenario.obstacles:
            pts = [self.projection.project(lat, lon) for lat, lon in obs.vertices]
            color = SLATE_GRAY
            draw_polygon(canvas, pts, color=color, filled=True)

        # intruder route polylines
        if self.DRAW_ROUTES:
            for route in self.scenario.intruder_routes:
                pxs = [self.projection.project(lat, lon) for lat, lon in route.waypoints]
                for a, b in zip(pxs[:-1], pxs[1:]):
                    draw_line(canvas, a[0], a[1], b[0], b[1], color=LIGHT_GRAY, width=1)

        ac_idx = bs.traf.id2idx(self.agent)

        # intruders + safety circles
        for int_id in self.intruder_ids:
            j = bs.traf.id2idx(int_id)
            if j < 0:
                continue
            ix, iy = self.projection.project(bs.traf.lat[j], bs.traf.lon[j])
            _, d_nm = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                                  bs.traf.lat[j], bs.traf.lon[j])
            draw_intruder(canvas, ix, iy, bs.traf.hdg[j], self.projection,
                          body_km=2, heading_km=5,
                          safety_radius_km=self.intrusion_distance * NM2KM,
                          in_intrusion=d_nm < self.intrusion_distance)

        # goal
        gx, gy = self.projection.project(self.goal_lat, self.goal_lon)
        draw_waypoint(canvas, gx, gy, margin_km=self.distance_margin, projection=self.projection)

        # agent (drawn last, on top)
        ax, ay = self.projection.project(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        draw_aircraft(canvas, ax, ay, bs.traf.hdg[ac_idx],
                      body_km=8, heading_km=20, projection=self.projection, color="black")

    def close(self):
        bs.stack.stack("quit")
