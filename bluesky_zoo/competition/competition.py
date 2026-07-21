"""
Multi-agent competition environment: 2D navigation with conflict resolution
and static obstacle avoidance, as a PettingZoo ``ParallelEnv``.

Multi-agent version of ``bluesky_gym.envs.competition_env.CompetitionEnv``:
every aircraft is a controlled agent flying to its own goal waypoint by
changing heading and speed, while avoiding the other agents, static obstacles
(restricted areas / weather cells), and while staying inside the sector.
There are no scripted intruders. Each episode is a ``core.scenario.Scenario``:
pass one to the constructor (``scenario=``) for a fixed evaluation episode
(its ``intruder_routes`` must be empty), or leave it out to get seeded random
generation for training.

Agent lifecycle (PettingZoo Parallel API)
-----------------------------------------
An agent terminates when it reaches its goal waypoint: its final observation,
reward and info are returned in that step, after which it leaves
``self.agents`` and its aircraft is deleted from the simulation (so it stops
posing conflicts to the remaining agents). The episode ends when every agent
has terminated, or truncates for all remaining agents once the simulated time
reaches ``episode_time_limit``.

Customizing the MDP
-------------------
Same contract as the single-agent env: competitors design their own MDP by
overriding the three hooks (all take an aircraft id):

    * ``_get_obs(ac_id)``    — return the observation dict. If you change it,
      also rebuild the observation space in your subclass ``__init__``.
    * ``_get_reward(ac_id)`` — return the scalar reward. Does NOT affect the
      scored metrics or termination.
    * ``_get_action(ac_id, action)`` — map the action vector to BlueSky commands.

The scored metrics live in the per-agent ``info`` dicts and are produced by
``_update_metrics`` / ``_get_info``. Do NOT override those — they are the
fixed, objective competition score. Note that intrusions are recorded
symmetrically: one conflict between A and B counts as an event (and occupancy
time) in *both* agents' metrics — per-agent scoring, not a global conflict
count.

Episode length is measured in *simulated time*, not env steps: the episode
truncates when the shared clock reaches ``episode_time_limit`` seconds, and
metric sampling is fixed to the 1-second simulation step. A competitor can
therefore change ``action_frequency`` freely without changing either the
scoring resolution or the simulated-time budget of an episode.

Limitations
-----------
BlueSky is a process-global singleton, so only one BlueSky-backed environment
instance can be active per process. Tools that interleave two instances
(e.g. ``pettingzoo.test.parallel_seed_test``) do not work with this env.
"""

import numpy as np
import bluesky as bs
import bluesky_gym.envs.common.functions as fn

from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from core.tools import kwikqdrdist
from core.scenario import ScenarioGenerator, agent_callsigns
from core.observations import (
    WaypointObservation, OwnAirspeedObservation, IntruderObservation,
    ObstacleObservation, SectorBoundaryObservation,
)
from core.actions import HeadingAction, SpeedAction, combine_action_spaces
from core.rendering import (
    PygameCanvas, TopDownProjection,
    draw_intruder, draw_polygon, draw_waypoint, draw_line,
    BRIGHT_GREEN, SLATE_GRAY, LIGHT_GRAY, BLACK,
)

NM2KM = 1.852


def parallel_env(**kwargs):
    """PettingZoo-conventional factory."""
    return CompetitionZooEnv(**kwargs)


class CompetitionZooEnv(ParallelEnv):
    """Multi-agent RL competition environment (see module docstring)."""

    metadata = {"name": "competition_v0", "render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode=None, scenario=None,
                 n_agents=10, n_obstacles=5,
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

        if scenario is not None:
            assert not scenario.intruder_routes, (
                "CompetitionZooEnv has no scripted intruders; "
                "scenario.intruder_routes must be empty")
            n_agents = len(scenario.agents)
            self.possible_agents = [spec.ac_id for spec in scenario.agents]
        else:
            self.possible_agents = agent_callsigns(n_agents)
        assert n_agents >= 2, "need at least 2 agents (use CompetitionEnv for 1)"
        self.agents = []

        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        self.n_agents = n_agents
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

        # --- observation components (simple base MDP) ---
        self.waypoint_obs = WaypointObservation(n=1, distance_norm=170.0)
        self.airspeed_obs = OwnAirspeedObservation(spd_mean=ac_spd, spd_std=20.0)
        self.intruder_obs = IntruderObservation(n=n_agents - 1, sort_by="distance", frame="body")
        self.obstacle_obs = ObstacleObservation(
            n=n_obstacles, distance_norm=170.0, radius_norm=float(np.sqrt(1000.0 / np.pi)))
        self.sector_obs = SectorBoundaryObservation(n=12, distance_norm=170.0)

        # spaces are homogeneous: one space object shared by all agents
        observation_space = spaces.Dict({
            **self.waypoint_obs.space(),
            **self.airspeed_obs.space(),
            **self.intruder_obs.space(),
            **self.obstacle_obs.space(),
            **self.sector_obs.space(),
        })

        # --- action (heading + speed) ---
        self.heading_action = HeadingAction(d_heading=d_heading)
        self.speed_action = SpeedAction(d_speed=d_speed)
        action_space = combine_action_spaces([self.heading_action, self.speed_action])

        self.observation_spaces = {a: observation_space for a in self.possible_agents}
        self.action_spaces = {a: action_space for a in self.possible_agents}

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
            n_obstacles=n_obstacles, n_intruders=0, n_agents=n_agents,
            agent_speed=ac_spd, center=center)

        self._np_random = None
        self._area_names = []
        self.metrics = {}

        self.pygame_canvas = PygameCanvas(self.window_width, self.window_height,
                                          mode=render_mode)
        self.projection = TopDownProjection(
            max_distance=350, ref_lat=center[0], ref_lon=center[1],
            window_size=self.window_size)

    # ---------------------------------------------------------------- spaces
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        # ParallelEnv has no super().reset(seed); rng persists across unseeded resets
        if seed is not None or self._np_random is None:
            self._np_random, _ = seeding.np_random(seed)
        bs.traf.reset()

        # remove the previous episode's areas
        for name in self._area_names:
            bs.tools.areafilter.deleteArea(name)
        self._area_names = []
        self._obstacle_area_names = []

        # fixed scenario if one was supplied to the constructor, else seeded random
        scenario = self._fixed_scenario or self.scenario_generator.generate(self._np_random)
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

        # create the agents
        self.agents = list(self.possible_agents)
        self._goal = {}
        for spec in scenario.agents:
            self._goal[spec.ac_id] = tuple(spec.goal)
            bs.traf.cre(spec.ac_id, actype="A320",
                        aclat=spec.start[0], aclon=spec.start[1],
                        achdg=spec.heading, acspd=spec.speed, acalt=self.altitude)
        bs.stack.stack('RESO OFF')   # agents do not run BlueSky conflict resolution

        # fit the render projection to the sector
        sector_nm = np.array([fn.latlong_to_nm(center_arr, np.array(v)) for v in scenario.sector])
        span = np.hypot(np.ptp(sector_nm[:, 0]), np.ptp(sector_nm[:, 1]))
        self.projection = TopDownProjection(
            max_distance=span * NM2KM * 1.1,
            ref_lat=scenario.center[0], ref_lon=scenario.center[1],
            window_size=self.window_size)

        # shared episode clock + per-agent metrics and edge-detection state
        self.sim_time = 0.0
        self.metrics = {}
        self._reward_baseline = {}
        self._in_intrusion_with = {}
        self._obstacles_occupied = {}
        self._outside_sector = {}
        self._waypoint_reached = {}
        for agent in self.agents:
            self._init_metrics(agent)
            self._update_metrics(agent, dt=0.0)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    # ------------------------------------------------------------------- step
    def step(self, actions):
        step_agents = list(self.agents)   # output dicts cover exactly these agents
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent in step_agents:
            self._get_action(agent, actions[agent])
        for agent in step_agents:
            self._reward_baseline[agent] = dict(self.metrics[agent])

        # fixed-length loop: remaining agents get their full sim seconds even
        # when others reach their goal mid-step
        for _ in range(self.action_frequency):
            if not self.agents:
                break
            bs.sim.step()
            self.sim_time += self.sim_dt
            for agent in self.agents:
                self._update_metrics(agent)   # every sim step (fixed 1 s DT)

            reached = [a for a in self.agents if self._waypoint_reached[a]]
            # snapshot final obs/reward/info while the aircraft still exist
            for agent in reached:
                observations[agent] = self._get_obs(agent)
                reward = self._get_reward(agent)
                self.metrics[agent]["total_reward"] += reward
                rewards[agent] = reward
                terminations[agent] = True
                truncations[agent] = False
                infos[agent] = self._get_info(agent)
            # then delete: every delete shifts the traf arrays, so a fresh
            # id2idx per aircraft is required
            for agent in reached:
                idx = bs.traf.id2idx(agent)
                if idx >= 0:
                    bs.traf.delete(idx)
                self.agents.remove(agent)

            if self.render_mode == "human":
                self._render_frame()

        truncate_all = self.sim_time >= self.episode_time_limit
        for agent in self.agents:
            observations[agent] = self._get_obs(agent)
            reward = self._get_reward(agent)
            self.metrics[agent]["total_reward"] += reward
            rewards[agent] = reward
            terminations[agent] = False
            truncations[agent] = truncate_all
            infos[agent] = self._get_info(agent)
        if truncate_all:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # -------------------------------------------------------------- MDP hooks
    def _get_obs(self, ac_id):
        goal_lat, goal_lon = self._goal[ac_id]
        inside = not self._outside_sector[ac_id]
        return {
            **self.waypoint_obs.observe(ac_id, [goal_lat], [goal_lon]),
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
        base = self._reward_baseline[ac_id]
        ac_idx = bs.traf.id2idx(ac_id)
        goal_lat, goal_lon = self._goal[ac_id]

        qdr, _ = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], goal_lat, goal_lon)
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
        self._in_intrusion_with[ac_id] = set()
        self._obstacles_occupied[ac_id] = set()
        self._outside_sector[ac_id] = False
        self._waypoint_reached[ac_id] = False

    def _update_metrics(self, ac_id, dt=None):
        """Objective, competition-scored metrics. Called on every simulation
        step (fixed 1 s sampling, independent of action_frequency); never called
        from the reward. Uses the deterministic core.tools.kwikqdrdist copy.
        Intrusions are recorded in each participant's own metrics (symmetric)."""
        if dt is None:
            dt = self.sim_dt
        idx = bs.traf.id2idx(ac_id)
        lat, lon, alt = bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx]
        m = self.metrics[ac_id]
        m["flight_time"] += dt

        # intrusions with the other live agents (rising-edge events + occupancy time)
        now_intr = set()
        for other in self.agents:
            if other == ac_id:
                continue
            j = bs.traf.id2idx(other)
            if j < 0:
                continue
            _, d_nm = kwikqdrdist(lat, lon, bs.traf.lat[j], bs.traf.lon[j])
            if d_nm < self.intrusion_distance:
                now_intr.add(other)
        m["intrusion_events"] += len(now_intr - self._in_intrusion_with[ac_id])
        m["intrusion_time"] += len(now_intr) * dt
        self._in_intrusion_with[ac_id] = now_intr

        # restricted-area entries
        now_obs = set()
        for name in self._obstacle_area_names:
            if self._check_inside(name, lat, lon, alt):
                now_obs.add(name)
        m["restricted_area_events"] += len(now_obs - self._obstacles_occupied[ac_id])
        if now_obs:
            m["time_in_restricted_area"] += dt
        self._obstacles_occupied[ac_id] = now_obs

        # sector exit
        outside = not self._check_inside("sector", lat, lon, alt)
        if outside and not self._outside_sector[ac_id]:
            m["sector_exit_events"] += 1
        if outside:
            m["time_outside_sector"] += dt
        self._outside_sector[ac_id] = outside

        # waypoint capture (the only terminal condition)
        goal_lat, goal_lon = self._goal[ac_id]
        _, d_goal_nm = kwikqdrdist(lat, lon, goal_lat, goal_lon)
        if d_goal_nm * NM2KM < self.distance_margin and not self._waypoint_reached[ac_id]:
            self._waypoint_reached[ac_id] = True
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

        # obstacles (filled)
        for obs in self.scenario.obstacles:
            pts = [self.projection.project(lat, lon) for lat, lon in obs.vertices]
            draw_polygon(canvas, pts, color=SLATE_GRAY, filled=True)

        # goal lines + waypoints first, aircraft on top
        for agent in self.agents:
            idx = bs.traf.id2idx(agent)
            if idx < 0:
                continue
            ax, ay = self.projection.project(bs.traf.lat[idx], bs.traf.lon[idx])
            gx, gy = self.projection.project(*self._goal[agent])
            draw_line(canvas, ax, ay, gx, gy, color=LIGHT_GRAY, width=1)
            draw_waypoint(canvas, gx, gy, margin_km=self.distance_margin,
                          projection=self.projection)
        for agent in self.agents:
            idx = bs.traf.id2idx(agent)
            if idx < 0:
                continue
            ax, ay = self.projection.project(bs.traf.lat[idx], bs.traf.lon[idx])
            draw_intruder(canvas, ax, ay, bs.traf.hdg[idx], self.projection,
                          body_km=8, heading_km=20,
                          safety_radius_km=self.intrusion_distance * NM2KM,
                          in_intrusion=bool(self._in_intrusion_with[agent]),
                          color_safe=BLACK)

    def close(self):
        bs.stack.stack("quit")
