"""
Scenario definition and random generation for the competition environment.

A ``Scenario`` fully specifies one episode: the sector polygon, the static
obstacles, the agent (start/goal), and the pre-planned intruder routes. The
environment consumes a ``Scenario`` in ``reset``; it can either be supplied
explicitly (fixed evaluation scenarios) or produced by ``ScenarioGenerator``
for seeded random training.

This module depends only on ``numpy`` and ``shapely`` — no BlueSky, no
``bluesky_gym`` — so it is unit-testable without a simulator and reusable by a
future PettingZoo (MARL) package.

Units
-----
Everything *stored* in a ``Scenario`` is BlueSky-native: latitude/longitude in
degrees, speed in m/s, heading in degrees true. All *generation* geometry runs
in a local flat-earth x/y frame in nautical miles around ``center`` (x = north,
y = east), converted to lat/lon only at the boundary. The NM<->lat/lon math is
identical to ``bluesky_gym.envs.common.functions.nm_to_latlong``; it is copied
here (a few lines) rather than imported to keep ``core`` free of ``bluesky_gym``.
"""

from dataclasses import dataclass, field
import math

import numpy as np
from shapely.geometry import Polygon, LineString, Point

NM2KM = 1.852


# ---------------------------------------------------------------------------
# Scenario dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Obstacle:
    """A static obstacle (restricted area or weather cell).

    vertices : list of (lat, lon) in degrees, unclosed ring.
    kind : "restricted" or "weather" — a tag that only drives render color;
        both kinds are treated identically by the environment's scoring.
    """
    vertices: list
    kind: str = "restricted"


@dataclass
class Route:
    """A pre-planned route for one intruder aircraft.

    ac_id : BlueSky aircraft id.
    waypoints : list of (lat, lon) in degrees. Element [0] is the spawn point;
        the remaining elements are pushed as ADDWPT legs, the last being the
        sector exit.
    speed : cruise speed in m/s.
    """
    ac_id: str
    waypoints: list
    speed: float


@dataclass
class AgentSpec:
    """A controlled agent's start and goal.

    heading is the initial true heading in degrees (the generator points it at
    the goal); speed is in m/s.
    """
    ac_id: str
    start: tuple
    goal: tuple
    heading: float
    speed: float


@dataclass
class Scenario:
    """A full episode specification."""
    center: tuple                       # (lat, lon) projection/render reference
    sector: list                        # list of (lat, lon), clockwise, unclosed
    obstacles: list = field(default_factory=list)
    agents: list = field(default_factory=list)          # one AgentSpec per controlled aircraft
    intruder_routes: list = field(default_factory=list)


def agent_callsigns(n):
    """Aircraft ids for ``n`` controlled agents: KL001, KL002, ...

    Follows the existing convention (KL001..KL009, then KL0010, ...); the
    inconsistent width past 9 agents is a known quirk kept for now.
    """
    return [f"KL00{i + 1}" for i in range(n)]


# ---------------------------------------------------------------------------
# Geometry helpers (flat NM frame around a lat/lon center)
# ---------------------------------------------------------------------------
def _nm_to_latlon(center, point):
    """Convert an (x=north, y=east) NM offset to (lat, lon) degrees."""
    lat = center[0] + point[0] / 60.0
    lon = center[1] + point[1] / (60.0 * math.cos(math.radians(center[0])))
    return (lat, lon)


def _random_point_on_circle(rng, radius):
    alpha = 2.0 * math.pi * rng.random()
    return np.array([radius * math.cos(alpha), radius * math.sin(alpha)])


def _sort_points_clockwise(points):
    return [points[i] for i in np.argsort([math.atan2(p[1], p[0]) for p in points])]


def _bearing_nm(p_from, p_to):
    """True heading (deg) from one NM-frame point to another (x=north, y=east)."""
    dx = p_to[0] - p_from[0]   # north
    dy = p_to[1] - p_from[1]   # east
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _separated(p, others, min_d):
    """True if NM-frame point ``p`` is at least ``min_d`` NM from every point in ``others``."""
    return all(math.hypot(p[0] - o[0], p[1] - o[1]) >= min_d for o in others)


def resample_perimeter(vertices, n):
    """Resample a polygon boundary into ``n`` points evenly spaced by arc length.

    Parameters
    ----------
    vertices : sequence of (lat, lon)
        Polygon vertices, unclosed.
    n : int
        Number of output points.

    Returns
    -------
    np.ndarray of shape (n, 2) with columns [lat, lon].
    """
    ring = LineString(list(vertices) + [vertices[0]])
    length = ring.length
    pts = [ring.interpolate((i / n) * length) for i in range(n)]
    return np.array([[p.x, p.y] for p in pts])


# ---------------------------------------------------------------------------
# Random generator
# ---------------------------------------------------------------------------
class ScenarioGenerator:
    """Seeded random ``Scenario`` factory.

    ``generate(rng)`` takes a ``numpy.random.Generator`` (e.g. the env's
    ``self.np_random``), so a fixed seed yields a reproducible scenario.

    Parameters
    ----------
    n_obstacles, n_intruders : int
        Target counts. Fewer obstacles may result if placement repeatedly
        fails (the env zero-pads observation slots).
    n_agents : int
        Number of controlled agents (one AgentSpec each in ``Scenario.agents``).
    sector_area_range, obstacle_area_range : (min, max) in NM^2.
    goal_distance_range : (min, max) straight-line agent start->goal distance in km.
    agent_speed : m/s.
    intruder_speed_range : (min, max) m/s.
    center : (lat, lon) reference for the flat NM frame.
    agent_separation_nm : float
        Minimum spacing between different agents' start points. Keep above
        the intrusion distance so no episode begins with agents in conflict
        at their spawn points. Goals are unconstrained relative to each other
        (they may coincide); each start is far from its own goal via
        ``goal_distance_range``.
    """

    def __init__(self, n_obstacles=5, n_intruders=5, n_agents=1,
                 sector_area_range=(15_000, 23_000),
                 obstacle_area_range=(50, 1000),
                 goal_distance_range=(100, 170),
                 agent_speed=150.0,
                 intruder_speed_range=(120, 180),
                 center=(52.0, 4.0),
                 agent_start_margin_nm=10.0,
                 obstacle_clearance_nm=2.0,
                 route_agent_proximity_nm=15.0,
                 agent_separation_nm=10.0):
        self.n_obstacles = n_obstacles
        self.n_intruders = n_intruders
        self.n_agents = n_agents
        self.sector_area_range = sector_area_range
        self.obstacle_area_range = obstacle_area_range
        self.goal_distance_range = goal_distance_range
        self.agent_speed = agent_speed
        self.intruder_speed_range = intruder_speed_range
        self.center = center
        self.agent_start_margin_nm = agent_start_margin_nm
        self.obstacle_clearance_nm = obstacle_clearance_nm
        self.route_agent_proximity_nm = route_agent_proximity_nm
        self.agent_separation_nm = agent_separation_nm

    # -- sector --------------------------------------------------------------
    def _generate_sector(self, rng):
        min_area, max_area = self.sector_area_range
        R = math.sqrt(max_area / math.pi)
        pts = [_random_point_on_circle(rng, R) for _ in range(3)]
        pts = _sort_points_clockwise(pts)
        while Polygon(pts).area < min_area:
            pts.append(_random_point_on_circle(rng, R))
            pts = _sort_points_clockwise(pts)
        return pts, Polygon(pts)

    # -- obstacles -----------------------------------------------------------
    def _generate_obstacles(self, rng, sector_poly):
        minx, miny, maxx, maxy = sector_poly.bounds
        obstacle_polys = []
        for _ in range(self.n_obstacles):
            for _attempt in range(100):
                target_area = rng.uniform(*self.obstacle_area_range)
                r_obs = math.sqrt(target_area / math.pi)
                x_scale = rng.uniform(1.0, 2.5)
                n_vert = int(rng.integers(3, 9))
                angles = np.sort(rng.uniform(0, 2 * math.pi, n_vert))
                radii = r_obs * rng.uniform(0.6, 1.4, n_vert)
                cx = rng.uniform(minx, maxx)
                cy = rng.uniform(miny, maxy)
                verts = [(cx + r * math.cos(a) * x_scale, cy + r * math.sin(a))
                         for a, r in zip(angles, radii)]
                poly = Polygon(verts)
                if not poly.is_valid:
                    continue
                if sector_poly.contains(poly):
                    obstacle_polys.append(poly)
                    break
        return obstacle_polys

    # -- agent ---------------------------------------------------------------
    def _generate_agent(self, rng, sector_poly, obstacle_polys, existing_starts=()):
        minx, miny, maxx, maxy = sector_poly.bounds
        obstacle_buffers = [o.buffer(self.obstacle_clearance_nm) for o in obstacle_polys]
        goal_min, goal_max = self.goal_distance_range

        def _valid(inner, p):
            pt = Point(p)
            if not inner.contains(pt):
                return False
            return not any(ob.contains(pt) for ob in obstacle_buffers)

        def _try(inner, buffers_on):
            for _ in range(1000):
                s = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if not (_valid(inner, s) if buffers_on else inner.contains(Point(s))):
                    continue
                if not _separated(s, existing_starts, self.agent_separation_nm):
                    continue
                g = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if not (_valid(inner, g) if buffers_on else inner.contains(Point(g))):
                    continue
                d_km = math.hypot(g[0] - s[0], g[1] - s[1]) * NM2KM
                if goal_min <= d_km <= goal_max:
                    return s, g
            return None

        placement = _try(sector_poly.buffer(-self.agent_start_margin_nm), True)
        if placement is None:
            # relax: smaller inward margin, drop the obstacle clearance
            # (start separation is kept — it prevents episodes that begin in intrusion)
            placement = _try(sector_poly.buffer(-self.agent_start_margin_nm / 2.0), False)
        if placement is None:
            raise ValueError(
                "ScenarioGenerator: could not place agent start/goal; "
                "check sector/obstacle/goal-distance/separation parameters."
            )
        start, goal = placement
        return start, goal

    def _generate_agents(self, rng, sector_poly, obstacle_polys):
        """Place ``n_agents`` (start, goal) pairs with mutually separated starts."""
        starts, goals = [], []
        for _ in range(self.n_agents):
            start, goal = self._generate_agent(rng, sector_poly, obstacle_polys,
                                               existing_starts=starts)
            starts.append(start)
            goals.append(goal)
        return list(zip(starts, goals))

    # -- intruder routes -----------------------------------------------------
    def _generate_routes(self, rng, sector_poly, obstacle_polys, start, goal):
        perimeter = sector_poly.exterior
        total = perimeter.length
        agent_line = LineString([start, goal])
        routes = []

        for i in range(self.n_intruders):
            chosen = None
            fallback = None
            for attempt in range(200):
                s0 = rng.uniform(0, total)
                s1 = rng.uniform(0, total)
                sep = abs(s1 - s0)
                sep = min(sep, total - sep)
                if sep < 0.25 * total:
                    continue
                e = perimeter.interpolate(s0)
                x = perimeter.interpolate(s1)
                entry = (e.x, e.y)
                exit_ = (x.x, x.y)

                if rng.random() < 0.5:
                    mx = 0.5 * (entry[0] + exit_[0])
                    my = 0.5 * (entry[1] + exit_[1])
                    dxr = exit_[0] - entry[0]
                    dyr = exit_[1] - entry[1]
                    seg = math.hypot(dxr, dyr) or 1.0
                    perp = (-dyr / seg, dxr / seg)
                    j = rng.uniform(-15.0, 15.0)
                    mid = (mx + perp[0] * j, my + perp[1] * j)
                    pts = [entry, mid, exit_]
                else:
                    pts = [entry, exit_]

                route_line = LineString(pts)
                if fallback is None:
                    fallback = pts
                hits = any(route_line.intersects(o) for o in obstacle_polys)
                near = route_line.distance(agent_line) < self.route_agent_proximity_nm
                if attempt < 100:
                    if hits or not near:
                        continue
                else:
                    if hits:
                        continue
                chosen = pts
                break

            if chosen is None:
                if fallback is None:
                    e = perimeter.interpolate(0.0)
                    x = perimeter.interpolate(0.5 * total)
                    fallback = [(e.x, e.y), (x.x, x.y)]
                chosen = fallback

            route_line = LineString(chosen)
            frac = rng.uniform(0.0, 0.4)
            sp = route_line.interpolate(frac, normalized=True)
            waypoints_nm = [(sp.x, sp.y)] + list(chosen[1:])
            waypoints_ll = [_nm_to_latlon(self.center, p) for p in waypoints_nm]
            speed = float(rng.uniform(*self.intruder_speed_range))
            routes.append(Route(ac_id=f"AC{i + 1}", waypoints=waypoints_ll, speed=speed))

        return routes

    # -- assemble ------------------------------------------------------------
    def generate(self, rng):
        sector_nm, sector_poly = self._generate_sector(rng)
        obstacle_polys = self._generate_obstacles(rng, sector_poly)
        agent_pairs = self._generate_agents(rng, sector_poly, obstacle_polys)
        # intruder routes are biased toward the first agent's start->goal line
        routes = self._generate_routes(rng, sector_poly, obstacle_polys, *agent_pairs[0])

        sector_ll = [_nm_to_latlon(self.center, p) for p in sector_nm]
        obstacles = []
        for poly in obstacle_polys:
            verts_nm = list(poly.exterior.coords)[:-1]   # unclosed ring
            kind = "restricted" if rng.random() < 0.5 else "weather"
            obstacles.append(Obstacle(
                vertices=[_nm_to_latlon(self.center, v) for v in verts_nm],
                kind=kind,
            ))

        agents = [
            AgentSpec(
                ac_id=ac_id,
                start=_nm_to_latlon(self.center, start),
                goal=_nm_to_latlon(self.center, goal),
                heading=_bearing_nm(start, goal),
                speed=self.agent_speed,
            )
            for ac_id, (start, goal) in zip(agent_callsigns(self.n_agents), agent_pairs)
        ]

        return Scenario(
            center=tuple(self.center),
            sector=sector_ll,
            obstacles=obstacles,
            agents=agents,
            intruder_routes=routes,
        )
