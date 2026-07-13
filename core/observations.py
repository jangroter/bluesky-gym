"""
All logic related to constructing observations from BlueSky sim states.
Functions here are reused within gym, zoo and sandbox.
"""

import numpy as np
import bluesky as bs
from gymnasium import spaces

from core.scenario import resample_perimeter

NM2KM = 1.852


def _bound_angle(angle_deg):
    return ((angle_deg + 180) % 360) - 180

def _cpa(ac_idx, int_idx):
    """Return (tcpa in s, dcpa in m) for a pair of aircraft indices."""
    qdr, dist = bs.tools.geo.kwikqdrdist(
        bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
        bs.traf.lat[int_idx], bs.traf.lon[int_idx],
    )
    dist_m = dist * NM2KM * 1000
    dx = dist_m * np.cos(np.deg2rad(qdr))
    dy = dist_m * np.sin(np.deg2rad(qdr))

    vx_own = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.tas[ac_idx]
    vy_own = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.tas[ac_idx]
    vx_int = np.cos(np.deg2rad(bs.traf.hdg[int_idx])) * bs.traf.tas[int_idx]
    vy_int = np.sin(np.deg2rad(bs.traf.hdg[int_idx])) * bs.traf.tas[int_idx]

    dvx = vx_int - vx_own
    dvy = vy_int - vy_own
    dv2 = dvx**2 + dvy**2

    if dv2 < 1e-6:
        return np.inf, dist_m

    tcpa = -(dx * dvx + dy * dvy) / dv2
    if tcpa < 0:
        return np.inf, dist_m

    dcpa = np.sqrt((dx + tcpa * dvx)**2 + (dy + tcpa * dvy)**2)
    return tcpa, dcpa

def _dist_to_others(own_lat, own_lon, other_lats, other_lons):
    """Distance [nm] from one point to each of N other points.

    kwikdist_matrix isn't usable here: it always zeroes the diagonal
    (i == j) regardless of whether lat1[i]/lat2[i] actually coincide, since
    it's built for self-vs-self matrices (e.g. all-aircraft conflict
    detection), not a fixed point against a different set of N points.
    Plain kwikdist broadcasts correctly once both sides are equal-length
    arrays, so the own point is simply repeated N times.
    """
    n = len(other_lats)
    own_lat_arr = np.full(n, own_lat, dtype=np.float64)
    own_lon_arr = np.full(n, own_lon, dtype=np.float64)
    dists = bs.tools.geo.kwikdist(
        own_lat_arr, own_lon_arr,
        np.asarray(other_lats, dtype=np.float64), np.asarray(other_lons, dtype=np.float64),
    )
    return np.asarray(dists).flatten()

def _sort_by_distance(ac_idx, other_idx):
    if not other_idx:
        return []
    dists = _dist_to_others(
        bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
        bs.traf.lat[other_idx], bs.traf.lon[other_idx],
    )
    return [other_idx[int(i)] for i in np.argsort(dists)]

def _sort_by_tcpa(ac_idx, other_idx):
    tcpa_vals = [_cpa(ac_idx, i)[0] for i in other_idx]
    return [other_idx[i] for i in np.argsort(tcpa_vals)]

def _sort_by_dcpa(ac_idx, other_idx):
    dcpa_vals = [_cpa(ac_idx, i)[1] for i in other_idx]
    return [other_idx[i] for i in np.argsort(dcpa_vals)]

def _sort_points_by_distance(own_lat, own_lon, point_lats, point_lons):
    """Sort static lat/lon points by distance from own position. Returns sorted indices."""
    if not len(point_lats):
        return []
    dists = _dist_to_others(own_lat, own_lon, point_lats, point_lons)
    return [int(i) for i in np.argsort(dists)]

class DriftObservation:
    """
    Drift angle from ownship heading to a target heading.
    Drift is decomposed into cos/sin components.

    Parameters
    ----------
    None
    """
    def space(self):
        return {
            "cos_drift": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64),
            "sin_drift": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64),
        }

    def observe(self, ac_id, target_hdg):
        ac_idx = bs.traf.id2idx(ac_id)
        drift_rad = np.deg2rad(_bound_angle(bs.traf.hdg[ac_idx] - target_hdg))
        return {
            "cos_drift": np.array([np.cos(drift_rad)]),
            "sin_drift": np.array([np.sin(drift_rad)]),
        }

class WaypointObservation:
    """
    Drift angle and distance to n waypoints from ownship.

    Waypoints are sorted before filling slots, so the agent always sees the
    most relevant ones first. ``last_order`` is set after every ``observe``
    call so the environment can reorder any associated flag arrays (e.g.
    waypoint_reached) to match.

    Parameters
    ----------
    n : int
        Number of waypoint slots. Extra waypoints are dropped; missing slots
        are zero-padded.
    sort_by : str or callable
        "distance" or a callable with signature
        (own_lat, own_lon, point_lats, point_lons) -> list of sorted indices.
    distance_norm : float
        Normalization divisor for waypoint distance (km).
    status: bool
        Boolean flag if the waypoint has been reached already or not. 
        Used or environments with multiple waypoints that should be 
        visited once each.
    """

    SORTERS = {
        "distance": _sort_points_by_distance,
    }

    def __init__(self, n=1, sort_by="distance", distance_norm=150.0, include_status=False):
        if callable(sort_by):
            self._sorter = sort_by
        elif sort_by in self.SORTERS:
            self._sorter = self.SORTERS[sort_by]
        else:
            raise ValueError(f"unknown sort_by: {sort_by!r}, choices: {list(self.SORTERS)}")
        self.n = n
        self.distance_norm = distance_norm
        self.include_status = include_status

    def space(self):
        shape = (self.n,)
        space = {
            "cos_drift": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "sin_drift": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
        }
        if self.include_status:
            space["waypoint_status"] = spaces.Box(0, 1.0, shape=shape, dtype=np.float64)
        return space

    def observe(self, ac_id, wpt_lats, wpt_lons, reached_flags=None):
        if self.include_status and reached_flags is None:
            raise ValueError("include_status=True requires reached_flags")
        ac_idx = bs.traf.id2idx(ac_id)

        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]

        self.last_order = self._sorter(own_lat, own_lon, wpt_lats, wpt_lons)[:self.n]

        cos_drift = np.zeros(self.n)
        sin_drift = np.zeros(self.n)
        distances = np.zeros(self.n)
        reached = np.zeros(self.n) if self.include_status else None

        for slot, i in enumerate(self.last_order):
            qdr, dist = bs.tools.geo.kwikqdrdist(own_lat, own_lon, wpt_lats[i], wpt_lons[i])
            drift_rad = np.deg2rad(_bound_angle(own_hdg - qdr))
            cos_drift[slot] = np.cos(drift_rad)
            sin_drift[slot] = np.sin(drift_rad)
            distances[slot] = dist * NM2KM
            if reached_flags is not None:
                reached[slot] = reached_flags[i]

        obs = {
            "cos_drift": cos_drift,
            "sin_drift": sin_drift,
            "waypoint_distance": distances / self.distance_norm,
        }
        if self.include_status:
            obs["waypoint_status"] = reached
        return obs

class IntruderObservation:
    """
    Relative state of the n nearest (or most urgent) intruder aircraft.
    Slots beyond the actual intruder count are zero-padded.

    Parameters
    ----------
    n : int
        Number of intruder slots in the observation.
    sort_by : str or callable
        "distance", "tcpa", or "dcpa". Pass a callable with signature
        (ac_idx, other_idx) -> ordered list of indices for custom strategies.
    frame : str
        Reference frame used for all relative position and velocity outputs.

        ``"body"`` — Ownship body frame (default).

            All vectors are rotated so that the x-axis points along the
            ownship nose and the y-axis points to the starboard (right) wing.

            Aviation convention (heading measured clockwise from North) is
            preserved throughout: a bearing of ``brg`` degrees from the
            ownship maps to body-frame angle ``brg - own_hdg``.

            Sign conventions::

                +x  →  ahead of ownship (nose direction)
                +y  →  right of ownship (starboard)

            Rotation from global NE to body frame::

                x =  north * cos(hdg) + east * sin(hdg)
                y = -north * sin(hdg) + east * cos(hdg)

            Equivalently for position: x = dist * cos(brg - hdg),
                                       y = dist * sin(brg - hdg).

            Observation keys: ``x_r``, ``y_r``, ``vx_r``, ``vy_r``.

            Use this frame when the policy should be heading-invariant: the
            same physical geometry always produces the same observation,
            reducing the state space the agent must explore.

        ``"global"`` — Fixed geographic North-East frame.

            Vectors are expressed directly in the world frame; no heading
            rotation is applied.  Both position and velocity use aviation
            North-East decomposition::

                north = dist * cos(brg)   /   vn = cos(hdg) * tas
                east  = dist * sin(brg)   /   ve = sin(hdg) * tas

            Sign conventions::

                +north_r / +vn_r  →  geographic North
                +east_r  / +ve_r  →  geographic East

            Observation keys: ``north_r``, ``east_r``, ``vn_r``, ``ve_r``.

            Use this frame when the policy also receives an explicit ownship
            heading observation so it can reason about ego-relative geometry
            from global coordinates.

    include_vertical : bool
        Adds altitude_difference and vz_difference when True.
    pos_norm : float
        Normalization divisor for position components (metres).
    spd_norm : float
        Normalization divisor for velocity components (m/s).
    dist_norm : float
        Normalization divisor for intruder_distance (NM).
    alt_norm : float
        Normalization divisor for altitude_difference (m). Only used when
        include_vertical=True.
    """

    SORTERS = {
        "distance": _sort_by_distance,
        "tcpa": _sort_by_tcpa,
        "dcpa": _sort_by_dcpa,
    }

    def __init__(self, n=5, sort_by="distance", frame="body", include_vertical=False,
                 pos_norm=1_000_000, spd_norm=150, dist_norm=250, alt_norm=3000):
        if callable(sort_by):
            self._sorter = sort_by
        elif sort_by in self.SORTERS:
            self._sorter = self.SORTERS[sort_by]
        else:
            raise ValueError(f"unknown sort_by: {sort_by!r}, choices: {list(self.SORTERS)}")
        if frame not in ("body", "global"):
            raise ValueError(f"unknown frame: {frame!r}, choices: ('body', 'global')")
        self.n = n
        self.frame = frame
        self.include_vertical = include_vertical
        self.pos_norm = pos_norm
        self.spd_norm = spd_norm
        self.dist_norm = dist_norm
        self.alt_norm = alt_norm

    def space(self):
        shape = (self.n,)
        if self.frame == "body":
            pos_vel = {
                "x_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
                "y_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
                "vx_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
                "vy_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            }
        else:
            pos_vel = {
                "north_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
                "east_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
                "vn_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
                "ve_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            }
        s = {
            **pos_vel,
            "cos_track": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "sin_track": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "intruder_distance": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
        }
        if self.include_vertical:
            s["altitude_difference"] = spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
            s["vz_difference"] = spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
        return s

    def observe(self, ac_id):
        ac_idx = bs.traf.id2idx(ac_id)

        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]
        own_tas = bs.traf.tas[ac_idx]
        own_hdg_rad = np.deg2rad(own_hdg)
        # Ownship velocity in global NE frame
        vn_own = np.cos(own_hdg_rad) * own_tas
        ve_own = np.sin(own_hdg_rad) * own_tas

        other_idx = [i for i in range(bs.traf.ntraf) if i != ac_idx]
        order = self._sorter(ac_idx, other_idx)[:self.n]

        pos1 = np.zeros(self.n)
        pos2 = np.zeros(self.n)
        vel1 = np.zeros(self.n)
        vel2 = np.zeros(self.n)
        cos_track = np.zeros(self.n)
        sin_track = np.zeros(self.n)
        distances = np.zeros(self.n)
        alt_diff = np.zeros(self.n) if self.include_vertical else None
        vz_diff = np.zeros(self.n) if self.include_vertical else None

        for slot, i in enumerate(order):
            brg, dist = bs.tools.geo.kwikqdrdist(own_lat, own_lon, bs.traf.lat[i], bs.traf.lon[i])
            dist_m = dist * NM2KM * 1000

            # Intruder velocity in global NE frame
            int_hdg_rad = np.deg2rad(bs.traf.hdg[i])
            vn_int = np.cos(int_hdg_rad) * bs.traf.tas[i]
            ve_int = np.sin(int_hdg_rad) * bs.traf.tas[i]

            # Relative velocity in global NE frame
            vn_r = vn_int - vn_own
            ve_r = ve_int - ve_own

            if self.frame == "body":
                # Rotate position into body frame: x=nose, y=starboard
                rel_rad = np.deg2rad(brg - own_hdg)
                pos1[slot] = dist_m * np.cos(rel_rad)
                pos2[slot] = dist_m * np.sin(rel_rad)
                # Rotate velocity into body frame
                vel1[slot] =  vn_r * np.cos(own_hdg_rad) + ve_r * np.sin(own_hdg_rad)
                vel2[slot] = -vn_r * np.sin(own_hdg_rad) + ve_r * np.cos(own_hdg_rad)
            else:
                # Global NE frame: north = cos(brg), east = sin(brg)
                pos1[slot] = dist_m * np.cos(np.deg2rad(brg))
                pos2[slot] = dist_m * np.sin(np.deg2rad(brg))
                vel1[slot] = vn_r
                vel2[slot] = ve_r

            track = np.arctan2(vel2[slot], vel1[slot])
            cos_track[slot] = np.cos(track)
            sin_track[slot] = np.sin(track)
            distances[slot] = dist

            if self.include_vertical:
                alt_diff[slot] = bs.traf.alt[i] - bs.traf.alt[ac_idx]
                vz_diff[slot] = bs.traf.vs[i] - bs.traf.vs[ac_idx]

        if self.frame == "body":
            pos_vel = {
                "x_r": pos1 / self.pos_norm,
                "y_r": pos2 / self.pos_norm,
                "vx_r": vel1 / self.spd_norm,
                "vy_r": vel2 / self.spd_norm,
            }
        else:
            pos_vel = {
                "north_r": pos1 / self.pos_norm,
                "east_r": pos2 / self.pos_norm,
                "vn_r": vel1 / self.spd_norm,
                "ve_r": vel2 / self.spd_norm,
            }

        obs = {
            **pos_vel,
            "cos_track": cos_track,
            "sin_track": sin_track,
            "intruder_distance": distances / self.dist_norm,
        }
        if self.include_vertical:
            obs["altitude_difference"] = alt_diff / self.alt_norm
            obs["vz_difference"] = vz_diff
        return obs

class OwnAirspeedObservation:
    """
    Own aircraft airspeed.

    Parameters
    ----------
    spd_mean : float
        Mean for normalization (m/s).
    spd_std : float
        Standard deviation for normalization (m/s).
    """

    def __init__(self, spd_mean=0.0, spd_std=150.0):
        self.spd_mean = spd_mean
        self.spd_std = spd_std

    def space(self):
        return {"airspeed": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)}

    def observe(self, ac_id):
        ac_idx = bs.traf.id2idx(ac_id)
        return {"airspeed": np.array([(bs.traf.tas[ac_idx] - self.spd_mean) / self.spd_std])}

class OwnAltitudeObservation:
    """
    Own aircraft altitude and vertical speed.

    Parameters
    ----------
    alt_mean, alt_std : float
        Mean and std for altitude normalization (m).
    vz_mean, vz_std : float
        Mean and std for vertical speed normalization (m/s).
    """

    def __init__(self, alt_mean=1500.0, alt_std=3000.0, vz_mean=0.0, vz_std=5.0):
        self.alt_mean = alt_mean
        self.alt_std = alt_std
        self.vz_mean = vz_mean
        self.vz_std = vz_std

    def space(self):
        return {
            "altitude": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }

    def observe(self, ac_id):
        ac_idx = bs.traf.id2idx(ac_id)
        return {
            "altitude": np.array([(bs.traf.alt[ac_idx] - self.alt_mean) / self.alt_std]),
            "vz": np.array([(bs.traf.vs[ac_idx] - self.vz_mean) / self.vz_std]),
        }

class TargetAltitudeObservation:
    """
    Normalized target altitude, supplied by the environment at each step.
    The environment owns the target value; this class only handles the
    BlueSky read and normalization.

    Parameters
    ----------
    alt_mean, alt_std : float
        Mean and std matching those used in OwnAltitudeObservation so that
        own altitude and target altitude live in the same scale.
    """

    def __init__(self, alt_mean=1500.0, alt_std=3000.0):
        self.alt_mean = alt_mean
        self.alt_std = alt_std

    def space(self):
        return {"target_altitude": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)}

    def observe(self, target_alt):
        return {"target_altitude": np.array([(target_alt - self.alt_mean) / self.alt_std])}

class RunwayDistanceObservation:
    """
    Distance from ownship to a fixed runway point, expressed as remaining
    distance (positive = before runway, negative = past it).

    Parameters
    ----------
    rwy_lat, rwy_lon : float
        Runway threshold coordinates in degrees.
    default_distance : float
        Maximum expected distance to runway at episode start (km). Used to
        compute remaining distance: remaining = default_distance - actual_distance.
    dist_mean, dist_std : float
        Mean and std for normalization (km).
    """

    def __init__(self, rwy_lat, rwy_lon, default_distance=200.0,
                 dist_mean=100.0, dist_std=200.0):
        self.rwy_lat = rwy_lat
        self.rwy_lon = rwy_lon
        self.default_distance = default_distance
        self.dist_mean = dist_mean
        self.dist_std = dist_std

    def space(self):
        return {"runway_distance": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)}

    def observe(self, ac_id):
        ac_idx = bs.traf.id2idx(ac_id)
        dist_km = bs.tools.geo.kwikdist(
            self.rwy_lat, self.rwy_lon,
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
        ) * NM2KM
        remaining = self.default_distance - dist_km
        return {"runway_distance": np.array([(remaining - self.dist_mean) / self.dist_std])}

class ObstacleObservation:
    """
    Bearing, distance, and radius of n static circular obstacles.
    Obstacle positions and radii are supplied by the environment.
    Obstacles are sorted before filling slots; ``last_order`` is updated on
    every ``observe`` call so the environment can reorder associated data.

    Parameters
    ----------
    n : int
        Number of obstacle slots (must match what the environment generates).
    sort_by : str or callable
        "distance" or a callable with signature
        (own_lat, own_lon, point_lats, point_lons) -> list of sorted indices.
    distance_norm : float
        Normalization divisor for obstacle distances (km).
    radius_norm : float
        Normalization divisor for obstacle radii (NM).
    """

    SORTERS = {
        "distance": _sort_points_by_distance,
    }

    def __init__(self, n, sort_by="distance", distance_norm=170.0, radius_norm=1000.0):
        if callable(sort_by):
            self._sorter = sort_by
        elif sort_by in self.SORTERS:
            self._sorter = self.SORTERS[sort_by]
        else:
            raise ValueError(f"unknown sort_by: {sort_by!r}, choices: {list(self.SORTERS)}")
        self.n = n
        self.distance_norm = distance_norm
        self.radius_norm = radius_norm
        self.last_order = []

    def space(self):
        shape = (self.n,)
        return {
            "obstacle_distance": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "obstacle_cos_bearing": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "obstacle_sin_bearing": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "obstacle_radius": spaces.Box(0.0, np.inf, shape=shape, dtype=np.float64),
        }

    def observe(self, ac_id, obstacle_lats, obstacle_lons, obstacle_radii):
        ac_idx = bs.traf.id2idx(ac_id)

        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]

        self.last_order = self._sorter(own_lat, own_lon, obstacle_lats, obstacle_lons)[:self.n]

        distances = np.zeros(self.n)
        cos_bearing = np.zeros(self.n)
        sin_bearing = np.zeros(self.n)
        radii = np.zeros(self.n)

        for slot, k in enumerate(self.last_order):
            qdr, dist = bs.tools.geo.kwikqdrdist(own_lat, own_lon, obstacle_lats[k], obstacle_lons[k])
            bearing_rad = np.deg2rad(_bound_angle(own_hdg - qdr))
            distances[slot] = dist * NM2KM
            cos_bearing[slot] = np.cos(bearing_rad)
            sin_bearing[slot] = np.sin(bearing_rad)
            radii[slot] = obstacle_radii[k]

        return {
            "obstacle_distance": distances / self.distance_norm,
            "obstacle_cos_bearing": cos_bearing,
            "obstacle_sin_bearing": sin_bearing,
            "obstacle_radius": radii / self.radius_norm,
        }

class SectorBoundaryObservation:
    """
    Distance and bearing to n points sampled evenly along the sector boundary,
    plus a flag for whether the ownship is currently inside the sector.

    The environment passes the sector polygon vertices (which vary in count per
    episode); this class resamples them to exactly ``n`` points, evenly spaced
    by arc length, so the observation has a fixed shape. The points are kept in
    perimeter order — not sorted — so the boundary shape is presented stably.
    Resampling is deterministic, so the same boundary is seen every step.

    Parameters
    ----------
    n : int
        Number of boundary points in the observation.
    distance_norm : float
        Normalization divisor for boundary distances (km).
    """

    def __init__(self, n=12, distance_norm=170.0):
        self.n = n
        self.distance_norm = distance_norm

    def space(self):
        shape = (self.n,)
        return {
            "sector_point_distance": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "sector_point_cos_bearing": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "sector_point_sin_bearing": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "inside_sector": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
        }

    def observe(self, ac_id, sector_vertices, inside):
        ac_idx = bs.traf.id2idx(ac_id)

        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]

        boundary = resample_perimeter(sector_vertices, self.n)

        distances = np.zeros(self.n)
        cos_bearing = np.zeros(self.n)
        sin_bearing = np.zeros(self.n)

        for i in range(self.n):
            qdr, dist = bs.tools.geo.kwikqdrdist(own_lat, own_lon, boundary[i, 0], boundary[i, 1])
            bearing_rad = np.deg2rad(_bound_angle(own_hdg - qdr))
            distances[i] = dist * NM2KM
            cos_bearing[i] = np.cos(bearing_rad)
            sin_bearing[i] = np.sin(bearing_rad)

        return {
            "sector_point_distance": distances / self.distance_norm,
            "sector_point_cos_bearing": cos_bearing,
            "sector_point_sin_bearing": sin_bearing,
            "inside_sector": np.array([1.0 if inside else 0.0]),
        }
