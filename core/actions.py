"""
All logic related to translating normalized actions into BlueSky commands.
Functions here are reused within gym and zoo environments.

Discrete actions are considered symmetric, the resolution defined through n_discrete
will be mirrored around 0 and equally distributed over the interval [0,|1|]
"""

import numpy as np
import bluesky as bs
from gymnasium import spaces

MpS2Kt = 1.94384


def _bound_angle(angle_deg):
    return ((angle_deg + 180) % 360) - 180


def _discrete_to_float(action, n_discrete):
    """Map a Discrete(2*n+1) index to a float in [-1, 1]. Index n → 0.0."""
    return (int(action) - n_discrete) / n_discrete


def combine_action_spaces(action_classes):
    """Merge a list of action classes into a single flat action space.

    All continuous → Box(-1, 1, shape=(N,)).
    All discrete   → MultiDiscrete([n1, n2, ...]).
    Mixed          → raises ValueError.
    """
    individual = [a.space() for a in action_classes]
    types = {type(s) for s in individual}
    if types == {spaces.Box}:
        n = sum(s.shape[0] for s in individual)
        return spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float64)
    if types == {spaces.Discrete}:
        return spaces.MultiDiscrete([s.n for s in individual])
    raise ValueError(
        "combine_action_spaces requires all-continuous or all-discrete action classes, "
        f"got {[type(s).__name__ for s in individual]}"
    )


class HeadingAction:
    """
    Relative heading change action.

    The action is a normalized value in [-1, 1] (continuous) or a Discrete
    index (discrete). The ownship heading is queried from the current BlueSky
    simulation state and the delta ``action * d_heading`` is applied.

    Parameters
    ----------
    d_heading : float
        Maximum heading change in degrees (maps action ±1 to ±d_heading°).
    n_discrete : int or None
        None → continuous Box space. Integer N → Discrete(2*N+1) space where
        index N is always the zero/no-change action.
    """

    def __init__(self, d_heading, n_discrete=None):
        self.d_heading = d_heading
        self.n_discrete = n_discrete

    def space(self):
        if self.n_discrete is None:
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        return spaces.Discrete(2 * self.n_discrete + 1)

    def execute(self, ac_id, action):
        if self.n_discrete is not None:
            action = _discrete_to_float(action, self.n_discrete)
        ac_idx = bs.traf.id2idx(ac_id)
        new_hdg = _bound_angle(bs.traf.hdg[ac_idx] + float(action) * self.d_heading)
        bs.stack.stack(f"HDG {ac_id} {new_hdg}")


class SpeedAction:
    """
    Relative airspeed change action.

    The action is a normalized value in [-1, 1] (continuous) or a Discrete
    index (discrete). The ownship CAS is queried from the current BlueSky
    simulation state and ``action * d_speed`` (m/s) is added before issuing
    a SPD command in knots.

    Parameters
    ----------
    d_speed : float
        Maximum speed change in knots (maps action ±1 to ±d_speed kts).
    n_discrete : int or None
        None → continuous Box space. Integer N → Discrete(2*N+1) space.
    """

    def __init__(self, d_speed, n_discrete=None):
        self.d_speed = d_speed
        self.n_discrete = n_discrete

    def space(self):
        if self.n_discrete is None:
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        return spaces.Discrete(2 * self.n_discrete + 1)

    def execute(self, ac_id, action):
        if self.n_discrete is not None:
            action = _discrete_to_float(action, self.n_discrete)
        ac_idx = bs.traf.id2idx(ac_id)
        new_spd = (bs.traf.cas[ac_idx] + float(action) * self.d_speed / MpS2Kt) * MpS2Kt
        bs.stack.stack(f"SPD {ac_id} {new_spd}")


class VerticalAction:
    """
    Vertical speed action (climb/descend).

    The action is a normalized value in [-1, 1] (continuous) or a Discrete
    index (discrete). Positive action → climb, negative → descend. The target
    altitude selector is set to a very high/low value so the autopilot
    executes the commanded vertical speed without a conflicting altitude target.

    Parameters
    ----------
    vs_scale : float
        Vertical speed scale in m/s (maps action ±1 to ±vs_scale m/s).
    n_discrete : int or None
        None → continuous Box space. Integer N → Discrete(2*N+1) space.
    """

    def __init__(self, vs_scale, n_discrete=None):
        self.vs_scale = vs_scale
        self.n_discrete = n_discrete

    def space(self):
        if self.n_discrete is None:
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        return spaces.Discrete(2 * self.n_discrete + 1)

    def execute(self, ac_id, action):
        if self.n_discrete is not None:
            action = _discrete_to_float(action, self.n_discrete)
        ac_idx = bs.traf.id2idx(ac_id)
        vs = float(action) * self.vs_scale
        if vs >= 0:
            bs.traf.selalt[ac_idx] = 1_000_000
        else:
            bs.traf.selalt[ac_idx] = 0
        bs.traf.selvs[ac_idx] = vs


# ---------------------------------------------------------------------------
# Rate action stubs — needed for safe RL / control barrier function
# applications where the derivative of the state is commanded.
# execute() raises NotImplementedError until the BlueSky command mapping
# is confirmed.
# ---------------------------------------------------------------------------

class HeadingRateAction:
    """
    Heading rate-of-change action (placeholder).

    Parameters
    ----------
    d_heading_rate : float
        Maximum heading rate in deg/s (maps action ±1 to ±d_heading_rate°/s).
    n_discrete : int or None
        None → continuous Box space. Integer N → Discrete(2*N+1) space.
    """

    def __init__(self, d_heading_rate, n_discrete=None):
        self.d_heading_rate = d_heading_rate
        self.n_discrete = n_discrete

    def space(self):
        if self.n_discrete is None:
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        return spaces.Discrete(2 * self.n_discrete + 1)

    def execute(self, ac_id, action):
        raise NotImplementedError(
            "HeadingRateAction.execute() is not yet implemented — "
            "BlueSky heading-rate command mapping needs to be confirmed."
        )


class AccelerationAction:
    """
    Longitudinal acceleration action (placeholder).

    Parameters
    ----------
    d_accel : float
        Maximum acceleration in m/s² (maps action ±1 to ±d_accel m/s²).
    n_discrete : int or None
        None → continuous Box space. Integer N → Discrete(2*N+1) space.
    """

    def __init__(self, d_accel, n_discrete=None):
        self.d_accel = d_accel
        self.n_discrete = n_discrete

    def space(self):
        if self.n_discrete is None:
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        return spaces.Discrete(2 * self.n_discrete + 1)

    def execute(self, ac_id, action):
        raise NotImplementedError(
            "AccelerationAction.execute() is not yet implemented — "
            "BlueSky longitudinal acceleration command mapping needs to be confirmed."
        )


class VerticalAccelerationAction:
    """
    Vertical acceleration action (placeholder).

    Parameters
    ----------
    d_vert_accel : float
        Maximum vertical acceleration in m/s² (maps action ±1 to ±d_vert_accel m/s²).
    n_discrete : int or None
        None → continuous Box space. Integer N → Discrete(2*N+1) space.
    """

    def __init__(self, d_vert_accel, n_discrete=None):
        self.d_vert_accel = d_vert_accel
        self.n_discrete = n_discrete

    def space(self):
        if self.n_discrete is None:
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        return spaces.Discrete(2 * self.n_discrete + 1)

    def execute(self, ac_id, action):
        raise NotImplementedError(
            "VerticalAccelerationAction.execute() is not yet implemented — "
            "BlueSky vertical acceleration command mapping needs to be confirmed."
        )
