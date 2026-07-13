"""
Deterministic geo helpers used for competition scoring.

This is a deliberate pure-Python copy of BlueSky's ``kwikqdrdist``. BlueSky
silently swaps in compiled C implementations of its geo functions when they are
available, and those results do not match the pure-Python ones bit-for-bit. All
competition *metric* computations (intrusions, distances-to-goal, etc.) route
through this copy so that scores are reproducible across machines regardless of
whether a given install has the compiled extensions. Observation code, which is
competitor-owned and not scored, may use ``bs.tools.geo`` directly.
"""

import numpy as np


def kwikqdrdist(lata, lona, latb, lonb):
    """Gives quick and dirty qdr[deg] and dist [nm]
       from lat/lon. (note: does not work well close to poles)"""
    
    nm      = 1852.  # m       1 nautical mile
    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(((lonb - lona)+180)%360-180)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle / nm

    qdr     = np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360.

    return qdr, dist