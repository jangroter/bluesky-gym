"""Area filter module"""
from weakref import WeakValueDictionary
import numpy as np
from matplotlib.path import Path
import bluesky_gym.envs.common.functions as fn

def checkIntersect(coordinates, lat0, lon0, lat1, lon1):
    line1 = Path(np.array([[lat0,lon0],[lat1,lon1]]))
    line2 = Path(np.reshape(coordinates, (len(coordinates) // 2, 2)))
    return line2.intersects_path(line1)

def poly_arc(lat_c,lon_c,radius,lowerbound,upperbound):
    # Input data is latctr,lonctr,radius[nm]
    # Convert circle into polyline list

    # Circle parameters
    Rearth = 6371000.0             # radius of the Earth [m]
    numPoints = 36                 # number of straight line segments that make up the circrle

    # Inputs
    lat0 = lat_c             # latitude of the center of the circle [deg]
    lon0 = lon_c            # longitude of the center of the circle [deg]
    Rcircle = radius * 1852.0  # radius of circle [NM]

    # Compute flat Earth correction at the center of the experiment circle
    coslatinv = 1.0 / np.cos(np.deg2rad(lat0))

    lower = np.deg2rad(lowerbound)
    upper = np.deg2rad(upperbound)
    # compute the x and y coordinates of the circle
    angles = np.linspace(lower,upper,numPoints)   # ,endpoint=True) # [rad]

    # Calculate the circle coordinates in lat/lon degrees.
    # Use flat-earth approximation to convert from cartesian to lat/lon.
    latCircle = lat0 + np.rad2deg(Rcircle * np.sin(angles) / Rearth)  # [deg]
    lonCircle = lon0 + np.rad2deg(Rcircle * np.cos(angles) * coslatinv / Rearth)  # [deg]

    # make the data array in the format needed to plot circle
    coordinates = np.empty(2 * numPoints, dtype=np.float32)  # Create empty array
    coordinates[0::2] = latCircle  # Fill array lat0,lon0,lat1,lon1....
    coordinates[1::2] = lonCircle

    return coordinates

def poly_arc2(runway='27'):
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
    IAF_DISTANCE = 20 #km, from FAF
    IAF_ANGLE = 60 #degrees, symmetrical around FAF

    num_points = 36 # number of straight line segments that make up the circrle

    faf_lat, faf_lon = fn.get_point_at_distance(RUNWAYS_SCHIPHOL_FAF[runway]['lat'],
                                                RUNWAYS_SCHIPHOL_FAF[runway]['lon'],
                                                FAF_DISTANCE,
                                                RUNWAYS_SCHIPHOL_FAF[runway]['track']-180)

    # Compute bounds for the merge angles from FAF
    cw_bound = ((RUNWAYS_SCHIPHOL_FAF[runway]['track']-180+ 360)%360) + (IAF_ANGLE/2)
    ccw_bound = ((RUNWAYS_SCHIPHOL_FAF[runway]['track']-180+ 360)%360) - (IAF_ANGLE/2)

    angles = np.linspace(cw_bound,ccw_bound,num_points)

    # Calculate the iaf coordinates in lat/lon degrees. Is an approximation of an arg starting routing to FAF
    # Use flat-earth approximation to convert from cartesian to lat/lon.
    # lat_iaf = faf_lat + np.rad2deg(IAF_DISTANCE * 1000 * np.sin(angles) / r_earth)  # [deg]
    # lon_iaf = faf_lon + np.rad2deg(IAF_DISTANCE * 1000 * np.cos(angles) * coslatinv / r_earth)  # [deg]
    lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)
    # make the data array in the format needed to plot circle
    Coordinates = np.empty(2 * num_points, dtype=np.float32)  # Create empty array
    Coordinates[0::2] = lat_iaf  # Fill array lat0,lon0,lat1,lon1....
    Coordinates[1::2] = lon_iaf
    return Coordinates