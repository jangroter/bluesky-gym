import numpy as np
import math

nm = 1852.
nm2km = nm/1000.
ft  = 0.3048  

circlerad = 250
max_action = 30 

schiphol = [52.3068953,4.760783]

circle_lat = 52.3068953
circle_lon = 4.760783

re      = 6371000.  # radius earth [m]

MAX_DIS_NEXT_WPT = 30 #km
MIN_DIS_NEXT_WPT = 5 #km

def get_state(lat,lon):
    brg, dist = kwikqdrdist(circle_lat, circle_lon, lat, lon)

    x = np.sin(np.radians(brg))*dist*nm2km / circlerad
    y = np.cos(np.radians(brg))*dist*nm2km / circlerad
    observation = {
            "x" : np.array([x]),
            "y" : np.array([y])
        }
    return observation

def do_action(action,lat,lon):

    distance = max(max(action[0]*MAX_DIS_NEXT_WPT, action[1]*MAX_DIS_NEXT_WPT), MIN_DIS_NEXT_WPT)
    bearing = math.atan2(action[0],action[1])

    ac_lat = np.deg2rad(lat)
    ac_lon = np.deg2rad(lon)

    new_lat = get_new_latitude(bearing,ac_lat,distance)
    new_lon = get_new_longitude(bearing,ac_lon,ac_lat,new_lat,distance)
    
    return new_lat, new_lon

def do_action_alt(action,lat,lon,alt):

    action = action[0]
    distance = max(math.sqrt(action[0]**2 + action[1]**2)*max_action,2)
    bearing = math.atan2(action[0],action[1])

    slope = max(action[2],0)
    slope = np.deg2rad(slope*-3)
    
    alt = (alt + np.tan(slope)*distance*1000/ft)
    alt = max(alt,5000)

    ac_lat = np.deg2rad(lat)
    ac_lon = np.deg2rad(lon)

    new_lat = get_new_latitude(bearing,ac_lat,distance)
    new_lon = get_new_longitude(bearing,ac_lon,ac_lat,new_lat,distance)
    
    return new_lat, new_lon, alt

def get_latlon_state(state):
    distance = np.sqrt(state[0]**2 + state[1]**2)*circlerad
    bearing = math.atan2(state[0],state[1])

    lat = get_new_latitude(bearing,np.deg2rad(circle_lat),distance)
    lon = get_new_longitude(bearing,np.deg2rad(circle_lon),np.deg2rad(circle_lat),lat,distance)   

    return np.rad2deg(lat), np.rad2deg(lon)

def get_new_latitude(bearing,lat,radius):
    R = re/1000.
    return math.asin( math.sin(lat)*math.cos(radius/R) +\
            math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
    
def get_new_longitude(bearing,lon,lat1,lat2,radius):
    R = re/1000.
    return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                    math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))

def kwikqdrdist(lata, lona, latb, lonb):
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(((lonb - lona)+180)%360-180)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle / nm

    qdr     = np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360.

    return qdr, dist

def get_spawn(bearing, radius = circlerad):
    
    bearing = np.deg2rad(bearing)

    lat = np.deg2rad(circle_lat)
    lon = np.deg2rad(circle_lon)
    
    latspawn = np.rad2deg(get_new_latitude(bearing,lat, radius))
    lonspawn = np.rad2deg(get_new_longitude(bearing, lon, lat, np.deg2rad(latspawn), radius))
    
    return latspawn, lonspawn

def get_xy(lat,lon):
    brg, dist = kwikqdrdist(schiphol[0], schiphol[1], lat, lon)

    x = np.sin(np.radians(brg))*dist*nm
    y = np.cos(np.radians(brg))*dist*nm

    return x,y

def get_z(alt):
    """ Returns z between -1 and 1, input alt in feet"""
    z = (alt / (30000))*2 - 1 
    return z

def get_alt(z):
    """ Returns alt between 5000 and 30000, input z in -1 and 1"""
    alt = 30000*(z+1)/2
    return alt