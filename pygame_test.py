import numpy as np
import pygame

import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

# Set center of environment and other variables
SCHIPHOL = [52.3068953,4.760783]
NM2KM = 1.852
FAF_DISTANCE = 10 #km
IAF_DISTANCE = 30 #km, from FAF
IAF_ANGLE = 60 #degrees, symmetrical around FAF
RUNWAY = [52.318302, 4.796412]
RUNWAY_TRACK = 267
DISTANCE_ENV = 500

# Set up the Pygame window
screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))

# Set screen coordinates
env_max_distance = np.sqrt((DISTANCE_ENV/2)**2 + (DISTANCE_ENV/2)**2) #km
lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(SCHIPHOL[0], SCHIPHOL[1], 315, env_max_distance/NM2KM)
screen_coords = [lat_ref_point,lon_ref_point]

# Determine coordinates of FAF and IAF in latlon reference frame
faf_lat, faf_lon = fn.get_point_at_distance(RUNWAY[0],
                                            RUNWAY[1],
                                            FAF_DISTANCE,
                                            RUNWAY_TRACK-180)

# Compute bounds for the merge angles from FAF
cw_bound = ((RUNWAY_TRACK-180+ 360)%360) + (IAF_ANGLE/2)
ccw_bound = ((RUNWAY_TRACK-180+ 360)%360) - (IAF_ANGLE/2)
angles = np.linspace(cw_bound,ccw_bound,36)
lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)

coordinates = np.empty(2 * 36, dtype=np.float32)  # Create empty array
coordinates[0::2] = lat_iaf  # Fill array lat0,lon0,lat1,lon1....
coordinates[1::2] = lon_iaf

line_arc = np.reshape(coordinates, (len(coordinates) // 2, 2))
line_restrict = np.array([[lat_iaf[0],lon_iaf[0]],[faf_lat, faf_lon],[lat_iaf[-1], lon_iaf[-1]]])


# Convert all coordinates to Pygame window reference frame
qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], line_arc[:,0], line_arc[:,1])
dis = dis*NM2KM
x_arc = ((np.sin(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*screen_width
y_arc = ((-np.cos(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*screen_width

line_arc_pg = list(zip(x_arc, y_arc))
line_arc_pg = [(float(x), float(y)) for x, y in line_arc_pg]

qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], line_restrict[:,0], line_restrict[:,1])
dis = dis*NM2KM
x_restrict = ((np.sin(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*screen_width
y_restrict = ((-np.cos(np.deg2rad(qdr))*dis)/DISTANCE_ENV)*screen_width

line_restrict_pg = list(zip(x_restrict, y_restrict))
line_restrict_pg = [(float(x), float(y)) for x, y in line_restrict_pg]
import code
code.interact(local=locals())

# pygame.draw.line(canvas,
#     (235, 52, 52),
#     (x_actor, y_actor),
#     (x_actor+heading_end_x, y_actor-heading_end_y),
#     width = 5
# )

# Create background image from population data
pop_array = np.genfromtxt('bluesky_gym/envs/data/population_1km.csv', delimiter = ' ')
x_index_min = int(((500000)/1000)-(DISTANCE_ENV/2))
x_index_max = int(((500000)/1000)+(DISTANCE_ENV/2))
y_index_min = int(((500000)/1000)-(DISTANCE_ENV/2))
y_index_max = int(((500000)/1000)+(DISTANCE_ENV/2))

x_array = np.genfromtxt('bluesky_gym/envs/data/x_array.csv',delimiter=' ')
y_array = np.genfromtxt('bluesky_gym/envs/data/y_array.csv',delimiter=' ')

pop_array = pop_array[y_index_min:y_index_max,x_index_min:x_index_max]
c_map = cm.get_cmap("Blues").copy()
c_map.set_bad((0.8,0.8,0.9))
norm=LogNorm(vmin=100,vmax=100000)

normalized_array = norm(pop_array)

colored_array = c_map(normalized_array)[:, :, :3]  # Apply colormap and drop alpha channel
colored_array = (colored_array * 255).astype(np.uint8)  # Convert to 0-255 for Pygame

# Convert the array to a Pygame surface
surface = pygame.surfarray.make_surface(np.transpose(colored_array, (1, 0, 2)))

# Scale the surface to fit the screen
surface = pygame.transform.scale(surface, (screen_width, screen_height))

# Draw the FAF and IAF
pygame.draw.lines(surface, (0, 0, 0), False, line_arc_pg, 3)
pygame.draw.lines(surface, (255, 0, 0), False, line_restrict_pg, 2)

screen.blit(surface, (0,0))
pygame.display.update()

import code
code.interact(local=locals())