import numpy as np
import pygame

import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

pop_array = np.genfromtxt('bluesky_gym/envs/data/population_1km.csv', delimiter = ' ')

colormap = cm.viridis 
colormap.set_bad('k')
norm=LogNorm(vmin=100,vmax=100000)

normalized_array = norm(pop_array)

colored_array = colormap(normalized_array)[:, :, :3]  # Apply colormap and drop alpha channel
colored_array = (colored_array * 255).astype(np.uint8)  # Convert to 0-255 for Pygame

# Convert the array to a Pygame surface
surface = pygame.surfarray.make_surface(np.transpose(colored_array, (1, 0, 2)))

# Set up the Pygame window
screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("2D Array with LogNorm")

# Scale the surface to fit the screen
surface = pygame.transform.scale(surface, (screen_width, screen_height))
screen.blit(surface, (0,0))
pygame.display.update()

import code
code.interact(local=locals())