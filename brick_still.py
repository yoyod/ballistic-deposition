"""
Simulation of the Ballistic Deposition model
with random heights following a pareto law
and sticking paramer

This script simulates the fall of a number of bricks n_bricks
and outputs the figure of the 2d brick structure generated
"""

import numpy as np
import cv2

# Parameters
h_size = 100 # horizontal size
pareto_exp = 1.5 # exponent of pareto distribution
sticking = 1 # sticking (between 0 and 1)
boundary = "periodic" # boundary condition (periodic or free)
initial = "seed" # initial condition (seed, flat or random)
n_bricks = 10000



# Generate initial profile
def initialize(h_size, initial, pareto_exp = 1):
    if initial == "flat":
        profile = np.zeros(h_size)
    elif initial == "seed": 
        profile = np.zeros(h_size) - np.inf # height is -infinity everywhere except at 0
        profile[h_size // 2] = 0
    elif initial == "random":
        profile =  np.random.pareto(pareto_exp, h_size)
    else:
        print("Initial condition must be one of: 'flat', 'seed' or 'random'.")
    return profile


# determine the neighborhood to look at to compute the maximum
def determine_neighborhood(h_size, brick_location, sticky, boundary):

     # if the brick is not sticky consider only current column
    if sticky == 0:
        neighborhood = [brick_location]
    
    # otherwise check for boundary condition periodic
    elif boundary == "periodic":
        neighborhood = [brick_location -1, brick_location, (brick_location + 1) % h_size]

    # otherwise, for normal boundary conditions
    else:
        if brick_location == 0: # if the leftmost column is chosen
            neighborhood = [0,1]
        elif brick_location == h_size -1: # if the rightmost column is chosen
            neighborhood = [-2, -1]
        else:
            neighborhood = [brick_location-1, brick_location, brick_location+1]
    
    return neighborhood


# add one brick to the given profile
def one_step(h_size, pareto_exp, sticking, boundary, profile):
    brick_height = np.random.pareto(pareto_exp) # choose height of falling brick at random
    brick_location = np.random.randint(h_size) # choose location of falling brick at random
    sticky = np.random.binomial(1, sticking) # choose if brick is sticky (0 = no, 1 = yes)
    
    # determine the neighborhood to look at based on sticky = 0/1 or boundary = "free"/"periodic"
    neighborhood = determine_neighborhood(h_size, brick_location, sticky, boundary)
    max_height = np.amax(profile[neighborhood]) # compute max height in the neighborhood
    profile[brick_location] = max_height + brick_height # update height at brick_location
    return profile






# generate history of height profiles
profile = initialize(h_size, initial, pareto_exp)
history = [profile,]
for i in range(n_bricks):
    history.append(one_step(h_size, pareto_exp, sticking, boundary, np.copy(history[i])))
print(history[1000])


my_img = np.zeros((h_size, h_size, 3), dtype = "uint8")
cv2.imshow('Window', my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()