from animated_plots import generate_animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from numba import jit
import os
import time
from concurrent.futures import ThreadPoolExecutor
import pygame





number_of_balls = 1000
dimensions = [1920,1080]
number_of_frames = 10000
delta_t = 10 # Update with the actual time step


class Ball:
    """
    Class representing a ball in a 2D space.

    Attributes:
    - pos (numpy.ndarray): The position of the ball as a 2D numpy array [x, y].
    - vel (numpy.ndarray): The velocity of the ball as a 2D numpy array [vx, vy].
    - acc (numpy.ndarray): The acceleration of the ball as a 2D numpy array [ax, ay].

    Methods:
    - __init__(self, position, velocity, acceleration): Constructor to initialize the ball with
      a given position, velocity, and acceleration.
    - collide(self): Method to handle collisions with the walls and update the velocity accordingly.
    - update(self): Method to update the position, velocity, and handle collisions.

    Usage:
    1. Create an instance of the Ball class with initial position, velocity, and acceleration.
    2. Call the update method to simulate the motion of the ball, considering collisions.
    3. Access the attributes 'pos' and 'vel' to retrieve the current position and velocity.

    Note: This class assumes a 2D space, and collisions with the walls are handled by
          reversing the appropriate component of the velocity.

    Author: Sasi Mitra Behara
    Date: December 28, 2023
    """
    def __init__(self, position, velocity, acceleration, size):
        # Taking Position, velocity, acceleration as inputs and making them object variables
        self.pos = np.asarray(position)
        self.vel = np.asarray(velocity)
        self.acc = acceleration
        self.size = size
        self.color = np.random.randint(0, 256, size=3)



    def update(self):
        """
        Update the position, velocity, and handle collisions.
        """
        self.pos, self.vel, self.acc =  updateParameters(self.pos, self.vel, self.acc)
        
    def display(self):
        pygame.draw.circle(win, self.color, self.pos, self.size)
        

@jit(nopython=True)
def updateParameters(position, velocity, acceleration):
    velocity = velocity + acceleration * delta_t
    position = position + velocity * delta_t
    return position, velocity, acceleration

#Collision Logic for the Particles.
@jit(nopython=True)
def collide(pos1, pos2, vel1, vel2):
    """
    Handle collision between two objects.

    Parameters:
    - pos1 (numpy.ndarray): Position of the first object.
    - pos2 (numpy.ndarray): Position of the second object.
    - vel1 (numpy.ndarray): Velocity of the first object.
    - vel2 (numpy.ndarray): Velocity of the second object.

    Returns:
    - numpy.ndarray: Updated velocity of the first object.
    - numpy.ndarray: Updated velocity of the second object.

    Collision Formula:
    - Calculate the distance between the two objects manually using Euclidean norm:
      r = sqrt(sum((pos1 - pos2)**2))

    - Check if a collision occurs (distance is less than or equal to COLLISION_DISTANCE):
      if r <= COLLISION_DISTANCE:
        - Update velocities using collision formula:
          vel1_new = ((ELASTICITY_COEFFICIENT + 1) * vel1 + vel2 * (1 - ELASTICITY_COEFFICIENT)) / 2
          vel2_new = (vel1 * (1 - ELASTICITY_COEFFICIENT) + (1 + ELASTICITY_COEFFICIENT) * vel2) / 2

    The collision formula updates the velocities of two objects after a collision
    based on the conservation of linear momentum and kinetic energy. The coefficient
    of restitution 'e' determines the elasticity of the collision.
    """
    
    # Named constants
    COLLISION_DISTANCE = 200
    
    # Calculate the distance between the two objects manually
    delta_pos = pos1 - pos2
    r = np.sqrt(np.sum(delta_pos**2))

    # Check if a collision occurs (distance is less than or equal to COLLISION_DISTANCE)
    if r <= COLLISION_DISTANCE:
        ELASTICITY_COEFFICIENT = 1
        # Update velocities using collision formula
        vel1_new = ((ELASTICITY_COEFFICIENT + 1) * vel1 + vel2 * (1 - ELASTICITY_COEFFICIENT)) / 2
        vel2_new = (vel1 * (1 - ELASTICITY_COEFFICIENT) + (1 + ELASTICITY_COEFFICIENT) * vel2) / 2

        return vel1_new, vel2_new

    return vel1, vel2

def collide_parallel(i):
    ball = my_balls[i]
    for j in range(i + 1, number_of_balls):
        a, b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel)
        my_balls[i].vel = a
        my_balls[j].vel = b
        
        
#########################The Drawing Function########################


def draw(frame_count): #This is the function we'll be using to draw each frame.
    win.fill((0,0,0))
    start_time = time.time()  # Record the start time for the frame generation
    
    
    for i in range(number_of_balls):

        if i==(1-number_of_balls):
            pass
        else:
            for j in range(i+1, number_of_balls):
                a,b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel)
                my_balls[i].vel = a
                my_balls[j].vel = b

        my_balls[i].update()
        
        my_balls[i].display()


    end_time = time.time()  # Record the end time for the frame generation
    duration = end_time - start_time
    print(f"Frame number {frame_count + 1} has been generated in {duration:.4f} seconds.")


################################Generating the Ball Objects ##################################

# Making a list called my_balls where the balls are stored
my_balls = []

# Iterating 'number_of_balls' times and making a new ball, and appending it to the list
for i in range(number_of_balls):
    

    # Randomly generating position and velocity
    pos = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]
    vel = [random.random(), random.random()]
    acc = np.asarray([0,0])
    radius = 2
    # Making a ball
    i_th_ball = Ball(pos, vel, acc, radius )  # Creating an instance of the Ball class
    
    # Appending the ball to the list
    my_balls.append(i_th_ball)


################################ The Actual Simulation ##################################

#######Deleting all the files in animation folder 
folder_path = 'animation'



#Simulating and generating each individual plot







pygame.init() #Intializing the pygame engine


#Making a pygame window where we will be drawing stuff
win = pygame.display.set_mode(dimensions) 

'''fpsClock is basically a Clock that lets you control how fast the 
The program is going to be running'''
fpsClock = pygame.time.Clock() #Basically a clock.

run = True #Run is a variable that we can set to false when we want the window to close
frame_count = 0
while run: #As long as this while loop is true, we can see the window
    '''Pygame stores any interactions with the screen in a queue called 
	pygame.event.get(). We are going through that queue and looking for a specific
	event called pygame.QUIT, which is triggered when you press the x button in the
	window. When we set run to False, the while loop, and hence the window will 
	close'''

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        else:
            pass
    #Drawing to the screen each iteration
    frame_count += 1
    draw(frame_count)
    #Updating the display to reflect the changes made in the new frame
    pygame.display.flip()
    #Telling the clock to pause the code for 1/60th of a second, and then go ahead
    fpsClock.tick(60)

#We're out of the while loop, so it assumes we're done with the code.
#Might as well close the engine we intialized at the start of the code.
pygame.quit()