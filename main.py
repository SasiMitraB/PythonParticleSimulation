'''Ideal Gas Particle Based Simulation'''


#Importing the Nessecary Libraries
from animated_plots import generate_animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from numba import jit, njit
import os
import time
from concurrent.futures import ThreadPoolExecutor


number_of_balls = 1000
dimensions = np.asarray([500,500])
number_of_frames = 2000
delta_t = 1 # Update with the actual time step


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
    def __init__(self, position, velocity, acceleration):
        # Taking Position, velocity, acceleration as inputs and making them object variables
        self.pos = np.asarray(position)
        self.vel = np.asarray(velocity) 
        self.acc = acceleration

    def check_wall_collision(self):
        '''Check if Collisions occurs with the walls and implements elastic collision with the walls'''
        self.vel = wall_collision(self.pos, self.vel)


    def update(self):
        """
        Update the position, velocity, and handle collisions.
        """
        self.check_wall_collision()
        self.pos, self.vel, self.acc =  updateParameters(self.pos, self.vel, self.acc)
       
@jit(nopython = True)       
def wall_collision(position, velocity):
    # Find indices where position is less than or equal to 0 or greater than or equal to dimensions
    below_zero = position <= 0
    above_dimensions = position >= dimensions

    # Update velocities for the corresponding dimensions
    velocity[below_zero] = -velocity[below_zero]
    velocity[above_dimensions] = -velocity[above_dimensions]

    return velocity
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
    COLLISION_DISTANCE = 100
    
    # Calculate the distance between the two objects manually
    delta_pos = pos1 - pos2
    r = (np.sum(delta_pos**2))

    # Check if a collision occurs (distance is less than or equal to COLLISION_DISTANCE)
    if r <= COLLISION_DISTANCE:
        ELASTICITY_COEFFICIENT = 1
        # Update velocities using collision formula
        vel1_new = ((ELASTICITY_COEFFICIENT + 1) * vel1 + vel2 * (1 - ELASTICITY_COEFFICIENT)) / 2
        vel2_new = (vel1 * (1 - ELASTICITY_COEFFICIENT) + (1 + ELASTICITY_COEFFICIENT) * vel2) / 2

        return vel1_new, vel2_new

    return vel1, vel2


@jit(nopython = True)        
def calculate_norm(array):
    return np.sqrt(np.sum(array**2))
        
######################## Drawing Each Frame ##########################
@jit(forceobj = True)
def draw_frame_txt(frame_count):
    # Collect positions for all balls in this frame
    ball_positions = np.zeros((number_of_balls, 2))

    for ball_number in range(number_of_balls):
        ball = my_balls[ball_number]
        ball_positions[ball_number] = ball.pos
        for i in range(number_of_balls):

            if i == (1 - number_of_balls):
                pass
            else:
                for j in range(i + 1, number_of_balls):
                    a, b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel)
                    my_balls[i].vel = a
                    my_balls[j].vel = b

        ball.update()

    # Save ball positions as a text file
    filename = f"animation/{str(frame_count).zfill(zfilll)}.txt"
    np.savetxt(filename, ball_positions)

import numpy as np
import matplotlib.pyplot as plt

def draw_frame_png(frame_count):
    # Collect positions for all balls in this frame
    ball_positions = np.zeros((number_of_balls, 2))
    
    for i in range(number_of_balls):
        if i == (1 - number_of_balls):
            pass
        else:
            for j in range(i + 1, number_of_balls):
                a, b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel)
                my_balls[i].vel = a
                my_balls[j].vel = b

        my_balls[i].update()
        ball_positions[i] = my_balls[i].pos[:2]  # Take only the x and y components

    # Plot and save ball positions as a PNG file
    filename = f"animation/{str(frame_count).zfill(zfilll)}.png"

    # Create a 2D scatter plot
    plt.scatter(ball_positions[:, 0], ball_positions[:, 1])

    # Save the plot as a PNG file
    plt.savefig(filename)

    # Clear the plot for the next frame
    plt.close()


def draw_frame_3d(frame_count):
    # Collect positions for all balls in this frame
    ball_positions = np.zeros((number_of_balls, 3))
    
    for i in range(number_of_balls):

        if i==(1-number_of_balls):
            pass
        else:
            for j in range(i+1, number_of_balls):
                a,b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel)
                my_balls[i].vel = a
                my_balls[j].vel = b

        my_balls[i].update()
        ball_positions[i] = my_balls[i].pos


    # Plot and save ball positions as a PNG file
    filename = f"animation/{str(frame_count).zfill(zfilll)}.png"

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2])

    # Save the plot as a PNG file
    plt.savefig(filename)

    # Clear the plot for the next frame
    plt.close(fig)
    
    

def draw_frame_vels(frame_count):
    # Collect positions and velocities for all balls in this frame
    ball_positions = np.zeros((number_of_balls, 2))
    ball_velocities = np.zeros(number_of_balls)

    for i in range(number_of_balls):
        ball_positions[i] = my_balls[i].pos
        ball_velocities[i] = calculate_norm(my_balls[i].vel)
        if i == (1 - number_of_balls):
            pass
        else:
            for j in range(i + 1, number_of_balls):
                a, b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel)
                my_balls[i].vel = a
                my_balls[j].vel = b

        my_balls[i].update()
        

    # Plot and save ball positions and velocities as a PNG file
    filename = f"animation/{str(frame_count).zfill(zfilll)}.png"

    # Create a subplot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Scatter plot of positions
    axs[0].scatter(ball_positions[:, 0], ball_positions[:, 1], s = 0.5)
    axs[0].set_title('Ball Positions')

    # Histogram of velocities
    axs[1].hist(ball_velocities, bins=1000, color='skyblue', edgecolor='black')
    axs[1].set_title('Velocities Histogram')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(filename)

    # Clear the plot for the next frame
    plt.close(fig)


################################Generating the Ball Objects ##################################

# Making a list called my_balls where the balls are stored
my_balls = []

# Iterating 'number_of_balls' times and making a new ball, and appending it to the list
for i in range(number_of_balls):
    # Randomly generating position and velocity
    pos = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]
    vel = [random.random(), random.random()]
    acc = np.asarray([0,0])
    
    # Making a ball
    i_th_ball = Ball(pos, vel, acc )  # Creating an instance of the Ball class
    
    # Appending the ball to the list
    my_balls.append(i_th_ball)


################################ The Actual Simulation ##################################

#######Deleting all the files in animation folder 
folder_path = 'animation'

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Iterate through the files and delete each one
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

#Simulating and generating each individual plot

zfilll = len(str(number_of_balls)) + 2


for frame_count in range(number_of_frames):
    start_time = time.time()  # Record the start time for the frame generation
    
    draw_frame_png(frame_count)

    end_time = time.time()  # Record the end time for the frame generation
    duration = end_time - start_time
    
    print(f"Frame number {frame_count + 1} has been generated in {duration:.4f} seconds.")


print("Generating Video from the frames...")
generate_animation.generate_animation(folder_path, fps = 25)


