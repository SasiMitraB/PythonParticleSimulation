'''Ideal Gas Particle Based Simulation'''


#Importing the Nessecary Libraries
from animated_plots import generate_animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from numpy import random
import pandas as pd
from numba import jit, njit
import os
import time
from concurrent.futures import ThreadPoolExecutor


number_of_balls = 1000
dimensions = np.asarray([500,500])
number_of_frames = 200
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
    def __init__(self, position, velocity, acceleration, radius):
        # Taking Position, velocity, acceleration as inputs and making them object variables
        self.pos = np.asarray(position)
        self.vel = np.asarray(velocity) * 10
        self.acc = acceleration
        self.r = radius

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
def compute_acceleration(position):
    # Placeholder function to compute acceleration based on position
    # Modify this function based on your specific force model
    return np.zeros_like(position)

@jit(nopython=True)
def rk4_step(position, velocity, acceleration, delta_t):
    """
    Perform a single RK4 step for updating position and velocity.
    
    Parameters:
    position (ndarray): An array of particle positions.
    velocity (ndarray): An array of particle velocities.
    acceleration (ndarray): An array of particle accelerations.
    delta_t (float): Time step.
    
    Returns:
    tuple: Updated positions and velocities after RK4 step.
    """
    # k1 values
    k1_vel = acceleration
    k1_pos = velocity

    # k2 values
    k2_vel = compute_acceleration(position + 0.5 * k1_pos * delta_t)
    k2_pos = velocity + 0.5 * k1_vel * delta_t

    # k3 values
    k3_vel = compute_acceleration(position + 0.5 * k2_pos * delta_t)
    k3_pos = velocity + 0.5 * k2_vel * delta_t

    # k4 values
    k4_vel = compute_acceleration(position + k3_pos * delta_t)
    k4_pos = velocity + k3_vel * delta_t

    # Update velocity and position
    new_velocity = velocity + (delta_t / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
    new_position = position + (delta_t / 6.0) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)

    return new_position, new_velocity

@jit(nopython=True)
def updateParameters(position, velocity, acceleration):
    """
    Update the parameters using the RK4 method.
    
    Parameters:
    position (ndarray): An array of particle positions.
    velocity (ndarray): An array of particle velocities.
    delta_t (float): Time step.
    
    Returns:
    tuple: Updated positions and velocities.
    """
    #acceleration = compute_acceleration(position)
    position, velocity = rk4_step(position, velocity, acceleration, delta_t)
    return position, velocity, acceleration

#Collision Logic for the Particles.
@jit(nopython=True)
def collide(pos1, pos2, vel1, vel2, r1, r2):
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
        vel1_new = vel1 - np.sum((vel1 - vel2) * (pos1 - pos2)) * (pos1 - pos2) / np.sum((pos1 - pos2)**2)
        vel2_new = vel2 - np.sum((vel2 - vel1) * (pos2 - pos1)) * (pos2 - pos1) / np.sum((pos2 - pos1)**2) 

    The collision formula updates the velocities of two objects after a collision
    based on the conservation of linear momentum and kinetic energy.
    """
    
    # Named constant
    COLLISION_DISTANCE = (r1 + r2)**2

    # Calculate the distance between the two objects manually
    delta_pos = pos1 - pos2
    r = np.sum(delta_pos**2)
    

    # Check if a collision occurs (distance is less than or equal to COLLISION_DISTANCE)
    if r <= COLLISION_DISTANCE:
        # Use element-wise multiplication and summation instead of np.inner
        vel1_new = vel1 - np.sum((vel1 - vel2) * (pos1 - pos2)) * (pos1 - pos2) / np.sum((pos1 - pos2)**2)
        vel2_new = vel2 - np.sum((vel2 - vel1) * (pos2 - pos1)) * (pos2 - pos1) / np.sum((pos2 - pos1)**2) 
        
        return vel1_new, vel2_new

    return vel1, vel2


@jit(nopython = True)        
def calculate_norm(array):
    return np.sqrt(np.sum(array**2))
        
######################## Drawing Each Frame ##########################
@jit(forceobj = True)
def calculate_frame(frame_count):
    # Collect positions for all balls in this frame
    ball_positions = np.zeros((number_of_balls, 2))
    ball_velocities = np.zeros((number_of_balls))
    for i in range(number_of_balls):
        for j in range(i + 1, number_of_balls):
            a, b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel, my_balls[i].r, my_balls[j].r)
            my_balls[i].vel = a
            my_balls[j].vel = b

        my_balls[i].update()
        ball_positions[i] = my_balls[i].pos[:2]  # Take only the x and y components
        ball_velocities[i] = calculate_norm(my_balls[i].vel)

    return ball_positions, ball_velocities


    




################################Generating the Ball Objects ##################################
print("================================================================")
print("Generating the Ball Objects")
print("================================================================")
# Making a list called my_balls where the balls are stored
my_balls = []

# Iterating 'number_of_balls' times and making a new ball, and appending it to the list
for i in range(number_of_balls):
    # Randomly generating position and velocity
    pos = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]
    vel = [random.random(), random.random()]
    acc = np.asarray([0,0])
    
    # Making a ball
    i_th_ball = Ball(pos, vel, acc, 5)  # Creating an instance of the Ball class
    
    # Appending the ball to the list
    my_balls.append(i_th_ball)
 
 
print("================================================================")
print("Generating The Big Ball...")
print("================================================================")
   

pos = [10,10]
vel = [random.random(), random.random()]
acc = np.asarray([0,0])
big_ball = Ball(pos, vel, acc, 100) # Creating an instance of the big ball
my_balls.append(big_ball)

print("================================================================")
print("Ball Objects generated successfully")
print("================================================================")

################################ The Actual Simulation ##################################
print("================================================================")
print("Generating Blank Array")
print("================================================================")
#Simulating and generating each individual plot
positions_array = np.zeros((number_of_frames, number_of_balls, 2))
velocity_array = np.zeros((number_of_frames, number_of_balls))
print("================================================================")
print("Blank Array Generated Successfully")
print("================================================================")
for frame_count in range(number_of_frames):
    start_time = time.time()  # Record the start time for the frame generation
    
    positions_array[frame_count], velocity_array[frame_count] = calculate_frame(frame_count)

    end_time = time.time()  # Record the end time for the frame generation
    duration = end_time - start_time
    
    print(f"Frame number {frame_count + 1} has been generated in {duration:.4f} seconds.")


print("================================================================")
print("All Frames generated Successfully")
print("================================================================")


print("================================================================")
print("Generating Video from the frames...")
print("================================================================")

# Function to update the scatter plot and histogram for each frame
def update(frame):
    ax1.clear()  # Clear the previous frame

    # Generate an array of unique colors for each particle
    unique_colors = plt.cm.autumn(np.linspace(0, 1, number_of_balls))

    # Scatter plot of positions with different colors
    ax1.scatter(
        positions_array[frame, :, 0],
        positions_array[frame, :, 1],
        color = 'blue',
        label='Position'
    )
    

    ax1.set_title(f'Frame {frame}')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-5, dimensions[0] + 5)
    ax1.set_ylim(-5, dimensions[1] + 5)

    # Histogram of velocities
    ax2.clear()
    ax2.hist(velocity_array[frame, :], bins=100, alpha=0.5, label='Velocity')
    ax2.set_xlabel('Velocity')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper right')

# Create the animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
animation = FuncAnimation(fig, update, frames=number_of_frames, interval=100, repeat=False)

# Save the animation as a GIF
animation.save('gas_simulation_animation.gif', writer='pillow', fps=10)


# Show the plot (if running in a script)
plt.show()