'''Ideal Gas Particle Based Simulation'''

#Importing the Nessecary Libraries
from Ball import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from numpy import random
import pandas as pd
from numba import jit, njit
import os
import time



number_of_balls = 5000
dimensions = np.asarray([1000,1000])
number_of_frames = 100
delta_t = 1 # Update with the actual time step

        
######################## Drawing Each Frame ##########################

#Attempting to implement Grid Based Model



@jit(forceobj = True)
def calculate_frame(frame_count):
    # Collect positions for all balls in this frame
    ball_positions = np.zeros((number_of_balls, 2))
    ball_velocities = np.zeros((number_of_balls))
    for i in range(number_of_balls):
        for j in range(i + 1, number_of_balls):
            if my_balls[i].gridx == my_balls[j].gridx and my_balls[j].gridy == my_balls[i].gridy:
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
    pos = np.asarray([random.randint(0, dimensions[0]), random.randint(0, dimensions[1])])
    vel = np.asarray([random.random(), random.random()])
    acc = np.asarray([0,0])
    
    # Making a ball
    i_th_ball = Ball(pos, vel, acc, 5, 
                     dimensions=dimensions,
                     delta_t=delta_t,
                     gridx_size=10,
                     gridy_size=10)  # Creating an instance of the Ball class
    
    # Appending the ball to the list
    my_balls.append(i_th_ball)
 

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