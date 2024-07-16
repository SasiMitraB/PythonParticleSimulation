import numpy as np
from numba import jit, njit
from matplotlib import pyplot as plt

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
    def __init__(self, position, velocity, acceleration, radius, dimensions, delta_t):
        # Taking Position, velocity, acceleration as inputs and making them object variables
        self.pos = np.asarray(position)
        self.vel = np.asarray(velocity) * 10
        self.acc = acceleration
        self.r = radius
        self.dimensions = dimensions
        self.delta_t = delta_t

    def check_wall_collision(self):
        '''Check if Collisions occurs with the walls and implements elastic collision with the walls'''
        self.vel = wall_collision(self.pos, self.vel, self.dimensions)


    def update(self):
        """
        Update the position, velocity, and handle collisions.
        """
        self.check_wall_collision()
        self.pos, self.vel, self.acc =  updateParameters(self.pos, self.vel, self.acc, self.delta_t)



       
@jit(nopython = True)       
def wall_collision(position, velocity, dimensions):
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
def updateParameters(position, velocity, acceleration, delta_t):
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