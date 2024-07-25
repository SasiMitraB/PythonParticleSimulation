#Libraries for Scientific Computing Stuff
import numpy as np
from numpy import random
#Libraries for Plotting and Stuff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#Library for Speeding things up
from numba import jit

#Libraries for Doing the Fancy Outputs
from tqdm import tqdm


########################################################################
#Initial conditions: Move to a seperate file and import later
########################################################################
delta_t = 0.1
number_of_particles = 1000
number_of_frames = 500
dimensions = np.asarray((100, 100))
particle_mass = 1
num_rows = 25

########################################################################
#Scientific Functions: This also goes into a seperate file and import later
########################################################################
@jit(nopython = True)        
def calculate_norm(array):
    return np.sqrt(np.sum(array**2))


@jit(nopython=True)
def gaussian_smoothing_kernel(smoothing_radius, dst):
    # Calculate the normalization constant
    sigma = smoothing_radius / 2.0
    normalization_constant = 1.0 / (sigma * np.sqrt(2 * np.pi))
    
    # Gaussian kernel function
    value = np.exp(-0.5 * (dst / sigma) ** 2)
    
    # Normalize the kernel
    normalized_value = value * normalization_constant
    return normalized_value


@jit(nopython=True)
def calculate_density_at_point(position, particle_positions, mass):
    smoothing_radius = 10
    density = 0.0
    for particle_position in particle_positions:
        dst = calculate_norm(position - particle_position)
        if dst < smoothing_radius:
            influence = gaussian_smoothing_kernel(smoothing_radius, dst)
            density += mass * influence
    return density

def calculate_density_meshgrid():
    # Create the meshgrid
    x = np.linspace(0, dimensions[0], num_rows)
    y = np.linspace(0, dimensions[1], num_rows)
    grid = np.meshgrid(x, y)

    # Initialize an array to store the density values
    density = np.zeros_like(grid[0], dtype=float)

    particle_positions = np.array([particle.position for particle in particles])
    # Calculate the density at each point in the meshgrid
    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[1]):
            position = np.array([grid[0][i, j], grid[1][i, j]])
            density[i, j] = calculate_density_at_point(position, particle_positions, particle_mass)

    return density

########################################################################
#Helper Functions: This also goes into a seperate file and import later
########################################################################
@jit(nopython = True)       
def wall_collision(position, velocity):
    global dimensions
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
    acceleration = np.asarray([0, -1])
    return acceleration

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

########################################################################
#Class Particle: This also goes to a seperate file and imported later
########################################################################

class Particle:
    def __init__(self, position, velocity, acceleration, radius):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.radius = radius

    def update(self):
        self.check_wall_collision()
        self.position, self.velocity, self.acceleration = updateParameters(position=self.position, 
                                                                           velocity=self.velocity, 
                                                                           acceleration=self.acceleration, 
                                                                           delta_t=delta_t)
    def check_wall_collision(self):
        '''Check if Collisions occurs with the walls and implements elastic collision with the walls'''
        self.velocity = wall_collision(self.position, self.velocity)






########################################################################
#Creating the Set of Particles
########################################################################

particles = []
print("================================================")
print("GENERATING THE LIST OF PARTICLES...")
print("================================================")

def generate_particles():
    positions = []
    radius = 1
    particles_per_row = int(np.sqrt(number_of_particles))
    particles_per_column = int(np.ceil(number_of_particles / particles_per_row))
    spacing = radius * 2 + 2  # Extra spacing parameter to avoid overlap
    offset = radius + 1  # Offset to keep particles within the bounding box

    # Generate positions in a grid
    for row in range(particles_per_row):
        for col in range(particles_per_column):
            if len(positions) < number_of_particles:
                x_position = col * spacing + offset
                y_position = row * spacing + offset
                
                # Ensure positions are within the bounding box
                x_position = min(x_position, dimensions[0] - offset)
                y_position = min(y_position, dimensions[1] - offset)
                
                positions.append(np.asarray([x_position, y_position]))

    return positions


# Now you can use these positions to initialize your particles
particles = []
positions = generate_particles()

for position in tqdm(positions):
    velocity = np.asarray([random.random(), random.random()]) - 0.5
    acceleration = np.asarray([0, 0])
    radius = 1
    particle = Particle(position, velocity, acceleration, radius)
    particles.append(particle)


print("================================================")
print("PARTICLE OBJECTS GENERATED SUCCESSFULLY")
print("================================================")


########################################################################
#Main Simulation Loop
########################################################################


def calculate_frame(frame_count):
    # Collect positions for all balls in this frame
    particle_positions = np.zeros((number_of_particles, 2))
    for i in range(number_of_particles):

        particles[i].update()
        particle_positions[i] = particles[i].position[:2]  # Take only the x and y components
    
    density_grid = calculate_density_meshgrid()
    return particle_positions, density_grid


#Simulating and generating each individual plot
positions_array = np.zeros((number_of_frames, number_of_particles, 2))
densities_list = []

print("================================================================")
print("Blank Array Generated Successfully")
print("================================================================")
for frame_count in tqdm(range(number_of_frames)):
    positions_array[frame_count], density_grid = calculate_frame(frame_count)
    densities_list.append(density_grid)



print("================================================================")
print("All Frames generated Successfully")
print("================================================================")


print("================================================================")
print("Generating Video from the frames...")
print("================================================================")

# Function to update the scatter plot for each frame
def update(frame):
    ax1.clear()  # Clear the previous frame
    
    # Plot the density grid using imshow
    ax1.imshow(densities_list[frame], cmap='autumn', extent=[0, dimensions[0], 0, dimensions[1]], origin='lower')
    
    # Scatter plot of positions with different colors
    ax1.scatter(
        positions_array[frame, :, 0],
        positions_array[frame, :, 1],
        color='white',
        label='Position',
        s=0.5
    )
    
    ax1.set_title(f'Frame {frame}')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-5, dimensions[0] + 5)
    ax1.set_ylim(-5, dimensions[1] + 5)
    ax1.legend(loc='upper right')



# Create the animation
fig, ax1 = plt.subplots(figsize=(6, 5))
img = ax1.imshow(densities_list[0], cmap='autumn', extent=[0, dimensions[0], 0, dimensions[1]], origin='lower')
fig.colorbar(img, ax=ax1, orientation='vertical')
animation = FuncAnimation(fig, update, frames=number_of_frames, interval=100, repeat=False)

# Save the animation as a GIF
animation.save('SPH_simulation.gif', writer='pillow', fps=30)

# Show the plot (if running in a script)
plt.show()