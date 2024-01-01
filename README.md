# Ideal Gas Particle based Simulation

## Table of Contents

- [Ideal Gas Particle based Simulation](#ideal-gas-particle-based-simulation)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Folder Structure](#folder-structure)
  - [Key Components](#key-components)
    - [Ball Class](#ball-class)
    - [Simulation Logic](#simulation-logic)
    - [Visualization](#visualization)
  - [Project Overview](#project-overview)
  - [Video Generation](#video-generation)
  - [The Particle Class](#the-particle-class)
    - [Initialization Function](#initialization-function)
    - [Collision with Walls](#collision-with-walls)
    - [Update Function](#update-function)
    - [Generating the Balls](#generating-the-balls)
  - [Particle-Particle Collision Logic](#particle-particle-collision-logic)
  - [Code Optimization](#code-optimization)
  - [Features yet to be Implemented](#features-yet-to-be-implemented)
  - [Author Information and Contact](#author-information-and-contact)

## Description

The "Ideal Gas Particle Based Simulation" is a Python program that simulates the behavior of ideal gas particles in a 3D space. It utilizes a particle-based approach, where each particle represents a gas molecule, following principles of ideal gas behavior.

## Folder Structure

The repository has the following organized folder structure:

- **animated_plots**: This folder contains a Python library for generating animations with PNG images.
- **animation**: This folder stores all the saved plots.
- **documentation.md**: This markdown file provides a detailed description of the simulation's functionality.
- **generate_video.py**: This script generates a video using files from the animation folder.
- **main.py**: The main simulation script.
- **documentation.pdf**: A PDF version of the documentation.
- **output_video.mp4**: The video generated from the simulation.
- **pygame_rendering.py**: A script for pygame-based 2D simulations.

## Key Components

### Ball Class

The simulation revolves around the Ball class, representing a gas particle. It maintains the position, velocity, and acceleration of the particle. Key methods include:

- `__init__(self, position, velocity, acceleration)`: Initializes a particle with given initial conditions.
- `check_wall_collision(self, position, velocity)`: Checks and handles collisions with the walls of the simulation space.
- `update(self)`: Updates the position, velocity, and handles collisions for each time step.

### Simulation Logic

The simulation involves collision logic, particle update, and drawing frames for visualization. It assumes an elastic collision model and updates particle positions, velocities, and accelerations based on a defined time step.

### Visualization

The simulation produces frames for visualization, including text-based animations and 3D scatter plots with velocity histograms.

## Project Overview

The simulation begins by creating a set of particles stored in a list. The simulation loop includes steps for checking collisions, updating particle positions, and drawing frames. The simulation generates a specified number of frames, and these frames are stitched together into an MP4 video.

Adjustable simulation variables include `number_of_frames`, `dimensions`, `delta_t`, and `number_of_balls`.

## Video Generation

The video generation engine is a crucial part of the project. A function named `generate_animation` uses OpenCV to create an MP4 video from individual frames.

## The Particle Class

The Particle Class simplifies coding using Python's object-oriented capabilities. It includes methods for initialization, collision with walls, and updating particle positions and velocities.

### Initialization Function

The `__init__` method initializes a particle with given initial conditions.

### Collision with Walls

The `check_wall_collision` method handles collisions with the walls, implementing elastic collision models.

### Update Function

The `update` method updates the position and velocity of the particle, handling collisions and using Euler's method for numerical integration.

### Generating the Balls

A loop generates a set of balls for interaction, considering specified parameters.

## Particle-Particle Collision Logic

The collision logic involves detecting and handling collisions between particles, optimizing checks for efficiency.

## Code Optimization

Mathematical calculations are optimized with Just-In-Time Compilation using Numba. Future work involves exploring external libraries for further optimization.

## Features yet to be Implemented

Several potential enhancements are outlined, including particle interactions, Runge-Kutta integration, optimization with external libraries, alternative plotting methods, and parallelized collision detection.

## Author Information and Contact

- **Name:** Behara Sasi Mitra
- **Email:** beharasasimitra211141@students.iisertirupati.ac.in
- **GitHub:** [SasiMitraB](https://github.com/SasiMitraB)
