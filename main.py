'''
This is the main file of the simulation of Predator-Prey dynamics using Cellular Automata
'''

from ca_pred_prey_energy import Habitat, phase_diagram_init_pop, phase_diagram_ratio_pred_prey
from ca_pred_prey_energy import Habitat, phase_diagram_init_pop
import parameters

# Retrieve simulation parameters from parameters.py file (explained their interpreation there)
dimension, num_steps = parameters.dimension, parameters.num_steps
delay_ms_movie = parameters.delay_ms_movie
empty_init, predator_init = parameters.empty_init, parameters.predator_init
mean_energy, std_energy = parameters.mean_energy, parameters.std_energy
overpopulation, prey_defense = parameters.overpopulation, parameters.prey_defense
move_loss_predator, move_gain_prey = parameters.move_loss_predator, parameters.move_gain_prey
hunt_fail_loss, ovp_loss_prey = parameters.hunt_fail_loss, parameters.ovp_loss_prey
reproduction_loss_prey, reproduce_energy_prey, reproduce_energy_predator = parameters.reproduction_loss_prey, parameters.reproduce_energy_prey, parameters.reproduce_energy_predator
seasonal = parameters.seasonal

# Create an object of the Habitat class
habitat = Habitat(dimension, empty_init, predator_init, mean_energy, std_energy, overpopulation, move_loss_predator, move_gain_prey, 
                  hunt_fail_loss, ovp_loss_prey, reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense)

# Plot evolution of populations
# Set seasonal to True to consider seasonal effects and False to ignore them
habitat.population_evolution(num_steps, plot=True, fit=False, plot_energies=True, seasonal=seasonal)

# Generate the movie of the evolution of the habitat, but first restore the initial lattice
habitat.restore_initial_lattice()
finished = habitat.movie(delay_ms_movie, num_steps)
print("Movie ended" if finished else "Movie stopped manually")

# Plot phase diagram of predator-prey dynamics
phase_diagram_init_pop(dimension, empty_init, mean_energy, std_energy, num_steps, overpopulation, move_loss_predator, move_gain_prey, hunt_fail_loss, ovp_loss_prey, 
                       reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense, seasonal=seasonal)


# The following parameters are for testing the ratio
prey_defense = 10 # Remove prey defense
seasonal = False  # Remove seasonal effects

for ratio in [0.01, 0.1, 0.5, 1, 2, 10, 100]:
    phase_diagram_ratio_pred_prey(dimension, ratio, mean_energy, std_energy, num_steps, overpopulation, move_loss_predator, move_gain_prey, 
                                  hunt_fail_loss, ovp_loss_prey, reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense, seasonal=seasonal)

