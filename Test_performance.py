"""
This file is to test the performance of the energy-based CA predator-prey model.
"""

import timeit 
from ca_pred_prey_energy import Habitat
import parameters
import matplotlib.pyplot as plt
import numpy as np

# Retrieve simulation parameters from parameters.py file 
dimension, num_steps = parameters.dimension, parameters.num_steps
delay_ms_movie = parameters.delay_ms_movie
empty_init, predator_init = parameters.empty_init, parameters.predator_init
mean_energy, std_energy = parameters.mean_energy, parameters.std_energy
overpopulation, prey_defense = parameters.overpopulation, parameters.prey_defense
move_loss_predator, move_gain_prey = parameters.move_loss_predator, parameters.move_gain_prey 
hunt_fail_loss, ovp_loss_prey = parameters.hunt_fail_loss, parameters.ovp_loss_prey
reproduction_loss_prey, reproduce_energy_prey, reproduce_energy_predator = parameters.reproduction_loss_prey, parameters.reproduce_energy_prey, parameters.reproduce_energy_predator

def run_simulation():
    habitat = Habitat(
        dimension, empty_init, predator_init, mean_energy, std_energy,
        overpopulation, move_loss_predator, move_gain_prey, hunt_fail_loss, ovp_loss_prey,
        reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense
    )
    habitat.population_evolution(num_steps, plot=False, fit=False, plot_energies=False, seasonal=True)

time = timeit.timeit(run_simulation, number=1)

print(f"Time taken for a single run of the simulation: {time:.2f} seconds. Parameters: lattice size {dimension}, steps {num_steps}, empty init {empty_init}, predator init {predator_init}.")


# Let's now test how it behaves when varying the time_steps
def run_simulation_with_steps(steps):
    habitat = Habitat(
        dimension, empty_init, predator_init, mean_energy, std_energy,
        overpopulation, move_loss_predator, move_gain_prey, hunt_fail_loss, ovp_loss_prey,
        reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense
    )
    habitat.population_evolution(steps, plot=False, fit=False, plot_energies=False, seasonal=False)

performances_different_steps_times = {}
performances_different_steps_std = {}
num_repeats = 5 

for steps in [100, 500, 1000, 2000, 5000]:
    times = [timeit.timeit(lambda: run_simulation_with_steps(steps), number=1) for _ in range(num_repeats)]
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Time taken for {steps} steps: {mean_time:.2f} ± {std_time:.2f} seconds. Parameters: lattice size {dimension}, empty init {empty_init}, predator init {predator_init}.")
    performances_different_steps_times[steps] = mean_time
    performances_different_steps_std[steps] = std_time

fig, ax = plt.subplots()
ax.errorbar(
    list(performances_different_steps_times.keys()),
    list(performances_different_steps_times.values()),
    yerr=list(performances_different_steps_std.values()),
    marker='o', linestyle='-', color='b', capsize=5
)
ax.set_xlabel('Number of Steps')
ax.set_ylabel('Time (seconds)')
ax.set_title('Simulation performance with varying steps')
ax.grid(True)
plt.tight_layout()
plt.savefig(f"plots/Energies/Performance_different_steps.svg")
plt.savefig(f"plots/Energies/Performance_different_steps.pdf")
#plt.show()

# Let's now test how it behaves when varying the lattice size, with a fixed number of steps

def run_simulation_with_size(size):
    habitat = Habitat(
        (size, size), empty_init, predator_init, mean_energy, std_energy,
        overpopulation, move_loss_predator, move_gain_prey, hunt_fail_loss, ovp_loss_prey,
        reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense
    )
    habitat.population_evolution(num_steps, plot=False, fit=False, plot_energies=False, seasonal=True)

performances_different_dim_times = {}
performances_different_dim_std = {}
num_repeats=5
num_steps=100

for size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    times = [timeit.timeit(lambda: run_simulation_with_size(size), number=1) for _ in range(num_repeats)]
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Time taken for lattice size {size}x{size}: {mean_time:.2f} ± {std_time:.2f} seconds. Parameters: steps {num_steps}, empty init {empty_init}, predator init {predator_init}.")
    performances_different_dim_times[size] = mean_time
    performances_different_dim_std[size] = std_time

fig, ax =plt.subplots()
ax.errorbar(
    list(performances_different_dim_times.keys()),
    list(performances_different_dim_times.values()),
    yerr=list(performances_different_dim_std.values()),
    marker='o', linestyle='-', color='b', capsize=5
)
#ax.plot(performances_different_dim.keys(), performances_different_dim.values(), marker='o', linestyle='-', color='b')
ax.set_xlabel('Lattice length N')
ax.set_ylabel('Time (seconds)')
ax.set_title('Simulation performance with varying lattice size')
ax.grid(True)
plt.tight_layout()
plt.savefig(f"plots/Energies/Performance_different_lattice_size.svg")
plt.savefig(f"plots/Energies/Performance_different_lattice_size.pdf")
plt.show()
