'''
In this file, we define the parameters used for the CA Predator-prey simulation
'''

# Dimension lattice
dimension = (50,50)

# Number of steps of evolution
num_steps = 200 #We consider 360 for 90 days in each season 

# Mean value and standard deviation of energy in habitat creation
mean_energy = 5
std_energy = 3

# Numbers that determine initial proportions of empty spaces, predators and preys
empty_init=6
predator_init=8

# Movie parameters
delay_ms_movie = 250

# Evolution dynamics
overpopulation= 3                   # Minimum number of preys sorrounding a certain prey that causes it to lose energy
ovp_loss_prey = 1.7                 # Energy lost by prey when overpopulated
move_loss_predator= 2.5             # Energy lost by predator when moving
move_gain_prey= 0.8                 # Energy gained by prey when moving
hunt_fail_loss = 3.2                # Energy lost by predator when it fails to hunt a prey
prey_defense = 10                    # Minimum Number of preys sorrounding a certain predator that avoids the predator from hunting any of the preys
reproduction_loss_prey = 1.2        # Energy lost by prey when reproducing
reproduce_energy_prey= 3.9          # Energy required by prey to reproduce
reproduce_energy_predator = 4.7     # Energy required by predator to reproduce
seasonal = False
