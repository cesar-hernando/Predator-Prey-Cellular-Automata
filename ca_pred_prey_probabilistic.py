''' This file is a self-contained Cellular Automata simulation of a Predator-Prey system. It rests entirely on probabilistic updates.'''

import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import pygame
from threading import Thread
import copy

##############################################
################ CELL ########################
##############################################

class Cell:
    '''
    This class defines a cell in the lattice and resembles either empty land or an animal.
    Species are encoded numerically: 0, 1 and 2 denote an empty cell, predator and prey, respectively.
    '''

    def __init__(self, i,j, species):
        '''
        - i, j: tuple that indicates position on lattice
        - species: integer (0,1 or 2) that indicates species
        '''
        
        if species is not None:
            self.species = species
        else:
            self.species = 0
        self.i = i
        self.j = j

    def hunt(self, prey):
        prey.species = 1 # hunting adds a predator newborn
            
    def prey_reproduce(self, empty):
        empty.species = 2

    def move(self, empty):
        empty.species = self.species
        # A predator loses energy while moving, whereas a prey gains energy
       
       # Set the last position as empty
        self.species = 0
        
##############################################
################ Habitat ########################
##############################################

class Habitat:

    def __init__(self, dimension, empty_init, predator_init, kill_prob, prey_reproduction_rate, predator_death_rate, prey_migration_rate):
        '''
        - dimension (tuple): The first element is the horizontal length and the second one the vertical length
        - empty_init: indicates proportion of initially empty cells
        - predator_init: indicates proportion of cells initially filled with predator
        - prey_init: indicates proportion of cells initially filled with prey
        - kill_prob: kill probability for each evolution step
        - prey_reproduction_rate: reproduction rate of prey in each evolution step
        - predator_death_rate: death rate of predators in each evolution step
        - prey_migration_rate: rate at which prey is migrating in each evolution step
        '''
        
        self.height, self.width = dimension
        self.lattice = np.empty(dimension, dtype=object)
        self.num_empty, self.num_predator, self.num_prey = self.initialize_lattice(empty_init, predator_init)
        self.reference_lattice = copy.deepcopy(self.lattice)
        
        # For restoring the initial lattice and populations
        self.initial_lattice = copy.deepcopy(self.lattice)
        self.initial_num_empty, self.initial_num_predator, self.initial_num_prey = self.num_empty, self.num_predator, self.num_prey
        self.kill_prob = kill_prob
        self.prey_reproduction_rate = prey_reproduction_rate
        self.predator_death_rate = predator_death_rate
        self.prey_migration_rate = prey_migration_rate

    def initialize_lattice(self, empty_init, predator_init):
        '''
        This function initializes the lattice according to the probabilities defined as input.
        '''
        num_empty = 0
        num_predator = 0
        num_prey = 0
        
        for i in range(self.width):
            for j in range(self.height):
                # Place nothing, predator or prey with respective probabilites 
                k = random.uniform(0,1)
                if k < empty_init: # Empty
                    self.lattice[i,j] = Cell(i,j, species=0)
                    num_empty += 1
                elif k < empty_init + predator_init: # Predator
                    self.lattice[i,j] = Cell(i,j, species=1)
                    num_predator += 1
                else: # Prey
                    self.lattice[i,j] = Cell(i,j, species=2)
                    num_prey += 1

        return num_empty, num_predator, num_prey


    def obtain_neighbours(self, x, y):
        '''
        This function returns the neighboring cells in the lattice, respecting periodic boundary conditions.
        The inputs x and y represent the row and column indices of the lattice matrix
        '''

        top = self.lattice[x][(y - 1) % self.height]
        top_right = self.lattice[(x + 1) % self.width][(y - 1) % self.height]
        right = self.lattice[(x + 1) % self.width][y]
        bottom_right = self.lattice[(x + 1) % self.width][(y + 1) % self.height]
        bottom = self.lattice[x][(y + 1) % self.height]
        bottom_left = self.lattice[(x - 1) % self.width][(y + 1) % self.height]
        left = self.lattice[(x - 1) % self.width][y]
        top_left = self.lattice[(x - 1) % self.width][(y - 1) % self.height]

        neighbours = [top, top_right, right, bottom_right, bottom, bottom_left, left, top_left]
        return neighbours


    def evolve(self):
        '''
        This function defines one time step of evolution. It saves the state of the system at the beginning 
        of the evolution step and performs updates based on this reference lattice. In order to avoid update conflicts,
        we keep track of already updated cells by marking indices that have been acted upon.
        
        Evolution is performed according to the probabilities of hunting, reproduction and death.
        '''
        
        self.reference_lattice = copy.deepcopy(self.lattice) 
        indices = np.array([(i,j) for i in range(self.height) for j in range(self.width)])
        visited = np.zeros(len(indices))

        np.random.shuffle(indices)
        
        for idx in indices:
            i,j = idx
            
            if visited[i*self.width+j] == 0 : 
                
                current_cell = self.reference_lattice[i,j]
                assert current_cell.i == i
                assert current_cell.j == j 
                neighbours = self.obtain_neighbours(i,j)

                ##### Predator #####
                if current_cell.species == 1: 
                    
                    preys = [cell for cell in neighbours if cell.species == 2 and visited[cell.i * self.height + cell.j] == 0]
                    
                    # Hunting
                    for target in preys:
                        if np.random.uniform(0,1) < self.kill_prob:
                            self.lattice[target.i,target.j].species = 1  
                        visited[target.i*self.width+target.j] = 1
                    
                    # Movement
                    else:
                        empty_spaces = [cell for cell in neighbours if cell.species == 0 and visited[cell.i * self.height + cell.j] == 0]
                        if len(empty_spaces) > 0:
                            move_space = random.choice(empty_spaces) 
                            self.lattice[move_space.i, move_space.j].species = current_cell.species
                            self.lattice[current_cell.i, current_cell.j].species = 0
                            visited[move_space.i*self.width+move_space.j] = 1
                    
                    # Death
                    if self.reference_lattice[i,j].species != 0 and np.random.uniform(0,1) < self.predator_death_rate:
                        self.lattice[i,j].species = 0
                
                ##### Prey #####
                elif current_cell.species == 2: 
                    
                    preys = [cell for cell in neighbours if cell.species == 2 and visited[cell.i * self.height + cell.j] == 0]
                    empty_spaces = [cell for cell in neighbours if cell.species == 0 and visited[cell.i * self.height + cell.j] == 0]
                    if len(empty_spaces) > 0:
                        
                        # Reproduction
                        if np.random.uniform(0,1) < self.prey_reproduction_rate:
                            reprod_space = random.choice(empty_spaces) 
                            self.lattice[reprod_space.i,reprod_space.j].species = 2
                            visited[reprod_space.i*self.width+reprod_space.j] = 1
                        
                        # Movement
                        else:
                            if np.random.uniform(0,1) < self.prey_migration_rate:
                                move_space = random.choice(empty_spaces) 
                                self.lattice[move_space.i,move_space.j].species = current_cell.species
                                self.lattice[current_cell.i, current_cell.j].species = 0  
                                visited[move_space.i*self.width+move_space.j] = 1              
            
                visited[current_cell.i*self.width+current_cell.j] = 1
                
        self.determine_populations()
                     
    def determine_populations(self):
        self.num_empty = 0
        self.num_predator = 0
        self.num_prey = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.lattice[i,j].species == 0:
                    self.num_empty +=1
                elif self.lattice[i,j].species == 1:
                    self.num_predator +=1 
                elif self.lattice[i,j].species == 2:
                    self.num_prey +=1
                    
    def get_populations(self):
        return self.num_empty, self.num_predator, self.num_prey

    def get_lattice(self):
        return self.lattice
    
    def restore_initial_lattice(self):
        self.lattice = copy.deepcopy(self.initial_lattice)
        self.num_empty, self.num_predator, self.num_prey = self.initial_num_empty, self.initial_num_predator, self.initial_num_prey    


    def population_evolution(self, num_steps, plot=False, dimension = None, idx = None):
        populations = np.zeros((3, num_steps))
        populations[:,0] = self.get_populations()

        # Obtain populations in each step
        for i in range(num_steps - 1):
            self.evolve()
            populations[:,i+1] = self.get_populations()

        if plot==True:
            # Plot the number of empty spaces, predators, prey and total population
            #total_population = np.sum(populations, axis=0)
            #plt.plot(range(num_steps), populations[0,:], label='Empty spaces')
            plt.figure(figsize=(8,8)) 
            
            plt.plot(range(num_steps), populations[1,:], label='Predators')
            plt.plot(range(num_steps), populations[2,:], label='Preys')
            #plt.plot(range(num_steps), total_population, label='Total')
            plt.xlabel('Simulation Steps', size = 20)
            plt.ylabel('Population Numbers', size = 20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.title('Population dynamics in probabilistic CA simulation', size = 20)
            plt.legend(fontsize=18)
            plt.savefig(f"plots/CA_prob/CA_prob_dynamics_{idx}.png")
            plt.close()
            
        return populations


    def movie(self, delay_ms, num_steps):
        '''
        This function produces an animation of the system over time.
        '''

        size = width, height = 500, 500
        is_stopped = False

        black = 0  # Empty cell
        red = 16386570 # Predator 
        blue = 658170 # Prey
        color_arr = [black, red, blue]

        pygame.init()
        screen = pygame.display.set_mode(size)
        cell_size = width // self.width  # Dynamically scale cells

        def paint(x, y, color_int):
            pygame.draw.rect(screen, color_arr[color_int], (x, y, cell_size, cell_size))

        # Paints the species at each point on the play board
        def paint_map(board):
            for x in range(self.width):
                for y in range(self.height):
                    paint(x * cell_size, y * cell_size, board[x][y].species)

        # Game loop
        n = 0
        while n < num_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_stopped = True
                    return False
            paint_map(self.get_lattice())
            pygame.display.flip()
            pygame.time.delay(delay_ms) 
            self.evolve()
            n += 1
        
        return True
    
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

def phase_diagram_init_pop(dimension, empty_init, predator_init, kill_prob, prey_reproduction_rate, predator_death_rate, prey_migration_rate, num_simulations, num_steps, migration = False):
    """
    Creates a phase diagram with the predator-prey populations from different initial values
    """

    all_populations=[]

    for _ in range(num_simulations):
        predator_init = max(0.05, predator_init + np.random.normal(0,0.005))
        habitat = Habitat(dimension, empty_init, predator_init, kill_prob, prey_reproduction_rate, predator_death_rate, prey_migration_rate)
        plot_flag = not migration # only plot if migration is not investigated
        populations=habitat.population_evolution(num_steps, plot=plot_flag, dimension = dimension, idx = _)
        all_populations.append(populations)

    plt.figure(figsize=(8,8)) 

    for i in range(0, len(all_populations)):
        line, = plt.plot(all_populations[i][1,:], all_populations[i][2, :])
        plt.scatter(all_populations[i][1, 0], all_populations[i][2, 0], color=line.get_color(), s=50)
    plt.xlabel('Predator population', size = 20)
    plt.ylabel('Prey population', size = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    stable1_pred, stable1_prey = 0, 0 
    plt.plot(stable1_pred, stable1_prey, marker='o', markersize=9, label = 'Trivial stability (extinction)')
    stable2_pred, stable2_prey = 33.3, 100
    plt.plot(stable2_pred, stable2_prey, marker='o', markersize=9, label = 'Non-Trivial orbit point (predicted)')

    if migration:
        folder = f"plots/CA_prob/migration/CA_prob_phase_{prey_migration_rate:.1f}.png"
        plt.title(f'Population dynamics, Migration rate {prey_migration_rate:.1f}', size = 22)

    else:
        folder = f"plots/CA_prob/CA_prob_phase.png"
        plt.title(f'Phase space (probabilistic CA simulation)', size = 22) 
        
    plt.legend(fontsize = 14)
    plt.savefig(folder)
    plt.close()
    
    mean_populations = np.mean(all_populations, axis=0)
    std_populations = np.std(all_populations, axis=0)
    
    fig, ax = plt.subplots(figsize=(11,11))

    for i, label in enumerate(["Predators", "Preys"]):
        i = i+1 # omit empty spaces
        plt.plot(mean_populations[i], label=label)
        plt.fill_between(range(num_steps),
                        mean_populations[i] - std_populations[i],
                        mean_populations[i] + std_populations[i],
                        alpha=0.3)

    plt.xlabel('Simulation Steps', size = 20)
    plt.ylabel('Population numbers', size = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Population dynamics in probabilistic CA simulation', size = 22)
    if migration:
        folder = f"plots/CA_prob/migration/CA_prob_uncertainties_{prey_migration_rate:.1f}.png"
    else:
        folder = f"plots/CA_prob/CA_prob_uncertainties.png"
    plt.legend(fontsize=18)
    plt.savefig(folder)
    plt.close()
    

########################################################################
########################################################################
########################################################################

##### Parameters #####
dimension = (30,30)                            # Dimension of lattice
num_steps = 400                                # number of steps of evolution
empty_init, predator_init = 0.6, 0.2           # Numbers that determine initial proportions of empty spaces, predators and preys
alpha, beta, gamma = 3e-3*900/9, 0.3, 0.1      # Simulation parameters
delta = alpha
kill_prob, prey_reproduction_rate, predator_death_rate = alpha, gamma, beta
prey_migration_rate = 1
num_simulations = 3

####################################
####### Run main simulation ########
####################################

phase_diagram_init_pop(dimension, empty_init, predator_init, kill_prob, prey_reproduction_rate, predator_death_rate, prey_migration_rate, num_simulations, num_steps)

####################################
####### Bifurcation analysis #######
####################################

prey_migration_rates = np.linspace(0,1,10)
num_simulations = 3

for prey_migration_rate in prey_migration_rates:
   phase_diagram_init_pop(dimension, empty_init, predator_init, kill_prob, prey_reproduction_rate, predator_death_rate, prey_migration_rate, num_simulations, num_steps, migration = True)


####################################
### Animation of system behavior ###
####################################

# delay_ms_movie = 25                            # Movie parameter

# dimension = (50,50)                            
# habitat = Habitat(dimension, empty_init, predator_init, kill_prob, prey_reproduction_rate, predator_death_rate, prey_migration_rate)
# habitat.population_evolution(num_steps, plot=False)
# habitat.restore_initial_lattice()
# finished = habitat.movie(delay_ms_movie, num_steps)
