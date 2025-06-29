import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
import copy
from scipy.optimize import curve_fit


class Cell:

    """
    This class defines cells and their update rules in the predator-prey cellular automata.

    ---------------------------------------------------------------------------------------

    Attributes:

    species: 
        Determines the species of the cell.

    energy:
        Determines the energy of the cell.
    
    ---------------------------------------------------------------------------------------
    Methods:

    __init__: 
        Defines parameters used in each cell: species and energy.

    hunt:
        Defines the rules and rewards of hunting for predators.
        After hunting, if the predator has enough energy, it reproduces and adds a new predator to the lattice.
        Otherwise, it simply retrieves its energy  

    prey_reproduce:
        Characterises the prey reproduction.
        There's a loss of energy when reproducing.

    move:
        Updates predator's and prey's position in the lattice. Preys gain energy while moving (foraging), 
        whereas predators lose energy while moving (aging and starving).

    
    """
    # species = 0, 1 and 2 denotes empty, predator and prey, respectively
    def __init__(self, species=None, energy=None):

        if species is not None:
            self.species = species
        else:
            self.species = 0
        
        if energy is not None:
            self.energy = energy
        else:
            self.energy = 5
    

    def hunt(self, prey, mean_energy, reproduce_energy_predator):
        '''
        Inputs
        -------   
        prey: 
            Cell with the hunted prey. Updated with an empty cell or a newborn predator.

        mean_energy:
            The mean energy of the initial prey and predator cells in the habitat, as well as the energy of newborn predators.

        reproduce_energy_predator:
            The minimum energy required for a predator to reproduce after hunting.
        
        -------------------------------------------------------------------------------------------------------------------------
        Output
        -------
        reproduce:
            Boolean value that indicates whether the predator reproduces after hunting or not.
            If True, the prey is replaced by a newborn predator with the mean energy.
            If False, the prey is replaced by an empty cell and the predator gains energy from the prey.
        '''
        if self.energy > reproduce_energy_predator: # If the predator has enough energy, it reproduces, and does not lose energy nor gain it
            prey.species = 1 # Hunting adds a predator newborn
            prey.energy = mean_energy # Newborn predator has the mean energy
            reproduce=True
        else: # If the predator does not have enough energy to reproduce, it simply hunts the prey and increase its energy to the mean energy
            prey.species=0
            self.energy = mean_energy # The predator gains energy from the eating the prey
            reproduce=False

        return reproduce
            
            

    def prey_reproduce(self, empty, mean_energy, reproduction_loss_prey):
        '''
        Inputs
        -------
        mean_energy:
            The mean energy of the prey and predator cells in the habitat, as well as the energy of newborn preys.

        reproduction_loss_prey:
            The energy loss of a prey when reproducing.
        '''
        empty.species = 2 # A newborn prey is added where the empty cell was
        empty.energy = mean_energy # Newborn prey has the mean energy
        self.energy -= reproduction_loss_prey # The prey loses energy when reproducing


    def move(self, empty, move_loss_predator, move_gain_prey):
        '''
        Inputs
        -------
        empty: 
            Cell where the predator or prey moves to. Updated with the current cell's species and energy.
        
        move_loss_predator:
            Energy loss of a predator when moving.
        
        move_gain_prey:
            Energy gain of a prey when moving.
        '''
        empty.species = self.species
        # A predator loses energy while moving, whereas a prey gains energy
        if self.species == 1:
            empty.energy = self.energy - move_loss_predator
        else:
            empty.energy = self.energy + move_gain_prey
       
       # Set the last position as empty
        self.species = 0
        self.energy = 0

        
##########################################################################################################################

class Habitat:

    """
    Here the lattice where predators and preys lie is defined, updated, evolved and plotted (both phase diagram and a movie of the evolution).

    ---------------------------------------------------------------------------------------

    Attributes:
        Explained in parameters.py file.

    ---------------------------------------------------------------------------------------

    Methods:

    __init__:
        Parameters are initialised and the lattice is created with the given dimensions.
    
    initialize_lattice:
        Creates the lattice with the given initial proportions of empty spaces, predators and preys (proportions are approximative, as the populations is randomly generated).

    obtain_neightbours:
        Returns neighbouring cells and their indices in the lattice.

    death_check:
        Kills a predator or a prey it its energy falls below zero.
    
    seasonality:
        Updates the parameters of the habitat my modeling seasonality as an oscillation.
    
    evolve:
        Updates the lattice: predators hunt preys if they can; move to empty cells if they cannot hunt; or die if they lose all their energy. Preys reproduce if they have enough energy; or they move to an empty cell; or they die if they lose all their energy.
    
    get_populations:
        Returns the number of empty spaces, predators and preys in the lattice.

    get_lattice:
        Returns the lattice.

    get_energies:
        Returns the energies of predators and preys.
    
    population_evolution:
        Evolves the habitat for a given number of steps following 'evolve()' and plots the populations of empty spaces, predators and preys.
    
    get_cell_color:
        Returns the color of a cell based on its species and energy. The color is represented as an (R, G, B) tuple.

    movie:
        Movie of the lattice evolution. Black squares represent empty cells, red squares represent predators and blue squares represent preys (the intensity of the color is proportional to the energy of the cell, following 'get_cell_color').

    restore_initial_lattice:
        Restores the initial lattice and populations of empty spaces, predators and preys.
   
    """

    seed = 42

    def __init__(self, dimension, empty_init, predator_init, mean_energy, std_energy, overpopulation, move_loss_predator, move_gain_prey, 
                 hunt_fail_loss, ovp_loss_prey, reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense):
        '''
        Initializes the habitat with the given parameters (explained in parameters.py file).
        '''

        # Dimensions of the lattice
        self.width, self.height = dimension

        # Static initial parameters
        self.mean_energy, self.std_energy = mean_energy, std_energy
        self.prey_defense, self.init_overpopulation = prey_defense, overpopulation
        self.init_move_loss_predator, self.init_move_gain_prey = move_loss_predator, move_gain_prey
        self.init_hunt_fail_loss, self.init_ovp_loss_prey = hunt_fail_loss, ovp_loss_prey
        self.init_reproduce_energy_prey, self.reproduce_energy_predator = reproduce_energy_prey, reproduce_energy_predator
        self.init_reproduction_loss_prey = reproduction_loss_prey
        self.init_prop_empty, self.init_prop_predator = empty_init, predator_init 

        # Dynamic parameters, varying with seasonality
        self.overpopulation = overpopulation
        self.move_loss_predator, self.move_gain_prey, self.hunt_fail_loss, self.ovp_loss_prey =  move_loss_predator, move_gain_prey, hunt_fail_loss, ovp_loss_prey
        self.reproduce_energy_prey = reproduce_energy_prey
        self.reproduction_loss_prey = reproduction_loss_prey
        
        
        # Initialize the lattice
        self.lattice = np.empty(dimension, dtype=object)
        random.seed(self.seed)
        self.num_empty, self.num_predator, self.num_prey = self.initialize_lattice(empty_init, predator_init)

        # For restoring the initial lattice and populations
        self.initial_lattice = copy.deepcopy(self.lattice)
        self.initial_num_empty, self.initial_num_predator, self.initial_num_prey = self.num_empty, self.num_predator, self.num_prey
        
        

    def initialize_lattice(self, empty_init, predator_init):
        '''
        Initializes the lattice with the given initial proportions of empty spaces, predators and preys 
        given by empty_init and predator_init.
        '''
        num_empty = 0
        num_predator = 0
        num_prey = 0
        
        for i in range(self.width):
            for j in range(self.height):
            # Place nothing, predator or prey with probabilites 0.4, 0.2 and 0.4, respectively
                k = random.uniform(0,10)
                if k <= empty_init: # Empty
                    self.lattice[i,j] = Cell(species=0, energy=0)
                    num_empty += 1
                elif k <= predator_init: # Predator
                    self.lattice[i,j] = Cell(species=1, energy=max(0, random.normalvariate(self.mean_energy, self.std_energy)))
                    num_predator += 1
                else: # Prey
                    self.lattice[i,j] = Cell(species=2, energy=max(0, random.normalvariate(self.mean_energy, self.std_energy)))
                    num_prey += 1

        return num_empty, num_predator, num_prey


    def obtain_neighbours(self, x, y):
        '''
        Returns the neighbouring cells (and their coordinates) of a given cell in the lattice, 
        considering periodic boundary conditions.
        
        x and y represent the row and column indices of the lattice matrix
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

        neighbours_coordinates = [(x,(y - 1) % self.height), ((x + 1) % self.width, (y - 1) % self.height), 
                   ((x + 1) % self.width, y), ((x + 1) % self.width, (y + 1) % self.height), 
                   (x, (y + 1) % self.height), ((x - 1) % self.width, (y + 1) % self.height), 
                   ((x - 1) % self.width, y), ((x - 1) % self.width, (y - 1) % self.height)]
        
        return neighbours, neighbours_coordinates

    def death_check(self, cell):
        '''
        Checks if a cell's energy is less than or equal to zero. If so, it kills the cell and updates the populations.
        '''

        if cell.energy <= 0:
            
            if cell.species == 1:
                self.num_predator -= 1
                self.num_empty += 1
            elif cell.species == 2:
                self.num_prey -= 1
                self.num_empty += 1
            
            cell.energy = 0
            cell.species = 0

    def seasonality(self, time, num_time_steps):
        """
        This function varies the parameters of the habitat with seasonality.
        We do three cycles of seasonality.
        """

        self.overpopulation = round(self.init_overpopulation+(np.cos(time*3*np.pi/num_time_steps+np.pi))**2)
        self.move_loss_predator =  self.init_move_loss_predator+(1/2)*(np.cos(time*3*np.pi/num_time_steps)) # We vary the move loss of predators with seasonality by +-0.5
        self.move_gain_prey = self.init_move_gain_prey-(1/2)*(np.cos(time*3*np.pi/num_time_steps)) # We vary the move gain of preys with seasonality by +-0.5
        self.hunt_fail_loss = self.init_hunt_fail_loss+(1/4)*(np.cos(time*3*np.pi/num_time_steps)) # We vary the hunt fail loss of predators with seasonality by +-0.25
        self.ovp_loss_prey = self.init_ovp_loss_prey+(3/2)*(np.cos(time*3*np.pi/num_time_steps)) # We vary the overpopulation loss of preys with seasonality by +-0.75
        self.reproduction_loss_prey = self.init_reproduction_loss_prey+(1/2)*(np.cos(time*3*np.pi/num_time_steps)) # We vary the reproduction energy of preys with seasonality by +-0.5

    def evolve(self):
        
        # Randomly shuffle the indices of the lattice to ensure that the order of evolution is random
        indices = np.array([(i,j) for i in range(self.width) for j in range(self.height)])
        np.random.shuffle(indices)
        indices = indices.tolist()

        for idx in indices:
                i,j = idx
                current_cell = self.lattice[i,j]
                neighbours, neighbours_coordinates = self.obtain_neighbours(i,j)

                '''---------------Predator moves ----------------'''
                if current_cell.species == 1:
                    preys = [cell for cell in neighbours if cell.species == 2] # Search for preys in the neighbours
                    if len(preys) > 0 and len(preys) < self.prey_defense: # If there are enough preys to hunt and not too many to defend themselves
                        target = random.choice(preys)
                        index_target = preys.index(target)
                        indices.remove(neighbours_coordinates[index_target]) if neighbours_coordinates[index_target] in indices else None # Remove the index of the target prey from the indices list
                        if current_cell.energy > target.energy: # If the predator has more energy than the prey, it hunts it
                            predator_reproduce = current_cell.hunt(target, self.mean_energy, self.reproduce_energy_predator)

                            if predator_reproduce: # If the predator reproduces, it adds a new predator
                                self.num_predator += 1 # When a predator hunts, it reproduces automatically
                                self.num_prey -= 1

                            else: # If the predator does not reproduce, it restores its energy to the mean energy
                                self.num_prey -= 1
                                self.num_empty += 1
                        else:
                            current_cell.energy -= self.hunt_fail_loss # If it doesn't hunt, it loses energy due to aging
                    else:
                        empty_spaces = [cell for cell in neighbours if cell.species == 0]
                        if len(empty_spaces) > 0:
                            chosen_empty_space = random.choice(empty_spaces)
                            index_chosen_empty_space = empty_spaces.index(chosen_empty_space)
                            indices.remove(neighbours_coordinates[index_chosen_empty_space]) if neighbours_coordinates[index_chosen_empty_space] in indices else None
                            current_cell.move(chosen_empty_space, self.move_loss_predator, self.move_gain_prey)
                            self.death_check(chosen_empty_space)
                    
                '''---------------Prey moves ----------------'''
                if current_cell.species == 2:
                    preys = [cell for cell in neighbours if cell.species == 2]

                    if len(preys) > self.overpopulation: # Overpopulation causes the prey to lose energy
                        current_cell.energy -= self.ovp_loss_prey

                    # Search for empty spaces in the neighbours to move or reproduce
                    empty_spaces = [cell for cell in neighbours if cell.species == 0]
                    if len(empty_spaces) > 0:
                        if current_cell.energy >= self.reproduce_energy_prey: # If the prey has enough energy, it reproduces
                            chosen_empty_space = random.choice(empty_spaces)
                            # Remove the index of the chosen empty space from the indices list
                            index_chosen_empty_space = empty_spaces.index(chosen_empty_space)
                            indices.remove(neighbours_coordinates[index_chosen_empty_space]) if neighbours_coordinates[index_chosen_empty_space] in indices else None 
                            # The prey reproduces in the chosen empty space
                            current_cell.prey_reproduce(chosen_empty_space, self.mean_energy, self.reproduction_loss_prey)
                            self.num_prey += 1
                            self.num_empty -= 1

                        elif current_cell.energy > 0: # Overpopulation did not kill the prey, so it can move  (but not reproduce)  
                            chosen_empty_space = random.choice(empty_spaces)
                            # Remove the index of the chosen empty space from the indices list
                            index_chosen_empty_space = empty_spaces.index(chosen_empty_space)
                            indices.remove(neighbours_coordinates[index_chosen_empty_space]) if neighbours_coordinates[index_chosen_empty_space] in indices else None
                            current_cell.move(chosen_empty_space, self.move_loss_predator, self.move_gain_prey)
                            # Check if the prey has died after moving
                            self.death_check(chosen_empty_space)
                        
                # Check if the current cell has died after its action (hunting, moving or reproducing)
                self.death_check(current_cell)
                
                    
                        
    def get_populations(self):
        return self.num_empty, self.num_predator, self.num_prey


    def get_lattice(self):
        return self.lattice
    

    def get_energies(self):
        predator_energies = []
        prey_energies = []
        for i in range(self.width):
            for j in range(self.height):
                cell = self.lattice[i, j]
                if cell.species == 1:
                    predator_energies.append(cell.energy)
                elif cell.species == 2:
                    prey_energies.append(cell.energy)
        return predator_energies, prey_energies
    

    def population_evolution(self, num_steps, plot=True, fit=True, plot_energies=False, seasonal=True):
        populations = np.zeros((3, num_steps))
        populations[:,0] = self.get_populations()
        predator_energies_list = []
        prey_energies_list = []

        # Obtain populations in each step
        for i in range(num_steps - 1):
            if seasonal:
                self.seasonality(i, num_steps)
            self.evolve()
            populations[:,i+1] = self.get_populations()
            if plot_energies:
                pred_e, prey_e = self.get_energies()
                predator_energies_list.append(pred_e)
                prey_energies_list.append(prey_e)

        #We will fit the prey and predator populations with a sinus function
        def sin_func(x, amplitude, frequency, phase, offset):
            return amplitude * np.sin(frequency * x + phase) + offset

        def osc_fitting(populations, num_steps):
            initial_guess_prey = [20, 0.5, 1, populations[2, 0]]
            initial_guess_predator = [20, 0.5, 1, populations[1, 0]]

            params_prey, params_covariance_prey = curve_fit(sin_func, range(num_steps), populations[2, :], p0=initial_guess_prey)
            params_predator, params_covariance_predator = curve_fit(sin_func, range(num_steps), populations[1, :], p0=initial_guess_predator)

            return params_prey, params_covariance_prey, params_predator, params_covariance_predator

        if fit:
            params_prey, params_covariance_prey, params_predator, params_covariance_predator = osc_fitting(populations, num_steps)

        if plot==True:
            fig, ax = plt.subplots(figsize=(20,9))

            # Plot the number of empty spaces, predator, preys and total population
            total_population = np.sum(populations, axis=0)
            ax.plot(range(num_steps), populations[0,:], label='Empty spaces')
            line1, =ax.plot(range(num_steps), populations[1,:], label='Predators')
            if fit:
                ax.plot(range(num_steps), sin_func(range(num_steps), params_predator[0], params_predator[1], params_predator[2], params_predator[3]), label='Fitted Predators', ls="--", color=line1.get_color())
            line2, =ax.plot(range(num_steps), populations[2,:], label='Preys')
            if fit:
                ax.plot(range(num_steps), sin_func(range(num_steps), params_prey[0], params_prey[1], params_prey[2], params_prey[3]), label='Fitted Preys', ls="--", color=line2.get_color())
            ax.plot(range(num_steps), total_population, label='Total')
            ax.set_xlabel('Days')
            ax.set_ylabel('Population')
            ax.set_title('Analysis of populations in Predator-Prey CA')
            plt.legend()
            plt.savefig(f"plots/Energies/population_initial_empty_{self.init_prop_empty*10}_initial_predator{(self.init_prop_predator-self.init_prop_empty)*10}.svg")
            plt.savefig(f"plots/Energies/population_initial_empty_{self.init_prop_empty*10}_initial_predator{(self.init_prop_predator-self.init_prop_empty)*10}.pdf")
            plt.show()
        
        if plot_energies:
            # Plot mean energies over time
            mean_predator_energy = [np.mean(e) if e else 0 for e in predator_energies_list]
            mean_prey_energy = [np.mean(e) if e else 0 for e in prey_energies_list]
            plt.figure(figsize=(9,5))
            plt.plot(mean_predator_energy, label='Mean Predator Energy')
            plt.plot(mean_prey_energy, label='Mean Prey Energy')
            plt.xlabel('Step')
            plt.ylabel('Mean Energy')
            plt.legend()
            plt.title('Mean Energies of Predators and Preys')
            plt.show()
        
        return populations


    def get_cell_color(self, cell, max_energy=8):
        """
        Returns an (R, G, B) tuple for the cell based on its species and energy.
        max_energy: the energy value that maps to full intensity.
        """
        if cell.species == 0:
            return (0, 0, 0)  # Black for empty
        elif cell.species == 1:
            # Predator: Red intensity based on energy
            intensity = int(255 * min(cell.energy, max_energy) / max_energy)
            return (intensity, 0, 0)
        elif cell.species == 2:
            # Prey: Blue intensity based on energy
            intensity = int(255 * min(cell.energy, max_energy) / max_energy)
            return (0, 0, intensity)
        else:
            return (0, 0, 0)

    def movie(self, delay_ms, num_steps):
        random.seed(self.seed)

        grid_size = 800
        bar_width = 90  # Increase to allow labels and both bars to fit
        size = width, height = grid_size + bar_width, 850

        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Predator-Prey Simulation")
        cell_size = grid_size // self.width
        font = pygame.font.SysFont(None, 22)

        def paint(x, y, cell):
            color = self.get_cell_color(cell)
            pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))

        def paint_map(board):
            for x in range(self.width):
                for y in range(self.height):
                    paint(x * cell_size, y * cell_size, board[x][y])

        def paint_energy_bar():
            bar_x = grid_size + 5  # shift left
            bar_top = 50
            bar_height = height - 180
            segment_height = bar_height // 100

            for i in range(100):
                energy = (i / 100) * 8
                predator_color = self.get_cell_color(Cell(species=1, energy=energy))
                prey_color = self.get_cell_color(Cell(species=2, energy=energy))
                y = bar_top + (99 - i) * segment_height

                pygame.draw.rect(screen, predator_color, (bar_x, y, bar_width // 2 - 6, segment_height))
                pygame.draw.rect(screen, prey_color, (bar_x + bar_width // 2 + 2, y, bar_width // 2 - 6, segment_height))

            # Title label
            energy_label = font.render("Energy", True, (255, 255, 255))
            screen.blit(energy_label, (bar_x, bar_top - 30))

            # Value labels
            screen.blit(font.render("8", True, (255, 255, 255)), (bar_x + bar_width // 2 - 8, bar_top - 8))
            screen.blit(font.render("0", True, (255, 255, 255)), (bar_x + bar_width // 2 - 8, bar_top + bar_height + 5))

            # Species labels
            screen.blit(font.render("Pred", True, (255, 100, 100)), (bar_x, bar_top + bar_height + 25))
            screen.blit(font.render("Prey", True, (100, 100, 255)), (bar_x + bar_width // 2 + 2, bar_top + bar_height + 25))



        # Game loop
        n = 0
        running = True
        while n < num_steps and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            paint_map(self.get_lattice())
            paint_energy_bar()
            pygame.display.flip()
            pygame.time.delay(delay_ms)
            self.evolve()
            n += 1

        pygame.quit()
        return running


        

    def restore_initial_lattice(self):
        self.lattice = copy.deepcopy(self.initial_lattice)

        #self.lattice = self.initial_lattice
        self.num_empty, self.num_predator, self.num_prey = self.initial_num_empty, self.initial_num_predator, self.initial_num_prey    

    
#####################################################################################################################################


def phase_diagram_init_pop(dimension, empty_init, mean_energy, std_energy, num_steps, overpopulation, move_loss_predator, move_gain_prey, 
                           hunt_fail_loss, ovp_loss_prey, reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense, seasonal):
    """
    Creates a phase diagram with the predator-prey populations from different initial values
    """
    
    predator_init_range=list(range(empty_init+1, 10))
    n_repeat=2
    custom_colors = ['green', 'purple', 'orange', 'brown']

    for predator_init in predator_init_range:
        all_populations=[]
        for n in range(0, n_repeat):
            habitat = Habitat(dimension, empty_init, predator_init, mean_energy, std_energy, overpopulation, move_loss_predator, move_gain_prey, 
                              hunt_fail_loss, ovp_loss_prey, reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense)
            populations=habitat.population_evolution(num_steps, plot=False, fit=False, seasonal=seasonal)
            all_populations.append(populations)

        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_prop_cycle(plt.cycler('color', custom_colors))
        for i in range(0, len(all_populations)):
            ax.plot(all_populations[i][1,:], all_populations[i][2, :], '--', alpha=0.7) #Plot predators vs preys
            if i == 0:
                ax.scatter(all_populations[i][1, 0], all_populations[i][2, 0], marker='s', color='blue', s=50, label='Initial population')
                ax.scatter(all_populations[i][1, -1], all_populations[i][2, -1], marker='s', color='red', s=50, label='Final population')
            else:
                ax.scatter(all_populations[i][1, 0], all_populations[i][2, 0], marker='s', color='blue', s=50)
                ax.scatter(all_populations[i][1, -1], all_populations[i][2, -1], marker='s', color='red', s=50)
            ax.axline((0, dimension[0]*dimension[1]), (dimension[0]*dimension[1], 0), linewidth=1.5, color="black")
            ax.axhline(xmin=0, xmax=dimension[0]*dimension[1], color="black", ls='-.') #Prey extinction
            ax.axvline(ymin=0, ymax=dimension[0]*dimension[1], color="black", ls='-.') #Predator extinction
        ax.set_xlabel('Predator population')
        ax.set_ylabel('Prey population')
        ax.set_title(f'Phase diagram with an approximate initial {(predator_init-empty_init)*10}% predator population')
        ax.legend()

        ax.grid(True,linewidth=0.1)
        plt.savefig(f"plots/Energies/Phase_Diagram_empyt_{empty_init*10}_predator{(predator_init-empty_init)*10}.svg")
        plt.savefig(f"plots/Energies/Phase_Diagram_empyt_{empty_init*10}_predator{(predator_init-empty_init)*10}.pdf")
        plt.tight_layout()
        plt.show()  


def phase_diagram_ratio_pred_prey(dimension, ratio, mean_energy, std_energy, num_steps, overpopulation, move_loss_predator, move_gain_prey, 
                                  hunt_fail_loss, ovp_loss_prey, reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense, seasonal):
    """
    Creates a phase diagram with the predator-prey populations for a fixed ratio of predators to preys. The initial proportion of empty spaces is varied.
    """

    #Ratio is prey population divided by predator population
    max_pred_pop = 10/(1+ratio) #Maximum predator population for the given ratio
    list_pred_init = [i*max_pred_pop/5 for i in range(1, 5)]                    #Predator populations are 1/5, 2/5, 3/5 and 4/5 of the maximum predator population
    list_prey_init= [i*max_pred_pop*ratio/5 for i in range(1, 5)]               #Prey populations are 1/5, 2/5, 3/5 and 4/5 of the maximum prey population
    list_empty_init = 10 - np.array(list_pred_init) - np.array(list_prey_init)  #Empty spaces are the rest of the population
    all_populations = []
    custom_colors = ['green', 'orange', 'purple', 'brown', 'lightblue']

    for pred_init, prey_init, empty_init in zip(list_pred_init, list_prey_init, list_empty_init):
        habitat = Habitat(dimension, empty_init, empty_init+pred_init,mean_energy, std_energy, overpopulation, move_loss_predator, move_gain_prey,
               hunt_fail_loss, ovp_loss_prey, reproduce_energy_prey, reproduce_energy_predator, reproduction_loss_prey, prey_defense)
        populations = habitat.population_evolution(num_steps=200, plot=False, fit=False, seasonal=seasonal)
        all_populations.append(populations)

    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_prop_cycle(plt.cycler('color', custom_colors))
    for i in range(0, len(all_populations)):
        ax.plot(all_populations[i][1,:], all_populations[i][2,:], '--', alpha=0.7) #Plot predators vs preys
        ax.scatter(all_populations[i][1, 0], all_populations[i][2, 0], marker='s', color='blue', s=50)
        ax.scatter(all_populations[i][1,-1], all_populations[i][2,-1], marker='s', color='red', s=50)
        ax.axline((0,0), slope=ratio, linewidth=0.75, color="blue")
        ax.axline((0, dimension[0]*dimension[1]), (dimension[0]*dimension[1], 0), linewidth=1.5,color="black")
        ax.axhline(xmin=0,xmax=dimension[0]*dimension[1], color="black", ls='-.') #Prey extinction
        ax.axvline(ymin=0,ymax=dimension[0]*dimension[1], color="black", ls='-.') #Predator extinction
    ax.set_xlabel('Predator population')
    ax.set_ylabel('Prey population')
    ax.set_title(f'Phase diagram with a ratio of {ratio} predator to prey')

    ax.grid(True,linewidth=0.1)
    plt.savefig(f"plots/Energies/Phase_Diagram_ratio{ratio}.svg")
    plt.savefig(f"plots/Energies/Phase_Diagram_ratio{ratio}.pdf")
    plt.tight_layout()
    plt.show()  
