'''
This file contains a numerical simulation based on the analyical LV model for a predator-prey system.
'''

import numpy as np
import matplotlib.pyplot as plt

##### Parameters ######

sim_time = 10000
dt = 0.01
num_traj = 8
growth_prey = 'exp'

if growth_prey == 'logistic':
    alpha, beta, gamma, delta = 1, 50, 48, 1
    prey_caps = np.arange(40,100) # for logistic growth bifurcation

elif growth_prey == 'exp':
    alpha, beta, gamma, delta = 3e-3, 0.3, 0.1, 3e-3
 

##### Lotka-Volterra ######

def lotka_volterra(pred, prey, growth_prey, alpha, beta, gamma, delta, prey_cap = 100):
    '''
    This function defines the updates of predator and prey populations
    as provided by the Lotka Volterra system of ODEs.
    '''
    if growth_prey == 'logistic':
        
        dpred_dt = alpha*pred*prey-beta*pred
        dprey_dt = gamma*prey*(1-prey/prey_cap)-delta*pred*prey
        
    elif growth_prey == 'exp':
        
        dpred_dt = alpha*pred*prey-beta*pred
        dprey_dt = gamma*prey-delta*pred*prey
        
    return dpred_dt, dprey_dt

##### Simulation ######

if growth_prey == 'logistic':
    for prey_cap in prey_caps: # iterate over all prey capacities

        plt.figure(figsize=(6, 4)) # set up figure that hosts the different trajectories
        
        for _ in range(num_traj):

            pred = np.zeros(sim_time)
            prey = np.zeros(sim_time)
            # randomly initialize pred \in (1,60) and prey \in (1, prey_cap)
            pred[0] = np.random.uniform(1,60)
            prey[0] =np.random.uniform(1,prey_cap)

            for t in range(sim_time-1):         # integrate for sim_time timesteps
                dpred_dt, dprey_dt = lotka_volterra(pred[t], prey[t], growth_prey, alpha, beta, gamma, delta, prey_cap)
                pred[t+1] = max(pred[t] + dt*dpred_dt,0)
                prey[t+1] = max(prey[t] + dt*dprey_dt,0)
                
            plt.plot(pred, prey)  # add trajectory 
        
        # Plotting of all trajectories
        plt.title(f"Phase Portrait for prey capacity N = {prey_cap}", size = 22)
        plt.xlabel("Predator population (x)", size = 18)
        plt.ylabel("Prey population (y)", size = 18)
        stable1_pred, stable1_prey = 0, 0 
        plt.plot(stable1_pred, stable1_prey, marker='o', markersize=8, label = 'Trivial stability (extinction)')
        stable2_pred, stable2_prey = 0,prey_cap
        plt.plot(stable2_pred, stable2_prey, marker='o', markersize=8, label = 'Trivial stability (predator extinction)')
        stable3_pred, stable3_prey = max(0,gamma/delta*(1-beta/(prey_cap*alpha))), max(0,beta/alpha)
        plt.plot(stable3_pred, stable3_prey, marker='o', markersize=8, label = 'Non-Trivial stability point (co-existence)')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"plots/analytical_LV/{growth_prey}/trajectory_cap_{prey_cap}.png")
        plt.close()

elif growth_prey == 'exp':
            
        ########### PHASE PLOT ############
        
        plt.figure(figsize=(8, 5)) # set up figure that hosts the different trajectories
                
        for _ in range(num_traj):

            pred = np.zeros(sim_time)
            prey = np.zeros(sim_time)
           
            pred[0] = np.random.uniform(3,30)
            prey[0] = pred[0] * 3

            for t in range(sim_time-1):      # integrate for sim_time timesteps
                dpred_dt, dprey_dt = lotka_volterra(pred[t], prey[t], growth_prey, alpha, beta, gamma, delta)
                pred[t+1] = max(pred[t] + dt*dpred_dt,0)
                prey[t+1] = max(prey[t] + dt*dprey_dt,0)
            
                
            plt.plot(pred, prey)  # add trajectory 
            plt.plot(pred[0],prey[0],marker='x')
            
        # Plotting of all trajectories
        
        plt.title(f"Phase portrait LV model (exponential prey growth)", size = 20)
        plt.xlabel("Predator population (x)", size = 18)
        plt.ylabel("Prey population (y)", size = 18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        stable1_pred, stable1_prey = 0, 0 
        plt.plot(stable1_pred, stable1_prey, marker='o', markersize=8, label = 'Trivial stability (extinction)')
        stable3_pred, stable3_prey = gamma/delta, beta/alpha
        plt.plot(stable3_pred, stable3_prey, marker='o', markersize=8, label = 'Non-Trivial orbit point (co-existence)')
        plt.tight_layout()
        plt.legend(fontsize=14)
        plt.savefig(f"plots/analytical_LV/{growth_prey}/phase_reproduction_{gamma}.png")
        plt.close()
        
        ########### POPULATION DYNAMICS ############

        predators = []
        preys = []
        pred = np.zeros(sim_time)
        prey = np.zeros(sim_time)
        pred[0] = np.random.uniform(3,30)
        prey[0] = pred[0] * 3
        predators.append(pred[0])
        preys.append(prey[0])
        
        for t in range(sim_time-1):         # integrate for sim_time timesteps
                dpred_dt, dprey_dt = lotka_volterra(pred[t], prey[t], growth_prey, alpha, beta, gamma, delta)
                pred[t+1] = max(pred[t] + dt*dpred_dt,0)
                prey[t+1] = max(prey[t] + dt*dprey_dt,0)
                predators.append(pred[t+1])
                preys.append(prey[t+1])
        
        # Create an array for time steps
        time_steps = np.arange(sim_time)

        # Plot the population dynamics
        plt.figure(figsize=(9, 5))
        plt.title(f"Population dynamics LV model (exponential prey growth)", size = 20)
        plt.plot(time_steps, predators, label='Predators')
        plt.plot(time_steps, preys, label='Preys')
        plt.xlabel('Evolution steps', size = 18)
        plt.ylabel('Population numbers', size = 18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=14)

        plt.savefig(f"plots/analytical_LV/{growth_prey}/dynamics_reproduction_{gamma}.png")
        plt.close()        
