import numpy as np
import nest
import itertools
import os
import shutil
import math
import copy
import time
import pickle
import random
import matplotlib.pyplot as plt
from scripts import model
from simanneal import Annealer

nest.set_verbosity('M_FATAL')

class SimulatedAnnealing1(Annealer):
    def __init__(self, state, place_obs, int_obs, lamb, categorized_neurons, move_params = None):
        self.categorized_neurons = categorized_neurons
        self.place_obs = place_obs
        self.int_obs = int_obs
        self.lamb = lamb
        self.state = state
        self.objs = []
        self.best_objs = []

        if move_params is None:
            #Default move params
            self.move_params = {
                'num_weights': 3000,
                'weights_range': (-5, 5),
                'weights_change_range': (-0.2, 0.2),
            }
        else:
            self.move_params = move_params

        self.voltage_traces_place = None
        self.voltage_traces_int = None

    def energy(self):
        #Change this later
        network = model.Model1(self.categorized_neurons, self.state)
        network.simulate()
        place_pred = network.get_voltage_traces('Place')
        int_pred = network.get_voltage_traces('Inter')

        self.objs.append(cost_function)
        if self.best_objs == []:
            self.best_objs.append(cost_function)
        else:
            if cost_function < self.best_objs[-1]:
                self.best_objs.append(cost_function)
            else:
                self.best_objs.append(self.best_objs[-1])

        cost_function = self.ssd_with_l1(place_pred, int_pred)
        self.voltage_traces_place = network.get_voltage_traces('Place')
        self.voltage_traces_int = network.get_voltage_traces('Inter')

        
        return cost_function
    
    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        #modifications
        objs = []
        best_objs = []

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            #modification
            objs.append(E)

            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            #modification
            best_objs.append(self.best_energy)
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        #modification
        self.objs = objs
        self.best_objs = best_objs

        # Return best state and energy
        return self.best_state, self.best_energy

    def move(self):
        for i in range(self.move_params['num_weights']):
            x = np.random.randint(0, np.shape(self.state)[1])
            y = np.random.randint(0, np.shape(self.state)[0])
            if self.state[x][y] != 0:
                self.state[x][y] = min(max(self.state[x][y] + np.random.uniform(self.move_params['weights_change_range'][0], 
                                                                                self.move_params['weights_change_range'][1]), 
                                                                                self.move_params['weights_range'][0]), 
                                                                                self.move_params['weights_range'][1])    

    def ssd_with_l1(self, place_pred, int_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays, with interneurons
        '''

        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2) + np.sum(0.5 * (int_pred - self.int_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.weights))
        return sum_squared_difference + l1_penalty
    
    def plot_objs(self):
        plt.figure(figsize=(15, 5))
        plt.title('Objective Function vs Best Objective Function')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.objs, label='Objective Function')

        plt.plot(self.best_objs, label='Best Objective Function')
        plt.legend()

class SimulatedAnnealing2(Annealer):

    def __init__(self, weights, place_obs, lamb, categorized_neurons, spike_weights, move_params = None):
        
        self.state = weights, spike_weights

        self.objs = []
        self.best_objs = []

        self.categorized_neurons = categorized_neurons
        self.place_obs = place_obs
        self.lamb = lamb

        self.accepted = []

        if move_params is None:
            #Default move params
            self.move_params = {
                'weights_range': (-5, 5),
                'weights_change_range': (-0.2, 0.2),
                'spike_weights_range': (-5, 5),
                'spike_weights_change_range': (-2, 2),
                'spike_weights_prob' : 0.2,
                'optimise_spike_weights': False
            }
        else:
            self.move_params = move_params

        self.voltage_traces = None
        

    def energy(self):
        network = model.Model2(self.categorized_neurons, self.state[1], self.state[0])
        network.simulate()
        place_pred = network.get_voltage_traces()
        cost_function = self.ssd_with_l1(place_pred)

        self.voltage_traces = network.get_voltage_traces()
        return cost_function
    
    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        #modifications
        objs = []
        best_objs = []

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            #modification
            objs.append(E)

            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            #modification
            best_objs.append(self.best_energy)
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        #modification
        self.objs = objs
        self.best_objs = best_objs

        # Return best state and energy
        return self.best_state, self.best_energy

    def move(self):

        weights, spike_weights = self.state

        for i in range(5):
            for j in range(5):
                if weights[i][j] != 0:
                    weights[i][j] = min(max(weights[i][j] + np.random.uniform(self.move_params['weights_change_range'][0],
                                                                            self.move_params['weights_change_range'][1]), 
                                                                            self.move_params['weights_range'][0]), self.move_params['weights_range'][1])    
        if self.move_params['optimise_spike_weights']:
            for i in range(np.size(spike_weights, axis=0)):
                for j in range(np.size(spike_weights, axis=1)):
                    if np.random.rand(1) < self.move_params['spike_weights_prob']:
                        spike_weights[i][j] = min(max(spike_weights[i][j] + np.random.uniform(self.move_params['spike_weights_change_range'][0],
                                                                                self.move_params['spike_weights_change_range'][1]), 
                                                                                self.move_params['spike_weights_range'][0]), self.move_params['spike_weights_range'][1]) 

        self.state = weights, spike_weights 

    def plot_objs(self):
        plt.figure(figsize=(15, 5))
        plt.title('Objective Function vs Best Objective Function')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.objs, label='Objective Function')

        plt.plot(self.best_objs, label='Best Objective Function')
        plt.legend()
        

    def ssd_with_l1(self, place_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
        '''
        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))

        return sum_squared_difference + l1_penalty

    def generate_param_permutations(self):
        permutations = itertools.product(*(self.move_param_ranges[param] for param in self.move_param_ranges))
        param_perms = [dict(zip(self.move_param_ranges.keys(), permutation)) for permutation in permutations]
        return param_perms


class SensitivityAnalysis:
    def __init__(self, move_param_ranges, optimiser, save_results = False):
        self.move_param_ranges = move_param_ranges
        self.optimiser = optimiser
        self.save_results = save_results
    
    def generate_param_permutations(self):
        permutations = itertools.product(*(self.move_param_ranges[param] for param in self.move_param_ranges))
        param_perms = [dict(zip(self.move_param_ranges.keys(), permutation)) for permutation in permutations]
        return param_perms
                
    def run_analysis(self):
        param_perms = self.generate_param_permutations()
        if isinstance(self.optimiser, Annealer):
            for i in range(len(param_perms)):
                self.optimiser.move_params = param_perms[i]

                results = []
                for run in range(3):
                    temp_optimiser = copy.deepcopy(self.optimiser)
                    x, func = temp_optimiser.anneal()
                    results.append(temp_optimiser.best_energy)
                    temp_optimiser.plot_objs()

                if self.save_results:
                    results_dir = '/hpc/mzhu843/modelling/nest/notebooks/optimization/simulated_annealing/model2/sensitivity_analysis results/'
                    perm_dir = '/hpc/mzhu843/modelling/nest/notebooks/optimization/simulated_annealing/model2/sensitivity_analysis results/' + str(i + 1) + '/'
                    if not os.path.exists(perm_dir):
                        os.makedirs(perm_dir)
                    else:
                        shutil.rmtree(perm_dir)         
                        os.makedirs(perm_dir)

                    
                    plt.savefig(perm_dir + 'Objective Functions '  + str(i+1) + '.png')
                    plt.close()

                    with open(perm_dir + 'params.pkl', 'wb') as file:
                        pickle.dump(param_perms[i], file)

                    with open(perm_dir + 'data.txt', 'w') as data:
                        data.write('Parameters: ' + str(param_perms[i]) + '\n')
                        data.write('\n')
                        data.write('Run 1 Best Energy: ' + str(results[0]) + '\n')
                        data.write('Run 2 Best Energy: ' + str(results[1]) + '\n')
                        data.write('Run 3 Best Energy: ' + str(results[2]) + '\n')
                        data.write('\n')
                        




                
    




    