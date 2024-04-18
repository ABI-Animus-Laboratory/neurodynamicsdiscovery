import numpy as np
import nest
import itertools
import os
import datetime
import math
import copy
import time
import pickle
import random
import pandas as pd
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

    def __init__(self, weights, place_obs, lamb, categorized_neurons, input_weights, move_params = None):
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
                'input_weights_range': (-5, 5),
                'input_weights_change_range': (-2, 2),
                'input_weights_prob' : 0.2,
                'optimise_input_weights': False
            }
        else:
            self.move_params = move_params

        self.voltage_traces = None

        self.state = weights, input_weights

    def energy(self):
        network = model.Model2(self.categorized_neurons, self.state[0], self.state[1])
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

        weights, input_weights = self.state

        for i in range(5):
            for j in range(5):
                if weights[i][j] != 0:
                    weights[i][j] = min(max(weights[i][j] + np.random.uniform(self.move_params['weights_change_range'][0],
                                                                            self.move_params['weights_change_range'][1]), 
                                                                         self.move_params['weights_range'][0]), self.move_params['weights_range'][1])    
                    
        #Need to modify this later so its more efficient
        if self.move_params['optimise_input_weights']:
            for i in range(np.size(input_weights, axis=0)):
                for j in range(np.size(input_weights, axis=1)):
                    if np.random.rand(1) < self.move_params['input_weights_prob']:
                        input_weights[i][j] = min(max(input_weights[i][j] + np.random.uniform(self.move_params['input_weights_change_range'][0],
                                                                                self.move_params['input_weights_change_range'][1]), 
                                                                                self.move_params['input_weights_range'][0]), self.move_params['input_weights_range'][1]) 

        self.state = weights, input_weights 

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
    def __init__(self, move_param_ranges, optimiser, param_keys, num_iters = 3, save_results = False):
        self.move_param_ranges = move_param_ranges
        self.optimiser = optimiser
        self.save_results = save_results
        self.param_keys = param_keys
        self.num_iters = num_iters
    
    def generate_param_permutations(self):
        permutations = itertools.product(*(self.move_param_ranges[param] for param in self.move_param_ranges))
        param_perms = [dict(zip(self.move_param_ranges.keys(), permutation)) for permutation in permutations]
        return param_perms
    
    def create_experiment_dir(self):

        current_datetime = str(datetime.datetime.now())[:-7]
        results_dir = '/hpc/mzhu843/modelling/nest/notebooks/optimization/simulated_annealing/model2/sensitivity_analysis results/'
        experiment_dir = results_dir + current_datetime
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        return experiment_dir + '/'

    def run_analysis(self):

        param_perms = self.generate_param_permutations()
        param_metrics = []

        if self.save_results:
            experiment_dir = self.create_experiment_dir()

        start = time.time()

        for i in range(len(param_perms)):

            param_perm_subset = {key: param_perms[i][key] for key in self.param_keys if key in param_perms[i]}

            if self.save_results:
                perm_dir = experiment_dir + str(param_perm_subset) + '/'
                if not os.path.exists(perm_dir):
                    os.makedirs(perm_dir)
            
            if isinstance(self.optimiser, Annealer):
                self.optimiser.move_params = param_perms[i]

                best_objs = []
                objs = []
                optimized_weights = []
                
                for run in range(self.num_iters):
                    temp_optimiser = copy.deepcopy(self.optimiser)
                    x = temp_optimiser.anneal()[0]
                    best_objs.append(temp_optimiser.best_energy)
                    objs.append(temp_optimiser.objs)
                    flattened_objs = [x for sublist in objs for x in sublist]
                    optimized_weights.append(x)
                    temp_optimiser.plot_objs()

                    if self.save_results:
                        plt.savefig(perm_dir + f'Run {run+1} Objective Functions.png')
                        plt.close()

                param_metrics.append((param_perm_subset, best_objs[0], best_objs[1], best_objs[2], sum(best_objs)/len(best_objs), 
                                           sum(objs[0]) / len(objs[0]), sum(objs[1]) / len(objs[1]), sum(objs[2]) / len(objs[2]), sum(flattened_objs) / len(flattened_objs)))

                if self.save_results:

                    np.save(perm_dir + 'Run 1 Optimized Weights', optimized_weights[0][0])
                    np.save(perm_dir + 'Run 2 Optimized Weights', optimized_weights[1][0])
                    np.save(perm_dir + 'Run 3 Optimized Weights', optimized_weights[2][0])

                    with open(perm_dir + 'params.pkl', 'wb') as file:
                        pickle.dump(param_perms[i], file)

        if self.save_results:

            param_metrics = sorted(param_metrics, key=lambda x: x[4])

            end = time.time()
            duration = end - start
            df_cols = ['Parameters', 'Best Obj 1', 'Best Obj 2', 'Best Obj 3', 'Mean Best Obj', 'Mean Obj 1', 'Mean Obj 2', 'Mean Obj 3', 'Mean Mean Obj']
            df = pd.DataFrame(param_metrics, columns = df_cols)
            df.to_csv(experiment_dir + 'summary_stats.csv')
            np.save(experiment_dir + 'weights', self.optimiser.state[0])
            np.save(experiment_dir + 'spike weights', self.optimiser.state[1])
            weights_statistics = get_2d_statistics(self.optimiser.state[0])
            input_weights_statistics = get_2d_statistics(self.optimiser.state[1])

            with open(experiment_dir + 'details.txt', 'w') as file:
                file.write(f'Duration: {duration} seconds\n')
                file.write(f'Number of Iterations Per Experiment: {self.num_iters}\n')
                file.write(f'Number of Optimisation Steps: {self.optimiser.steps}\n\n')
                file.write(f'PARAMETER RANGES\n')
                for param_range in self.move_param_ranges.keys():
                    file.write(f'{param_range}: {self.move_param_ranges[param_range]}\n') 
                for statistic in weights_statistics.keys():
                    file.write(f'{statistic}: {weights_statistics[statistic]}\n') 
                for statistic in input_weights_statistics.keys():
                    file.write(f'{statistic}: {input_weights_statistics[statistic]}\n') 
        
        return param_metrics

def get_2d_statistics(twod_array):
    
    statistics_dict = {}

    statistics_dict['mean'] = np.mean(twod_array)
    statistics_dict['median'] = np.median(twod_array)
    statistics_dict['std'] = np.std(twod_array)
    statistics_dict['min'] = np.min(twod_array)
    statistics_dict['max'] = np.max(twod_array)

    return statistics_dict



                        




                
    




    