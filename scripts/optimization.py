import numpy as np
import nest
import matplotlib.pyplot as plt
from scripts import model
from simanneal import Annealer

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
                'weights_min': 0.01,
                'weights_max' : 3,
                'weights_change_min' : -0.2,
                'weights_change_max' : 0.2,
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
    
    def move(self):
        for i in range(self.move_params['num_weights']):
            x = np.random.randint(0, np.shape(self.state)[1])
            y = np.random.randint(0, np.shape(self.state)[0])
            if self.state[x][y] != 0:
                self.state[x][y] = min(max(self.state[x][y] + np.random.uniform(self.move_params['weights_change_min'], 
                                                                                self.move_params['weights_change_max']), 
                                                                                self.move_params['weights_min']), 
                                                                                self.move_params['weights_max'])    

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
        plt.xlabel('Runtime (ms)')
        plt.ylabel('Voltage (V)')
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

        if move_params is None:
            #Default move params
            self.move_params = {
                'weights_min': 0.01,
                'weights_max' : 3,
                'weights_change_min' : -0.2,
                'weights_change_max' : 0.2,
                'spike_weights_min': -3,
                'spike_weights_max' : 3,
                'spike_weights_change_min' : -0.2,
                'spike_weights_change_max' : 0.2,
                'spike_weights_prob' : 0.2
            }
        else:
            self.move_params = move_params

        self.voltage_traces = None
        

    def energy(self):
        network = model.Model2(self.categorized_neurons, self.state[1], self.state[0])
        network.simulate()
        place_pred = network.get_voltage_traces()
        cost_function = self.ssd_with_l1(place_pred)

        self.objs.append(cost_function)
        if self.best_objs == []:
            self.best_objs.append(cost_function)
        else:
            if cost_function < self.best_objs[-1]:
                self.best_objs.append(cost_function)
            else:
                self.best_objs.append(self.best_objs[-1])

        self.voltage_traces = network.get_voltage_traces()
        return cost_function
    
    def move(self):

        weights, spike_weights = self.state

        for i in range(np.size(weights, axis=0)):
            for j in range(np.size(weights, axis=1)):
                weights[i][j] = min(max(weights[i][j] + np.random.uniform(self.move_params['weights_change_min'],
                                                                          self.move_params['weights_change_max']), 
                                                                          self.move_params['weights_min']), self.move_params['weights_max'])    
        
        for i in range(np.size(spike_weights, axis=0)):
            for j in range(np.size(spike_weights, axis=1)):
                if np.random.rand(1) < self.move_params['spike_weights_prob']:
                    spike_weights[i][j] = min(max(spike_weights[i][j] + np.random.uniform(self.move_params['spike_weights_change_min'],
                                                                            self.move_params['spike_weights_change_max']), 
                                                                            self.move_params['spike_weights_min']), self.move_params['spike_weights_max']) 

        self.state = weights, spike_weights 

    def plot_objs(self):
        plt.figure(figsize=(15, 5))
        plt.title('Candidate Objective Function vs Best Objective Function')
        plt.xlabel('Simulation Timestep')
        plt.ylabel('Voltage (V)')
        plt.plot(self.objs, label='Candidate Objective Function')
        plt.plot(self.best_objs, label='Best Objective Function')
        plt.legend()
        

    def ssd_with_l1(self, place_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
        '''
        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))

        return sum_squared_difference + l1_penalty


    


    