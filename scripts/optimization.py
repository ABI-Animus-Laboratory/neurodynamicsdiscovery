import numpy as np
import nest
from scripts import model
from simanneal import Annealer

class SimulatedAnnealing1(Annealer):
    def __init__(self, state, place_obs, int_obs, lamb, categorized_neurons):
        self.categorized_neurons = categorized_neurons
        self.place_obs = place_obs
        self.int_obs = int_obs
        self.lamb = lamb
        self.state = state

    def energy(self):
        #Change this later
        network = model.Model1(self.categorized_neurons, self.state)
        network.simulate()
        place_pred = network.get_voltage_traces('Place')
        int_pred = network.get_voltage_traces('Inter')
        cost_function = self.ssd_with_l1_with_int(place_pred, int_pred)
        return cost_function
    
    def move(self):
        for i in range(3000):
            x = np.random.randint(0, np.shape(self.state)[1])
            y = np.random.randint(0, np.shape(self.state)[0])
            if self.state[x][y] != 0:
                self.state[x][y] = min(max(self.state[x][y] + np.random.uniform(-0.2, 0.2), 0.01), 3)    

    def ssd_with_l1_with_int(self, place_pred, int_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays, with interneurons
        '''

        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2) + np.sum(0.5 * (int_pred - self.int_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.weights))
        return sum_squared_difference + l1_penalty

class SimulatedAnnealing2(Annealer):
    def __init__(self, weights, place_obs, lamb, categorized_neurons, spike_weights):
        self.state = weights, spike_weights
        self.categorized_neurons = categorized_neurons
        self.place_obs = place_obs
        self.lamb = lamb

    def energy(self):
        network = model.Model2(self.categorized_neurons, self.state[1], self.state[0])
        network.simulate()
        place_pred = network.get_voltage_traces('Place')
        cost_function = self.ssd_with_l1(place_pred)
        return cost_function
    
    def move(self):

        weights, spike_weights = self.state
        for i in range(10):
            x = np.random.randint(0, 5)
            y = np.random.randint(0, 5)
            weights[x][y] = min(max(weights[x][y] + np.random.uniform(-0.2, 0.2), 0.01), 3)    
        
        for i in range(np.size(spike_weights, axis=0)):
            for j in range(np.size(spike_weights, axis=1)):
                if np.random.rand(1) > 0.8 and spike_weights[i][j] != 0:
                    spike_weights[i][j] = min(max(spike_weights[i][j] + np.random.uniform(-0.2, 0.2), -5), 5)   

        self.state = weights, spike_weights 

    def ssd_with_l1(self, place_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
        '''
        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))
        return sum_squared_difference + l1_penalty


    


    