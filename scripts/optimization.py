import numpy as np
import nest
from scripts import model
from simanneal import Annealer

class SimulatedAnnealing(Annealer):
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
        cost_function = ssd_with_l1(self.place_obs, place_pred, self.int_obs, int_pred, self.lamb, self.state)
        return cost_function
    
    def move(self):
        for i in range(3000):
            x = np.random.randint(0, np.shape(self.state)[1])
            y = np.random.randint(0, np.shape(self.state)[0])
            if self.state[x][y] != 0:
                self.state[x][y] = min(max(self.state[x][y] + np.random.uniform(-0.2, 0.2), 0.01), 3)    
        
def ssd_with_l1(place_obs, place_pred, int_obs, int_pred, lamb, weights):
    '''
    Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
    '''

    sum_squared_difference = np.sum(0.5 * (place_pred - place_obs) ** 2) + np.sum(0.5 * (int_pred - int_obs) ** 2)
    l1_penalty = np.sum(lamb * np.abs(weights))
    return sum_squared_difference + l1_penalty


    