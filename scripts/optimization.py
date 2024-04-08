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
        network = model.NeuronalNetwork(self.state, self.categorized_neurons)
        network.simulate()
        place_pred = network.get_voltage_traces('Place')
        int_pred = network.get_voltage_traces('Inter')
        cost_function = ssd_with_l1(self.place_obs, place_pred, self.int_obs, int_pred, self.lamb, self.state)
        return cost_function
    
    def move(self):
        for i in range(4000):
            x = np.random.randint(0, 175)
            y = np.random.randint(0, 175)
            if self.state[x][y] != 0:
                self.state[x][y] = min(max(self.state[x][y] + np.random.uniform(-0.2, 0.2), 0.01), 3)    
        
def ssd_with_l1(place_obs, place_pred, int_obs, int_pred, lamb, weights):
    '''
    Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
    '''

    sum_squared_difference = np.sum(0.5 * (place_pred - place_obs) ** 2) + np.sum(0.5 * (int_pred - int_obs) ** 2)
    l1_penalty = np.sum(lamb * np.abs(weights))
    return sum_squared_difference + l1_penalty

def initialize_connectivity_matrix_normal_distribution(categorised_neurons):
    '''
    order and quantities
    num_pyr pyramidal neurons
    20 ec
    20 ca3
    num_int interneurons
    10 medial septum

    excitatory connections
    pyr -> pyr
    ca3 -> pyr, inter
    ec -> pyr, inter

    inh connections
    int -> pyr
    ms -> int
    '''

    num_pyr = len(categorised_neurons['Place'])
    num_int = len(categorised_neurons['Interneuron'])
    num_neurons = num_pyr + num_int + 50 #20 EC + 20 CA3 + 10 MS neurons = 50 total neurons

    matrix = np.zeros((num_neurons, num_neurons))
    #Pyramidal neurons
    for i in range(num_pyr):
        #Pyramidal to Pyramidal
        matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
    #EC
    for i in range(num_pyr, num_pyr+20):
        #EC to Pyramidal
        matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
        #EC to Interneurons
        matrix[i][num_pyr+40:num_pyr+40+num_int] = np.abs(np.random.normal(1, scale=0.4, size=num_int))
    #CA3
    for i in range(num_pyr+20, num_pyr+40):
        #CA3 to Pyramidal
        matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
        #CA3 to Interneurons
        matrix[i][num_pyr+40:num_pyr+40+num_int] = np.abs(np.random.normal(1, scale=0.4, size=num_int))
    #Interneurons
    for i in range(num_pyr+40, num_pyr+40+num_int):
        #Interneurons to Pyramidal
        matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
    #Medial Septum
    for i in range(num_pyr+40+num_int, num_pyr+50+num_int):
        #Medial Septum to Interneurons
        matrix[i][num_pyr+40:num_pyr+40+num_int] = np.abs(np.random.normal(1, scale=0.4, size=num_int))

    return matrix
    