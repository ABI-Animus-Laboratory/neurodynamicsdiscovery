import numpy as np
import nest
        
def ssd_with_l1(m1, m2, lamb, weights):
    '''
    Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
    '''
    sum_squared_difference = np.sum(0.5 * (m1 - m2) ** 2)
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
    num_neurons = num_pyr + num_int + 50
    print(num_pyr)

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
        print(matrix[i][num_pyr+40:num_pyr+40+num_int])
        print(np.abs(np.random.normal(1, scale=0.4, size=num_int)))
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
    