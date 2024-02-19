import numpy as np
import nest
        
def ssd_with_l1(m1, m2, lamb, weights):
    '''
    Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
    '''
    sum_squared_difference = np.sum(0.5 * (m1 - m2) ** 2)
    l1_penalty = np.sum(lamb * np.abs(weights))
    return sum_squared_difference + l1_penalty

def initialize_connectivity_matrix_normal_distribution():
    '''
    order and quantities
    206 pyramidal neurons
    20 ec
    20 ca3
    20 interneurons
    10 medial septum

    excitatory connections
    pyr -> pyr
    ca3 -> pyr, inter
    ec -> pyr, inter

    inh connections
    int -> pyr
    ms -> int
    '''
    num_neurons = 276
    matrix = np.zeros((num_neurons, num_neurons))
    #Pyramidal neurons
    for i in range(206):
        #Pyramidal to Pyramidal
        matrix[i][0:206] = np.abs(np.random.normal(1, scale=0.4, size=206))
    #EC
    for i in range(206, 226):
        #EC to Pyramidal
        matrix[i][0:206] = np.abs(np.random.normal(1, scale=0.4, size=206))
        #EC to Interneurons
        matrix[i][246:266] = np.abs(np.random.normal(1, scale=0.4, size=20))
    #CA3
    for i in range(226, 246):
        #CA3 to Pyramidal
        matrix[i][0:206] = np.abs(np.random.normal(1, scale=0.4, size=206))
        #CA3 to Interneurons
        matrix[i][246:266] = np.abs(np.random.normal(1, scale=0.4, size=20))
    #Interneurons
    for i in range(246, 266):
        #Interneurons to Pyramidal
        matrix[i][0:206] = np.abs(np.random.normal(1, scale=0.4, size=206))
    #Medial Septum
    for i in range(266, 276):
        #Medial Septum to Interneurons
        matrix[i][246:266] = np.abs(np.random.normal(1, scale=0.4, size=20))

    return matrix
    