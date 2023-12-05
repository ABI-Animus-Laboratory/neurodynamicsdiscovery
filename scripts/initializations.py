import numpy as np
import matplotlib.pyplot as plt
import nest

#Initializes a group of neurons as well as their default parameters, returns the group
#By default, sets initial membrane potential to a uniform distribution between reset and threshold voltages
def initialize_neuron_group(type, n=1, params={}, initial_vm = None):
    neurons = nest.Create(type, n=n, params=params)
    if not initial_vm:
        Vth = neurons.get('V_th')[0]
        Vreset = neurons.get('V_reset')[0]
        neurons.set({"V_m": Vreset + nest.random.uniform(0.0, Vth-Vreset)})
    else:
        neurons.set({"V_m": initial_vm})
    return neurons

def initialize_weights(conns_e, conns_i, G_e, G_i):
    for e in conns_e:
        presyn_conn = e[0]
        postsyn_conns = e[1]
        for postsyn_conn in postsyn_conns:
            nest.Connect(presyn_conn, postsyn_conn, syn_spec={'weight': G_e * nest.random.normal(mean=1, std = 0.4)})
    for i in conns_i:
        presyn_conn = i[0]
        postsyn_conns = i[1]
        for postsyn_conn in postsyn_conns:
            nest.Connect(presyn_conn, postsyn_conn, syn_spec={'weight': G_i * nest.random.normal(mean=1, std = 0.4)})

def initialize_connectivity_matrix_hardcoded():
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
    for i in range(246, 276):
        #Medial Septum to Interneurons
        matrix[i][246:266] = np.abs(np.random.normal(1, scale=0.4, size=20))

    return matrix


    