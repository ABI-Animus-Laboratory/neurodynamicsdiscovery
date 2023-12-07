import numpy as np
import nest
from scripts import visualizations, initializations, experiments

def simulation_results_to_spike_trains(results, runtime):
    #Input is a list of np arrays, each representing the times at which a neuron spiked
    #Output is an array of arrays, each of which is a spike train
    num_neurons = len(results)
    spike_trains = np.zeros((num_neurons, runtime))
    for i in range(num_neurons):
        spike_train = np.zeros(runtime)
        for time in results[i]:
            spike_train[int(time) - 1] = 1.0
        spike_trains[i] = spike_train
    return spike_trains

def connect_weights(A, B, W, G, V):
    for i, pre in enumerate(A):
        weights = W[:, i]
        nonzero_indices = np.where(weights != 0)[0]
        weights = weights[nonzero_indices]
        post = B[nonzero_indices]
        pre_array = np.ones(len(nonzero_indices), dtype=int) * pre.get('global_id')
        nest.Connect(pre_array, post, conn_spec='one_to_one', syn_spec={'weight': weights * G * V})
        
def set_connection_weights_s1(pyr, ec, ca3, inter, ms, weights, G_e, G_i, V_e, V_i):
    pyr_pyr_conns = weights[0:206, 0:206]
    connect_weights(pyr, pyr, pyr_pyr_conns, G_e, V_e)

    ec_pyr_conns = weights[206:226, 0:206]
    connect_weights(ec, pyr, ec_pyr_conns, G_e, V_e)

    ec_inter_conns = weights[206:226, 246:266]
    connect_weights(ec, inter, ec_inter_conns, G_e, V_e)

    ca3_pyr_conns = weights[226:246, 0:206]
    connect_weights(ca3, pyr, ca3_pyr_conns, G_e, V_e)

    ca3_inter_conns = weights[226:246, 246:266]
    connect_weights(ca3, inter, ca3_inter_conns, G_e, V_e)

    inter_pyr_conns = weights[246:266, 0:206]
    connect_weights(inter, pyr, inter_pyr_conns, G_i, V_i)

    ms_inter_conns = weights[266:276, 246:266]
    connect_weights(ms, inter, ms_inter_conns, G_i, V_i)
    
def ssd_with_l1(m1, m2, lamb, weights):
    '''
    Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
    '''
    squared_difference = 0.5 * (m1 - m2) ** 2
    l1_penalty = np.sum(lamb * weights)
    return np.sum(squared_difference) + l1_penalty

def simulate(weights, setup=1):
    nest.ResetKernel()
    if setup == 1:
        results = experiments.run_s1(weights)
    return results

    


