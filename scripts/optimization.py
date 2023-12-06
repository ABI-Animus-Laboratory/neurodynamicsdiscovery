import numpy as np
import nest
from params import pyr_hcamp_deco2012
from params import int_hcamp_deco2012
from scripts import visualizations
from scripts import initializations

def simulation_results_to_spike_trains(results, runtime):
    #Input is a list of np arrays, each representing the times at which a neuron spiked
    #Output is an array of arrays, each of which is a spike train
    num_neurons = len(results)
    spike_trains = np.zeros((num_neurons, runtime))
    for i in range(num_neurons):
        spike_train = np.zeros(runtime)
        for time in results[i]:
            spike_train[int(time)] = 1.0
        spike_trains[i] = spike_train
    return spike_trains

def connect_weights(A, B, W):
    for i, pre in enumerate(A):
        weights = W[:, i]
        nonzero_indices = np.where(weights != 0)[0]
        weights = weights[nonzero_indices]
        post = B[nonzero_indices]
        pre_array = np.ones(len(nonzero_indices), dtype=int) * pre.get('global_id')
        nest.Connect(pre_array, post, conn_spec='one_to_one', syn_spec={'weight': weights})
        
def initialize_connections_hardcoded(pyr, ec, ca3, inter, ms, weights):
    pyr_pyr_conns = weights[0:206, 0:206]
    connect_weights(pyr, pyr, pyr_pyr_conns)

    ec_pyr_conns = weights[206:226, 0:206]
    connect_weights(ec, pyr, ec_pyr_conns)

    ec_inter_conns = weights[206:226, 246:266]
    connect_weights(ec, inter, ec_inter_conns)

    ca3_pyr_conns = weights[226:246, 0:206]
    connect_weights(ca3, pyr, ca3_pyr_conns)

    ca3_inter_conns = weights[226:246, 246:266]
    connect_weights(ca3, inter, ca3_inter_conns)

    inter_pyr_conns = weights[246:266, 0:206]
    connect_weights(inter, pyr, inter_pyr_conns)

    ms_inter_conns = weights[266:276, 246:266]
    connect_weights(ms, inter, ms_inter_conns)
    
def ssd_with_l1(m1, m2, lamb):
    '''
    Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
    '''
    squared_difference = (m1 - m2) ** 2
    l1_penalty = np.sum(lamb * m1)
    return np.sum(squared_difference) + l1_penalty

def simulate(weights):
    nest.ResetKernel()
    nest.resolution = 1
    runtime = 17988
    gamma_rate = 40
    theta_rate = 7
    G_e = 10
    G_i = -3
    pyr = initializations.initialize_neuron_group('iaf_psc_alpha', 206, pyr_hcamp_deco2012.params)
    inter = initializations.initialize_neuron_group('iaf_psc_alpha', 20, int_hcamp_deco2012.params)
    ec_input = nest.Create('poisson_generator')
    ec_input.set(rate=gamma_rate)
    ec_parrot = nest.Create('parrot_neuron', n=20)
    nest.Connect(ec_input, ec_parrot)

    ca3_input = nest.Create('poisson_generator')
    ca3_input.set(rate=gamma_rate)
    ca3_parrot = nest.Create('parrot_neuron', n=20)
    nest.Connect(ca3_input, ca3_parrot)

    ms_input = nest.Create('poisson_generator')
    ms_input.set(rate=theta_rate)
    ms_parrot = nest.Create('parrot_neuron', n=10)
    nest.Connect(ms_input, ms_parrot)

    initialize_connections_hardcoded(pyr, ec_parrot, ca3_parrot, inter, ms_parrot, weights)

    spike_recorder = nest.Create('spike_recorder')
    nest.Connect(pyr, spike_recorder)

    nest.Simulate(runtime)

    spikes = nest.GetStatus(spike_recorder, "events")[0]
    senders = spikes["senders"]
    times = spikes["times"]
    results = [times[senders == neuron_id] for neuron_id in ec_input]

    return results

    


