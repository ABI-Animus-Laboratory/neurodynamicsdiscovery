import numpy as np


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