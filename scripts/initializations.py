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
    