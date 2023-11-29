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
            nest.Connect(presyn_conn, postsyn_conn, syn_spec={'weight': nest.random.normal(mean=1 * G_e, std = 0.4 * G_e)})
    for i in conns_i:
        presyn_conn = i[0]
        postsyn_conns = i[1]
        for postsyn_conn in postsyn_conns:
            nest.Connect(presyn_conn, postsyn_conn, syn_spec={'weight': nest.random.normal(mean= 1 * G_i, std = 0.4 * -G_i)})
    