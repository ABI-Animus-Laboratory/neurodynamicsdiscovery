import numpy as np
import matplotlib.pyplot as plt
import nest
from scripts import initializations, optimization
from params import pyr_hcamp_deco2012, int_hcamp_deco2012

class NeuronalNetwork:
    '''
    Class that represents the model
    '''
    def __init__(self, weights, G_e = 3.8, G_i = -1, runtime = 17988, gamma_rate = 40, theta_rate = 7):
        self.weights = weights
        self.G_e = G_e
        self.G_i = G_i
        self.runtime = runtime
        self.gamma_rate = gamma_rate
        self.theta_rate = theta_rate

        self.resolution = 1
        self.V_e = 5
        self.V_i = 5
        self.simulated = False

        self.num_neurons = 206

        self.spike_trains = None
        self.voltage_traces = None
        self.spike_recorder = None

    def simulate(self):
        '''
        Runs a simulation and assigns class variables 
        Takes in no inputs and has no output
        '''
        nest.ResetKernel()
        pyr = initialize_neuron_group('iaf_psc_alpha', 206, pyr_hcamp_deco2012.params)
        inter = initialize_neuron_group('iaf_psc_alpha', 20, int_hcamp_deco2012.params)
        ec_input = nest.Create('poisson_generator')
        ec_input.set(rate=self.gamma_rate)
        ec_parrot = nest.Create('parrot_neuron', n=20)
        nest.Connect(ec_input, ec_parrot)

        ca3_input = nest.Create('poisson_generator')
        ca3_input.set(rate=self.gamma_rate)
        ca3_parrot = nest.Create('parrot_neuron', n=20)
        nest.Connect(ca3_input, ca3_parrot)

        ms_input = nest.Create('poisson_generator')
        ms_input.set(rate=self.theta_rate)
        ms_parrot = nest.Create('parrot_neuron', n=10)
        nest.Connect(ms_input, ms_parrot)

        spike_recorder = nest.Create('spike_recorder')
        nest.Connect(pyr, spike_recorder)

        multimeter = nest.Create('multimeter')
        multimeter.set(record_from=["V_m"])
        nest.Connect(multimeter, pyr)

        optimization.set_connection_weights_s1(pyr, ec_parrot, ca3_parrot, inter, ms_parrot, self.weights, self.G_e, self.G_i, self.V_e, self.V_i)

        nest.Simulate(self.runtime)

        self.simulated = True

        spikes = nest.GetStatus(spike_recorder, "events")[0]
        senders = spikes["senders"]
        times = spikes["times"]

        dmm = multimeter.get()
        Vms = dmm["events"]["V_m"] #For some reason this only goes up to runtime - 1
        ts = dmm["events"]["times"]

        results = [times[senders == neuron_id] for neuron_id in pyr]
        results = simulation_results_to_spike_trains(results, self.runtime)

        self.spike_trains = results
        self.voltage_traces = tidy_Vms(Vms, self.num_neurons) 
        self.spike_recorder = spike_recorder

    def check_simulated(self):
        '''
        Checks if a simulation has been run
        Takes in no parameters and returns a boolean
        '''

        return self.simulated

    def get_spike_trains(self):
        '''
        Gets the spike trains for each neuron produced by the simulation
        Takes in no parameters and returns a 2d numpy array
        '''
        if self.simulated:
            return self.spike_trains
    
    def get_voltage_traces(self):
        '''
        Gets the voltage traces for each neuron produced by the simulation
        Takes in no paramters and returns a 2d numpy array
        '''
        if self.simulated:
            return self.voltage_traces
        
    def show_raster(self):
        '''
        Displays a raster plot based on the spike trains obtained from the simulation
        Takes in no paramters and returns nothing
        '''
        if self.simulated:
            nest.raster_plot.from_device(self.spike_recorder)
        else:
            print("No simulation has been run!")
    
    def show_voltage_trace(self, neuron_id):
        '''
        Displays the voltage trace for a specified neuron
        Takes in an integer and returns nothing
        '''
        if self.simulated:
            fig = plt.figure()
            voltages = self.voltage_traces[neuron_id - 1]
            ts = range(1, self.runtime)
            plt.plot(ts, voltages)
            plt.title(f'Voltage trace for Neuron {neuron_id}')
            plt.xlabel('Frames')
            plt.ylabel('Voltage (V)')
            plt.show()
        else:
            print("No simulation has been run!")
    
#Helper functions for the model class begin here
        
#Converts the Vms recorded from simulation to a nested array of voltage traces for each neuron
def tidy_Vms(Vms, num_neurons):
    '''
    Converts the Vms recorded from the simulation intoto a nested array of voltage traces for each neuron
    Takes in a 1d aray of voltages ([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]), the number of neurons (integer), and outputs a 2d numpy array
    '''
    voltage_traces = []
    for i in range(0, num_neurons):
        voltage_trace = []
        for j in range(i, len(Vms), num_neurons):
            voltage_trace.append(Vms[j])
        voltage_traces.append(voltage_trace)
    return np.array(voltage_traces)

def simulation_results_to_spike_trains(results, runtime):
    '''
    Input is a list of np arrays, each representing the times at which a neuron spiked
    Output is an array of arrays, each of which is a spike train
    '''

    num_neurons = len(results)
    spike_trains = np.zeros((num_neurons, runtime))
    for i in range(num_neurons):
        spike_train = np.zeros(runtime)
        for time in results[i]:
            spike_train[int(time) - 1] = 1.0
        spike_trains[i] = spike_train
    return spike_trains

def connect_weights(A, B, W, G, V):

    '''
    Connects all neurons in groups A and B according to weight matrix W with global scaling of G and voltage of V
    '''
    nest.Connect(A, B, 'all_to_all', syn_spec={'weight': np.transpose(W) * G * V})
        
def set_connection_weights_s1(pyr, ec, ca3, inter, ms, weights, G_e, G_i, V_e, V_i):

    '''
    Sets all connection weightings
    '''
    
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

def initialize_neuron_group(type, n=1, params={}, initial_vm = None):
    neurons = nest.Create(type, n=n, params=params)
    if not initial_vm:
        Vth = neurons.get('V_th')[0]
        Vreset = neurons.get('V_reset')[0]
        neurons.set({"V_m": Vreset + nest.random.uniform(0.0, Vth-Vreset)})
    else:
        neurons.set({"V_m": initial_vm})
    return neurons



    
