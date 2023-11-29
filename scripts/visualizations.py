import matplotlib.pyplot as plt

#Plots membrane potentials over time for specified neuron ids from a multimeter
def plot_vms_from_device(device, id_list):
    plt.figure(figsize=(18, 5))
    plt.title(f'Membrane potential(s) for Neuron(s) {id_list}')
    for id in id_list:
        ts = device.get('events')['times'][::id]
        vms = device.get('events')['V_m'][::id]
        plt.plot(ts, vms)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')