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

def plot_spikes_from_device(device, title='Spike timings'):
    spike_events = device.get('events')
    spikes = spike_events['senders']
    spike_times = spike_events['times']
    plt.title(title)
    plt.plot(spike_times, spikes, '.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron id')
    plt.xlim(xmin=-8)

def plot_matrix(matrix, cmap='cividis', colorbar=True):
    plt.figure(figsize=(7, 5))
    img = plt.imshow(matrix, cmap=cmap, aspect='auto')
    if colorbar:
        plt.colorbar(img)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()
