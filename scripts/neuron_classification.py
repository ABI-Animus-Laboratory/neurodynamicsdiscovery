import pandas as pd
import numpy as np
import math

def match_frames(spikes_array, X_coords, Y_coords, num_frames):
    '''
    Fixes issue where the frame rate of the calcium imaging is 30fps, twice that of the 15fps behavioural recording
    '''
    X_coords = np.vstack((X_coords, X_coords)).reshape(-1, order='F')[:num_frames]
    Y_coords = np.vstack((Y_coords, Y_coords)).reshape(-1, order='F')[:num_frames]
    return X_coords, Y_coords

def get_spike_coords(spikes_array, X_coords, Y_coords):
    '''
    Given an array of spike trains and the EzTrack behvaioural data, returns an array of spike coordinates for each neuron
    '''
    num_neurons = np.size(spikes_array, axis=0)
    num_frames = np.size(spikes_array, axis=1)
    spike_coords = []
    for i in range(num_neurons):
        coords = []
        for j in range(num_frames):
            if spikes_array[i][j] == 1:
                coords.append((X_coords[j], Y_coords[j]))
        spike_coords.append(coords)
    return spike_coords

def get_centroid(coords, lower_percentile=None, upper_percentile=None):
    #Returns the centre coordinate from an array of coordinates
    coords = np.array(coords)
    if not lower_percentile:
        lower_percentile = np.percentile(coords, 0)
    if not upper_percentile:
        upper_percentile = np.percentile(coords, 100)
    middle_percentile = coords[(coords >= lower_percentile) & (coords <= upper_percentile)]
    middle_percentile = middle_percentile.reshape(-1, 2)
    return np.average(middle_percentile, axis=0)

def categorize_neurons_box(spikes_coords, x_min, x_max, pf_area=0.5, acceptance=0.7):
    #Categorizes neurons into interneurons, silent cells and place cells using box method
    neuron_types = {'Interneuron':[], 'Silent':[], 'Place':[]}
    for i in range(len(spikes_coords)-1, -1, -1):
        if len(spikes_coords[i]) < 5:
            coords = spikes_coords.pop(i)
            neuron_types['Silent'].append([i+1, coords])
        if len(spikes_coords[i]) > 275:
            coords = spikes_coords.pop(i)
            neuron_types['Interneuron'].append([i+1, coord])

    field_w = x_max - x_min
    pf_radius = (math.sqrt(pf_area) * field_w) / 2
    for index, neuron_spikes in enumerate(spikes_coords):
        centroid = get_centroid(neuron_spikes)
        box_top, box_bottom = centroid[0] + pf_radius, centroid[0] - pf_radius
        box_left, box_right = centroid[1] + pf_radius, centroid[1] - pf_radius
        count = 0
        for coord in neuron_spikes:
            x, y = coord[0], coord[1]
            if x <= box_top and x >= box_bottom and y <= box_left and y >= box_right:
                count += 1
        if count >= acceptance * len(neuron_spikes):
            neuron_types['Place'].append([index+1, neuron_spikes])
        else:
            if len(neuron_spikes) < 50:
                neuron_types['Silent'].append([index+1, neuron_spikes])
            else:
                neuron_types['Interneuron'].append([index+1, neuron_spikes])
    return neuron_types

    

