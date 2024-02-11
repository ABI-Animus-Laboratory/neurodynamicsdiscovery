import pandas as pd
import numpy as np

def match_frames(spikes_array, X_coords, Y_coords, num_frames):
    #Matches the number of frames in the x and y coordinates from EzTrack and the spikes data
    #Calcium imaging in 30fps and behavioural data in 15fps assumption
    X_coords = np.vstack((X_coords, X_coords)).reshape(-1, order='F')[:num_frames]
    Y_coords = np.vstack((Y_coords, Y_coords)).reshape(-1, order='F')[:num_frames]
    return X_coords, Y_coords

def get_spike_coords(spikes_array, ez_df):
    #Gives an array of all the spike-time location coordinates for each neuron
    num_neurons = np.size(spikes_array, axis=0)
    num_frames = np.size(spikes_array, axis=1)
    X_coords, Y_coords = ez_df['X'].values, ez_df['Y'].values
    X_coords, Y_coords = match_frames(spikes_array, X_coords, Y_coords, num_frames)
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
        lower_percentile = np.percentile(coords, 30)
    if not upper_percentile:
        upper_percentile = np.percentile(coords, 70)
    middle_percentile = coords[(coords >= lower_percentile) & (coords <= upper_percentile)]
    middle_percentile = middle_percentile.reshape(-1, 2)
    return np.average(middle_percentile, axis=0)

def categorize_neurons_box(spikes_coords, x_min, x_max, y_min, y_max, place_field_percentage=0.30, acceptance_percentage=0.90):
    #Categorizes neurons into interneurons, silent cells and place cells
    neuron_type_counts = {'Interneuron':[], 'Silent':[], 'Place':[]}
    for i in range(len(spikes_coords)-1, -1, -1):
        if len(spikes_coords[i]) < 5:
            coord = spikes_coords.pop(i)
            neuron_type_counts['Silent'].append([i, coord])
        if len(spikes_coords[i]) > 275:
            coord = spikes_coords.pop(i)
            neuron_type_counts['Interneuron'].append([i, coord])

    open_w = x_max - x_min
    open_h = y_max - y_min
    p_field_half_w = place_field_percentage * open_w * 2
    p_field_half_h = place_field_percentage * open_h * 2
    for index, neuron_spikes in enumerate(spikes_coords):
        centroid = get_centroid(neuron_spikes)
        box_w_upper, box_w_lower = centroid[0] + p_field_half_w, centroid[0] - p_field_half_w
        box_h_upper, box_h_lower = centroid[1] + p_field_half_h, centroid[1] - p_field_half_h
        count = 0
        for coord in neuron_spikes:
            x, y = coord[0], coord[1]
            if x <= box_w_upper and x >= box_w_lower and y <= box_h_upper and y >= box_h_lower:
                count += 1
        if count >= acceptance_percentage * len(neuron_spikes):
            neuron_type_counts['Place'].append([index, coord])
        else:
            if len(neuron_spikes) < 60:
                neuron_type_counts['Silent'].append([index, coord])
            else:
                neuron_type_counts['Interneuron'].append([index, coord])
    return neuron_type_counts

    

