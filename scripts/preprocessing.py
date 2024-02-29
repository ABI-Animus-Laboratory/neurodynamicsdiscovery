import numpy as np
import nest
from scripts import visualizations

def preprocess_spike_data(spike_trains):
    '''
    Function for preprocessing the raw spike train data
    Removes spike residues by setting consecutive non-leading values to 0 and normalizes all non-zero values to 1
    Input is 2d numpy array
    Output is 2d numpy array
    '''
    
    for i in range(np.size(spike_trains, axis=0)):
        for j in range(np.size(spike_trains[i], axis=0) -1, 0, -1):
            if spike_trains[i][j] != 0 and spike_trains[i][j-1] != 0:
                spike_trains[i][j] = 0
            if spike_trains[i][j] != 0 and spike_trains[i][j-1] == 0:
                spike_trains[i][j] = 1
        if spike_trains[i][0] != 0:
            spike_trains[i][0] = 1
    return spike_trains
    

                


