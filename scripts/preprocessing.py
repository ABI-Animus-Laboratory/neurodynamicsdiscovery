import numpy as np
import nest
from scripts import visualizations, initializations, experiments

def preprocess_spike_data(an_array):
    #Removes spike residue and sets spike values to 1
    for i in range(np.size(an_array, axis=0)):
        for j in range(np.size(an_array[i], axis=0) -1, 0, -1):
            if an_array[i][j] != 0 and an_array[i][j-1] != 0:
                an_array[i][j] = 0
            if an_array[i][j] != 0 and an_array[i][j-1] == 0:
                an_array[i][j] = 1
        if an_array[i][0] != 0:
            an_array[i][0] = 1
    return an_array

    

                


