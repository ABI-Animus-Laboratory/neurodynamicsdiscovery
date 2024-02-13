import pandas as pd 
import numpy as np
import math

class NeuronCategorizer:

    place_cells = None
    silent_cells = None
    interneurons = None

    def __init__(self, spike_trains, eztrack_data, different_framerates = True, pf_area = 0.38, acceptance = 0.65, silent_cutoff = 5, interneuron_cutoff = 275,
                 separation_threshold = 75):
        
        self.spike_trains = spike_trains
        self.eztrack_data = eztrack_data
        self.num_calcium_frames = len(spike_trains[0])
        self.num_neurons = len(spike_trains)
        self.different_framerates = different_framerates
        self.pf_area = pf_area
        self.acceptance = acceptance
        self.silent_cutoff = silent_cutoff
        self.interneuron_cutoff = interneuron_cutoff
        self.separation_threshold = separation_threshold

        self.spike_coordinates = None
        self.categorized_neurons = None

    def get_spike_coordinates(self):
        '''
        Gets the coordinates for all spikes for each neuron
        Takes in no parameters and returns a 2d numpy array
        '''
        return self.spike_coordinates
    
    def get_categorized_neurons(self):
        '''
        Takes in no parameters and returns a dictionary where the keys are neuron types and the values are neuron ids with respective spike coordinates
        '''
        return self.categorized_neurons

    def double_frames(self, X_coords, Y_coords):
        '''
        This method is called if self.differentframerate is set to True
        Doubles the number of x and y coordinates for the behavioural data, then slices off coordinates from the end to match the number of frames in the calcium imaging
        Returns two numpy arrays of x and y coordinates
        '''
        return np.vstack((X_coords, X_coords)).reshape(-1, order='F')[:self.num_calcium_frames], np.vstack((Y_coords, Y_coords)).reshape(-1, order='F')[:self.num_calcium_frames]

    def calculate_spike_coordinates(self):
        '''
        Produces the coordinates at which each neuron spiked
        Returns a dictionary where the keys are neuron ids and the values are coordinates at which that neuron spiked
        '''

        x_coords_behaviour = self.eztrack_data['X'].values
        y_coords_behaviour = self.eztrack_data['Y'].values

        if self.different_framerates:
            x_coords_behaviour, y_coords_behaviour = self.double_frames(x_coords_behaviour, y_coords_behaviour)
        
        spike_coordinates = {}

        for i in range(self.num_neurons):
            neuron_spike_coords = []
            for j in range(self.num_calcium_frames):
                if self.spike_trains[i][j] == 1:
                    neuron_spike_coords.append((x_coords_behaviour[j], y_coords_behaviour[j]))
            spike_coordinates[f'{i+1}'] = neuron_spike_coords
        self.spike_coordinates = spike_coordinates

    def calculate_centroid(self, coords):
        '''
        Calculates the arithmetic mean coordinate for an array of coordinates
        MIGHT BE INCORRECT NEED TO DOUBLE CHECK
        Returns a two-element tuple
        '''

        return np.average(coords, axis=0)
    
    def categorize_neurons(self):
        
        '''
        Categorizes neurons into silent, interneuron or place cell categories according to their spike coordinates
        Takes in nothing and returns None
        '''

        categorized_neurons = {'Place':[], 'Silent':[], 'Interneuron':[]}

        x_min, x_max = self.eztrack_data['X'].min(), self.eztrack_data['X'].max()

        ofield_w = x_max - x_min

        for neuron_id in self.spike_coordinates.keys():
            neuron_spike_coords = self.spike_coordinates[neuron_id]
            label = self.get_neuron_categorization_square_box(neuron_spike_coords, ofield_w)
            categorized_neurons[label].append({neuron_id: neuron_spike_coords})
        
        self.categorized_neurons = categorized_neurons

    def get_neuron_categorization_square_box(self, neuron_spike_coords, ofield_w):

        '''
        Checks whether a neuron is silent, interneuron or place cell using the square box method
        Returns a string 
        '''

        if len(neuron_spike_coords) < self.silent_cutoff:
            return 'Silent'
        if len(neuron_spike_coords) > self.interneuron_cutoff:
            return 'Interneuron'

        box_radius = (math.sqrt(self.pf_area) * ofield_w) / 2
        centroid = self.calculate_centroid(neuron_spike_coords)

        #The origin of the box is at the top left
        box_left, box_right = centroid[0] + box_radius, centroid[0] - box_radius
        box_top, box_bottom = centroid[1] - box_radius, centroid[1] + box_radius

        count = 0
        for coord in neuron_spike_coords:
            x, y = coord[0], coord[1]
            if x <= box_left and x >= box_right and y >= box_top and y <= box_bottom:
                count += 1
        
        if count >= self.acceptance * len(neuron_spike_coords):
            return 'Place'
        if count < self.separation_threshold:
            return 'Silent'
        return 'Interneuron'

    def run_categorization(self):

        '''
        Runs the neuron categorization pipeline
        '''

        self.calculate_spike_coordinates()
        self.categorize_neurons()
    
    





        
            


    
    


            



        
        

        
        





    