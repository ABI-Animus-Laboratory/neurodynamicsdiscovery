import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

class NeuronCategorizer:

    place_cells = None
    silent_cells = None
    interneurons = None

    def __init__(self, spike_trains, eztrack_data, different_framerates = True, pf_area = 0.40, acceptance = 0.65, silent_cutoff = 5, interneuron_cutoff = 275,
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

        self.ofield_x_min = self.eztrack_data['X'].min()
        self.ofield_x_max = self.eztrack_data['X'].max()
        self.ofield_y_min = self.eztrack_data['Y'].min()
        self.ofield_y_max = self.eztrack_data['Y'].max()

        self.ofield_w = self.ofield_x_max - self.ofield_x_min
        self.ofield_h = self.ofield_y_max - self.ofield_y_min

        self.spike_coordinates = None
        self.categorized_neurons = None

        self.categorized = False

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
        Returns a two-element tuple
        '''

        return np.average(coords, axis=0)
    
    def categorize_neurons(self):
        
        '''
        Categorizes neurons into silent, interneuron or place cell categories according to their spike coordinates
        Takes in nothing and returns None
        '''

        categorized_neurons = {'Place':{}, 'Silent':{}, 'Interneuron':{}}

        for neuron_id in self.spike_coordinates.keys():
            neuron_spike_coords = self.spike_coordinates[neuron_id]
            label = self.get_neuron_categorization_square_box(neuron_spike_coords, self.ofield_w)
            categorized_neurons[label][neuron_id] = neuron_spike_coords
        
        self.categorized_neurons = categorized_neurons

    def get_neuron_categorization_square_box(self, neuron_spike_coords, ofield_w):

        '''
        Checks whether a neuron is silent, interneuron or place cell using the square box method
        Returns a string indicating the neuron category
        '''

        if len(neuron_spike_coords) < self.silent_cutoff:
            return 'Silent'
        if len(neuron_spike_coords) > self.interneuron_cutoff:
            return 'Interneuron'

        box_radius = (math.sqrt(self.pf_area) * self.ofield_w) / 2
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
        if len(neuron_spike_coords) < self.separation_threshold:
            return 'Silent'
        return 'Interneuron'

    def run_categorization(self):

        '''
        Runs the neuron categorization pipeline
        '''

        self.calculate_spike_coordinates()
        self.categorize_neurons()
        self.categorized = True

    def print_category_counts(self):

        '''
        Prints the number of neurons in each category
        '''

        if self.categorized:
            print('Number of place cells: ' + str(len(self.categorized_neurons['Place'].keys())))
            print('Number of silent cells: ' + str(len(self.categorized_neurons['Silent'].keys())))
            print('Number of interneurons: ' + str(len(self.categorized_neurons['Interneuron'].keys())))
        else:
            print('No neurons have been categorized yet!')

    def plot_place_field_box(self, neuron_spike_coords, neuron_id = None):

        '''
        Uses the array of spike coordinates for a single neuron to create a place field box plot 
        Takes in neuron spike coordinates and returns nothing
        '''
        
        box_radius = math.sqrt(self.ofield_w * self.ofield_h * self.pf_area) / 2

        centroid = self.calculate_centroid(neuron_spike_coords)

        box_top = (centroid[0] - self.ofield_x_min - box_radius) // 2
        box_left = (centroid[1] - self.ofield_y_min - box_radius) // 2

        heat_map = np.zeros((round(self.ofield_w // 2)+1, round(self.ofield_h // 2)+1))

        for coord in neuron_spike_coords:
            x, y = round((coord[0] - self.ofield_x_min) // 2), round((coord[1] - self.ofield_y_min) // 2)
            heat_map[x][y] += 1
        
        fig, ax = plt.subplots()
        cax = ax.imshow(heat_map, 'hot', vmin=0, vmax=5)
        box = Rectangle((box_left, box_top), box_radius, box_radius, linewidth=0.5, edgecolor='yellow', facecolor='none')
        ax.add_patch(box)
        if not neuron_id:
            ax.set_title('Spike coordinates')
        else:
            ax.set_title('Spike coordinates for Neuron ' + neuron_id)
        plt.colorbar(cax)
        plt.show()

    def save_place_fields_box(self):

        '''
        Saves the place field box plots for all neurons
        Takes in nothing and returns nothing
        '''

        box_radius = math.sqrt(self.ofield_w * self.ofield_h * self.pf_area) / 2

        for category in self.categorized_neurons.keys():
            for neuron_id in self.categorized_neurons[category].keys():
                neuron_spike_coords = self.categorized_neurons[category][neuron_id]

                centroid = self.calculate_centroid(neuron_spike_coords)
                box_top = (centroid[0] - self.ofield_x_min - box_radius) // 2
                box_left = (centroid[1] - self.ofield_y_min - box_radius) // 2

                heat_map = np.zeros((round(self.ofield_w // 2)+1, round(self.ofield_h // 2)+1))

                for coord in neuron_spike_coords:
                    x, y = round((coord[0] - self.ofield_x_min) // 2), round((coord[1] - self.ofield_y_min) // 2)
                    heat_map[x][y] += 1
                
                fig, ax = plt.subplots()
                cax = ax.imshow(heat_map, 'hot', vmin=0, vmax=5)
                box = Rectangle((box_left, box_top), box_radius, box_radius, linewidth=0.5, edgecolor='yellow', facecolor='none')
                ax.add_patch(box)
                if not neuron_id:
                    ax.set_title('Spike coordinates')
                else:
                    ax.set_title('Spike coordinates for Neuron ' + neuron_id)
                plt.colorbar(cax)
                if category == 'Place':
                    plt.savefig(f'/hpc/mzhu843/modelling/nest/plots/place field box plots/place/Neuron {neuron_id}.png')
                if category == 'Interneuron':
                    plt.savefig(f'/hpc/mzhu843/modelling/nest/plots/place field box plots/inter/Neuron {neuron_id}.png')
                if category == "Silent":
                    plt.savefig(f'/hpc/mzhu843/modelling/nest/plots/place field box plots/silent/Neuron {neuron_id}.png')
                plt.close(fig)               



        


        





    
    





        
            


    
    


            



        
        

        
        





    