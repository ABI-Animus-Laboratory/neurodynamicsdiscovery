import numpy as np
import matplotlib.pyplot as plt


def preprocess_spike_data(self, spike_trains):
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

def scale_calcium_data(calcium_traces, spike_trains, V_t, V_reset):

    calcium_traces, spike_trains = np.array(calcium_traces), np.array(spike_trains)

    for calcium in calcium_traces:

        og_calcium = calcium * 1

        f_max = calcium * 1

        index = np.where((calcium_traces == calcium).all(axis=1))[0][0]

        ts = np.nonzero(spike_trains[index])[0]

        ###Test something
        # new_ts = []
        # for t in ts:
        #     current = calcium[t]
        #     while current < max(calcium[t:t+15]):
        #         current = calcium[t+1] 
        #         t += 1
        #     new_ts.append(t)
        
        # print(new_ts)
        ###Test something

        new_ts = []
        for t in ts:

            current = calcium[t]

            if current > calcium[t-1] and current > calcium[t+1]:
                new_ts.append(t)
                
            elif current < calcium[t-1] and current > calcium[t+1]:
                while t + 1 < np.size(calcium) and current > calcium[t+1]:
                    current = calcium[t+1]
                    t += 1

                while t + 1 < np.size(calcium) and current < calcium[t+1]:
                    current = calcium[t+1]
                    t +=1
                new_ts.append(t)
            
            else:
                while current < calcium[t+1]:
                    current = calcium[t+1]
                    t +=1
                new_ts.append(t)

        mask = np.zeros_like(f_max, dtype=bool)
        mask[new_ts] = True
        f_max[~mask] = 0

        non_zero_indices = np.where(f_max != 0)[0]
        non_zero_indices = np.insert(non_zero_indices, 0, 0)
        for i in range(np.size(non_zero_indices)-1):
            start = f_max[non_zero_indices[i]]
            end = f_max[non_zero_indices[i + 1]]
            step_size = (end - start) / (non_zero_indices[i + 1] - non_zero_indices[i])
            for j in range(non_zero_indices[i]+1, non_zero_indices[i + 1]):
                f_max[j] = f_max[j-1] + step_size

        first_index = new_ts[0]
        first_value = f_max[first_index]
        for i in range(first_index):
            f_max[i] = first_value

        last_index = new_ts[-1]
        last_value = f_max[last_index]
        for i in range(last_index + 1, np.size(calcium)):
            f_max[i] = last_value
        
        calcium =  V_reset + (calcium / f_max) * (V_t - V_reset)

        outliers = np.where(calcium > -50)
        for j in outliers:
            calcium[j] = -50

        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        axs[1].plot(f_max)
        axs[1].set_title('Fmax')
        axs[2].plot(calcium)
        axs[2].set_title('Nosrmalized Calcium')
        axs[0].plot(og_calcium)
        axs[0].set_title('Original Calcium')
        fig.suptitle(f'Neuron {index+1}', fontsize=20)

        plt.savefig(f'/hpc/mzhu843/modelling/nest/plots/normalized calcium/Neuron {index+1}')
        plt.close()
        
    return calcium
    
    



    

                


