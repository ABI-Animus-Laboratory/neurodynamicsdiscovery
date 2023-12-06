import numpy as np

def ssd_with_l1(m1, m2, lamb):
    '''
    Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
    '''
    squared_difference = (m1 - m2) ** 2
    l1_penalty = np.sum(lamb * m1)
    return np.sum(squared_difference) + l1_penalty
