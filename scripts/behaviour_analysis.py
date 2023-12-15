import pandas as pd

def assign_quadrant(row, x_mid, y_mid):
    if row['X'] < x_mid and row['Y'] < y_mid:
        return 'Top Left'
    elif row['X'] > x_mid and row['Y'] < y_mid:
        return 'Top Right'
    elif row['X'] < x_mid and row['Y'] > y_mid:
        return 'Bottom Left'
    else:
        return 'Bottom Right'

