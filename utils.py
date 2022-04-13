import os
import numpy as np

def load_data(filename):
    data = np.loadtxt(filename)
    return data
