# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:15:11 2020

@author: Saint8312
"""

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need

def tanh(x):                 # Define a function
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

if __name__ == "__main__":
    g = grad(np.cos)       # Obtain its gradient function
    print(g(np.pi/2))