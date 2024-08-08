import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from io import BytesIO

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def make_window(X, w_length):
    """Create a window of a specific length

    Parameters:
    - X : (numpy.ndarray)   Univariate time series
    - w_length              Number of observation to be consider in each window.
                            It will depend on the horizon of the forecast
    
    Returns:
    - numpy.ndarray where each entry is a window
    """
    windows_array = np.zeros(())
    single_window = False
    if w_length == 1:
        return X
    else:
        
