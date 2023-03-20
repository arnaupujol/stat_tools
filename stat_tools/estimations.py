#This method contains methods for estimations such as interpolation or smoothing.

import numpy as np
import pandas as pd

def smooth(dataframe, size = 15, center = True, win_type = 'boxcar'):
    """
    This method returns a smoothed version of a datetime
    data series.

    Parameters:
    -----------
    dataframe: pd.DataFrame
        The datetime data series
    size: int
        Size of the window function used for the smoothing
    center: bool
        If True, the smoothed data is center to the central
        value of the smoothing.
    win_type: str
        Type of window function applied

    Returns:
    --------
    smooth_df: pd.DataFrame
        Smoothed data frame
    """
    smooth_df = dataframe.rolling(size, center = center, win_type = win_type).mean()
    return smooth_df

def convolve_ones(data, kernel_size = None):
    """
    This method convolves the data with a tophat kernel.

    Parameters:
    -----------
    data: np.array
        Array with data values
    kernel_size: int
        Size of kernel

    Returns:
    --------
    data_convolved: np.array
        Convolved data
    """
    if kernel_size is None:
        data_convolved = data
    else:
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved = np.convolve(data, kernel, mode='same')
    return data_convolved
