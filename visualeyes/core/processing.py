import numpy as np
import pandas as pd
import numbers

def epoch_data(eye_data, window_start, window_duration):
    '''
    Create epochs of data based on given window size
    
    Parameters:
    -----------
    eye_data : pd.DataFrame
        data from eyelinkio output to be epoched
    window_start : int/float or list of int/float
        start of the window(s)
    window_duration : int/float, or a list of int/float
        duration of the window(s)
        
    Returns:
    --------
    epochs : list
        list of epochs
    epoch_data : pd.DataFrame
        data epoched based on the given window size
    '''
    
    # check if data is a dataframe
    if not isinstance(eye_data, pd.DataFrame):
        raise ValueError('data should be a pandas dataframe')
    
    # check if eye_data contains time column
    if 'time' not in eye_data.columns:
        raise ValueError('data should contain a time column')
    
    # window_start must be either a list, a numpy array, or a single value
    if not isinstance(window_start, (list, np.ndarray, numbers.Integral, numbers.Real)):
        raise ValueError('window_start should be a list, a numpy array, or a single value')
    
    # window_duration must be either a list, a numpy array, or a single value
    if not isinstance(window_duration, (list, np.ndarray, numbers.Integral, numbers.Real)):
        raise ValueError('window_duration should be a list, a numpy array, or a single value')
    
    # if window_start is a single value, window_duration should also be a single value
    if isinstance(window_start, (numbers.Integral, numbers.Real)) and not isinstance(numbers.Integral, numbers.Real):
        raise ValueError('if window_start is a single value, window_duration should also be a single value')
    
    # if window_start is a list or numpy array, window_duration can be a list, a numpy array, or a single value
    # if window_duration is a list or numpy array, it should have the same length as window_start
    if isinstance(window_start, (list, np.ndarray)):
        if isinstance(window_duration, (list, np.ndarray)):
            if len(window_duration) != len(window_start):
                raise ValueError('window_duration should have the same length as window_start')
    
    # sort the start of windows ascendingly if it's a list
    if isinstance(window_start, (list, np.ndarray)):
        window_start = sorted(window_start) 
        
    # correspondingly, sort the duration of windows if it's a list
    if isinstance(window_duration, list):
        window_duration = [x for _, x in sorted(zip(window_start, window_duration))]
        
    # if window_duration is a list, it should have the same length as window_start
    if isinstance(window_duration, list):
        if len(window_duration) != len(window_start):
            raise ValueError('window_duration should have the same length as window_start')
        
    # convert window_start and window_duration to list if they are not
    if not isinstance(window_start, list):
        window_start = [window_start]

    # the end of the last window should be equal or less than the last time in the data
    if isinstance(window_duration, (list, np.ndarray)):
        if window_start[-1] + window_duration[-1] > eye_data['time'].iloc[-1]:
            raise ValueError('the end of the last window should be equal or less than the last time in the data')
    elif isinstance(window_duration, (numbers.Integral, numbers.Real)):
        if window_start[-1] + window_duration > eye_data['time'].iloc[-1]:
            raise ValueError('the end of the last window should be equal or less than the last time in the data')
        
    # create epochs of data
    epochs = []
    epoch_index = []
    epoch_data = []
    
    for start in range(len(window_start)):
        # get the start and end of the window
        start_time = window_start[start]
        
        if isinstance(window_duration, list):
            end_time = window_start[start] + window_duration[start]
        else:
            end_time = window_start[start] + window_duration
            
        # get data within the window
        # TODO: inclusive or exclusive of the start and end time?
        data = eye_data[(eye_data['time'] >= start_time) & (eye_data['time'] <= end_time)] 
        
        # append the data to the list
        epoch_data.append(data)
        
        # append the start and end of the window to the list
        epochs.append((start_time, end_time))
        
        num_samples = len(data)
        epoch_index.append([start]*num_samples)
        
    # convert epoch_data to a single dataframe, including all the original columns
    # and add a column for the epoch number, and the start and end of the epoch
    epoch_data = pd.concat(epoch_data)
    
    # epoch_index contains sublists, convert it to a single list
    epoch_index = [item for sublist in epoch_index for item in sublist]
    
    # check if epoch_index has the same length as epoch_data
    if len(epoch_index) != len(epoch_data):
        raise ValueError('epoch_index should have the same length as epoch_data')
    
    # check if unique epoch_index is the same as the number of epochs
    if len(set(epoch_index)) != len(epochs):
        raise ValueError('epoch_index should have the same length as epochs')
    
    # add epoch_index to the epoch_data
    epoch_data['epoch_index'] = epoch_index
    
    return epochs, epoch_data

def define_aoi(aoi_definitions):
    """
    Define Areas of Interest (AOIs).
    """
    aoi_boundaries = []

    for aoi in aoi_definitions:
        shape = aoi['shape'].lower()
        coordinates = aoi['coordinates']
        
        if shape == 'rectangle':
            x1, x2, y1, y2 = map(int, coordinates)
            # Add the rectangle's boundary as a list of (x, y) coordinates
            aoi_boundaries.append([(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)])
        elif shape == 'circle':
            x_center, y_center, radius = map(int, coordinates)
            # For circle, plot an outline by calculating points around the perimeter
            theta = np.linspace(0, 2 * np.pi, 100)
            x = x_center + radius * np.cos(theta)
            y = y_center + radius * np.sin(theta)
            aoi_boundaries.append(list(zip(x, y)))
        else:
            raise ValueError(f"Unsupported AOI shape: {shape}")
    
    return aoi_boundaries