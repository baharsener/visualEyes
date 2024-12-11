import numpy as np
import pandas as pd
import numbers
from .utility import aoi_mask_validation, dataframe_validation

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
    if isinstance(window_start, (numbers.Integral, numbers.Real)) and not isinstance(window_duration, (numbers.Integral, numbers.Real)):
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
        # inclusive of the start time and exclusive of the end time
        data = eye_data[(eye_data['time'] >= start_time) & (eye_data['time'] < end_time)] 
        
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

def define_aoi(screen_dimensions, aoi_definitions):
    """
    Define Areas of Interest (AOIs).
    
    Parameters:
    -----------
    screen_dimensions : tuple, list, or np.array
        Screen dimensions (height, width).
        
    aoi_definitions : dict or a list of dict
        Each dictionary defines one AOI (so we can have multiple) with keys:
        - 'shape': 'rectangle' or 'circle'.
        - 'coordinates': Tuple of coordinates:
            - For rectangluar AOI's: (x1, x2, y1, y2), upper-bounds non-inclusive. 
            - For circlular AOI's: (x_center, y_center, radius).
        
    Returns:
    --------
    mask : 2D numpy array
        Binary mask of the AOIs
    """
    
    # Check if screen_dimensions is a tuple, list, or numpy array
    if not isinstance(screen_dimensions, (tuple, list, np.ndarray)):
        raise ValueError("Screen dimensions must be a tuple, list, or numpy array.")
    
    # Check if screen_dimensions has two elements
    if len(screen_dimensions) != 2:
        raise ValueError("Screen dimensions must have two elements.")
    
    # Check if screen_dimensions are positive integers    
    if not all((isinstance(dim, (int, np.integer)) and dim > 0) for dim in screen_dimensions):
        raise ValueError("Screen dimensions must be positive integers.")

    # check if aoi_definitions is a dictionary, a list of dictionaries, or a numpy array of dictionaries
    if not isinstance(aoi_definitions, (dict, list, np.ndarray)):
        raise ValueError("AOI definitions must be a dictionary, a list of dictionaries, or a numpy array of dictionaries.")
    
    # check if the list or numpy array actually contains dictionaries
    if isinstance(aoi_definitions, (list, np.ndarray)):
        if not all(isinstance(aoi, dict) for aoi in aoi_definitions):
            raise ValueError("AOI definitions must be a list of dictionaries or a numpy array of dictionaries.")
    
    # if aoi_definitions is a dict, convert it to a list
    if isinstance(aoi_definitions, dict):
        aoi_definitions = [aoi_definitions]
        
    # check if all dictionaries in aoi_definitions have the keys 'shape' and 'coordinates'
    required_keys = {'shape', 'coordinates'}
        
    for aoi in aoi_definitions:
        if not required_keys.issubset(aoi.keys()):
            raise KeyError(f"AOI definitions must include {required_keys}")

    # Initialize a mask
    # mask is a 2D numpy array with the same dimensions as the screen
    screen_height, screen_width = screen_dimensions
    mask = np.zeros((screen_height, screen_width), dtype=np.uint8)

    # Define each AOI
    for aoi in aoi_definitions:
        shape = aoi['shape'].lower()
        coordinates = aoi['coordinates']
        
        if shape == 'rectangle':
            
            # Check if cordinates is a tuple, list, or numpy array that contains four elements
            if not isinstance(coordinates, (tuple, list, np.ndarray)) or len(coordinates) != 4:
                raise ValueError("Rectangle coordinates must be a tuple, list, or numpy array with four elements.")
            
            # Check if coordinates include four positive integers
            if not all((isinstance(coord, numbers.Integral) and coord > 0) for coord in coordinates):
                raise ValueError("Rectangle coordinates must be four positive integers.")
            
            x1, x2, y1, y2 = map(int, coordinates) # Convert input coordinates to integers
            
            # Check if the AOI is within the screen boundaries (non-inclusive)
            # If not, raise an error
            if not (0 <= x1 < x2 <= screen_width and 0 <= y1 < y2 <= screen_height):
                raise ValueError(f"Coordinates exceed screen boundaries or are invalid (e.g., x1 >= x2 or y1 >= y2).")

            mask[y1:y2, x1:x2] = 1  # All pixels within the rectangle are 1
            
        elif shape == 'circle':
            
             # Check if cordinates is a tuple, list, or numpy array that contains three elements
            if not isinstance(coordinates, (tuple, list, np.ndarray)) or len(coordinates) != 3:
                raise ValueError("Circle coordinates must be a tuple, list, or numpy array with three elements.")
            
            if not all((isinstance(coord, numbers.Integral) and coord > 0) for coord in coordinates):
                raise ValueError("Circle coordinates must be three positive integers.")
            
            x_center, y_center, radius = map(int, coordinates) # Convert input coordinates to integers
            
            # Check if the AOI is within the screen boundaries
            if ((x_center - radius) < 0) or ((y_center - radius) < 0) or ((x_center + radius > screen_width)) or ((y_center + radius) > screen_height):
                raise ValueError(f"Circle exceeds screen boundaries.")
            
            # # Loop through all pixels in the screen
            # for y in range(screen_height):
            #     for x in range(screen_width):
            #         # Use the equation of a circle to check if (x, y) is inside
            #         if (x - x_center)**2 + (y - y_center)**2 <= radius**2:
            #             mask[y, x] = 1 #All pixels within the circle are 1
                        
            # More efficiently, we can use numpy to create the mask (faster than looping)
            y, x = np.ogrid[:screen_height, :screen_width] # Create a grid of x and y coordinates
            mask[(x - x_center)**2 + (y - y_center)**2 <= radius**2] = 1 # Set all pixels within the circle to 1

        else:
            raise ValueError(f"Unsupported AOI shape: {shape}")
    
    return mask

def percent_data_in_aoi(df, aoi_mask, screen_dimension):
    
    """
    Calculate the percentage of data points in the AOI.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the x and y coordinates of the data points.
    aoi_mask : 2D np.array
        Binary mask of the AOI.
    screen_dimension : tuple
        Screen dimension (width, height).
    
    Returns:
    --------
    percent_in_aoi : float
        Percentage of data points in the AOI.
    """
    
    # validate aoi_mask
    aoi_mask_validation(aoi_mask, screen_dimension)
    
    # check if df contains valid data
    coords = dataframe_validation(df)
    
    x_coord = coords[0]
    y_coord = coords[1]
    
    # count the number of data points in the AOI
    in_aoi = aoi_mask[y_coord, x_coord] == 1
    num_data_in_aoi = np.sum(in_aoi)
 
    # calculate the percentage of data points in the AOI
    percent_in_aoi = num_data_in_aoi/ len(x_coord) * 100
    
    return percent_in_aoi