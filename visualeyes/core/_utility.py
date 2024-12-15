import pandas as pd
import numpy as np
import numbers

def aoi_mask_validation(aoi_mask, screen_dimension):
    
    '''
    Validate the Area of Interest (AOI) mask
    
    Parameters:
    -----------
    aoi_mask : numpy.ndarray
        Binary mask of the AOI (shape: height x width)
    screen_dimension : tuple
        Screen dimension (width, height)
        
    Returns:
    --------
    None
    
    '''
    
    # check if aoi_mask is a numpy array
    if not isinstance(aoi_mask, np.ndarray):
        raise ValueError('AOI mask should be a numpy array')
    
    # check if aoi_mask is a 2D array
    if len(aoi_mask.shape) != 2:
        raise ValueError('AOI mask should be a 2D array')
    
    # check if aoi_mask is a binary mask
    if not np.all(np.isin(aoi_mask, [0, 1])):
        raise ValueError('AOI mask should be a binary mask')
    
    # check if aoi_mask has the same shape as the screen dimension
    if aoi_mask.shape != screen_dimension:
        raise ValueError('AOI mask should have the same shape as the screen dimension')

    return None


def dataframe_validation(df, screen_dimensions=None, drop_outlier=False, drop_nan=True):
    '''
    Validate the input dataframe and return the x and y coordinates if the dataframe is valid.
    Optionally, return the indices of the data points outside the screen boundaries if screen_dimensions is provided.
    Drop the data points outside the screen boundaries if drop_outlier is also True.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    screen_dimensions : tuple, list, or numpy array, optional
        Screen dimensions (height, width)
    drop_outlier : bool, optional
        Drop the data points outside the screen boundaries
        
    Returns:
    --------
    (x_coord, y_coord) : tuple
        Tuple containing the x and y coordinates of the data points
    outlier_indices : numpy.ndarray
        Indices of the data points outside the screen boundaries
    df : pd.DataFrame
        Updated dataframe with outliers dropped (if drop_outlier is True)
    '''
    # Check input type
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input data should be a pandas DataFrame')

    # Validate coordinate columns
    if set(['axp', 'ayp']).issubset(df.columns):

        x_name, y_name = 'axp', 'ayp'
            
    elif set(['xpos', 'ypos']).issubset(df.columns):
        
        x_name, y_name = 'xpos', 'ypos'
            
    else:
        raise ValueError('Missing x and y coordinates')
    
    # Check for mismatched x and y coordinates
    if not df[x_name].shape == df[y_name].shape:
        raise ValueError('Mismatched x and y coordinates')
    
    # Drop NaN values
    if drop_nan:
        df = df.dropna(subset=[x_name, y_name])
    
    # Assign x and y coordinates
    x_coord, y_coord = df[x_name], df[y_name]
   
    # Initialize outlier_indices
    outlier_indices = np.array([], dtype=int)

    # Validate screen dimensions and check for outliers
    if screen_dimensions:
        screen_dimensions_validation(screen_dimensions)
        screen_height, screen_width = screen_dimensions

        # Find outlier indices based on actual DataFrame index
        outlier_mask = (x_coord < 0) | (x_coord >= screen_width) | \
                       (y_coord < 0) | (y_coord >= screen_height)
        
        outlier_indices = df.index[outlier_mask]

        if drop_outlier:
            
            # Drop rows and reassign coordinates
            df = df.drop(index=outlier_indices)
            x_coord = df[x_name] 
            y_coord = df[y_name]

    # Return results
    if drop_outlier:
        return (x_coord.values, y_coord.values), outlier_indices, df
    
    return (x_coord.values, y_coord.values), outlier_indices

def aoi_definitions_validation(aoi_definitions, screen_dimensions):
    """
    Validate the input AOI definitions
    
    Parameters:
    -----------
    aoi_definitions : dict, list of dict, or numpy.ndarray of dict
        The AOI definitions to overlay on the plot. Each dict should contain:
        - 'shape': 'rectangle' or 'circle'.
        - 'coordinates': tuple, list or np.ndarray of coordinates.
            - for rectangluar AOI's: (x1, x2, y1, y2), upper-bounds non-inclusive. 
            - for circlular AOI's: (x_center, y_center, radius).
    screen_dimensions : tuple
        The dimensions of the screen in pixels (height, width).
        
    Returns:
    --------
    None
    """
    # check if aoi_definitions is not empty
    if not aoi_definitions:
        raise ValueError('AOI definitions cannot be empty')

    # check if aoi_definitions is a dictionary, list, or numpy array
    if not isinstance(aoi_definitions, (dict, list, np.ndarray)):
        raise ValueError('AOI definitions should be a dictionary or a list of dictionaries')
    
    # check if aoi_definitions is a list of dictionaries
    if isinstance(aoi_definitions, (list, np.ndarray)):
        if any([not isinstance(aoi, dict) for aoi in aoi_definitions]):
            raise ValueError('AOI definitions should be a list of dictionaries')
    
    # check if aoi_definitions is a dictionary
    if isinstance(aoi_definitions, dict):
        # convert the dictionary to a list
        aoi_definitions = [aoi_definitions]
    
    for aoi in aoi_definitions:
        
        # check if the required keys are present in the dictionary
        required_keys = ['shape', 'coordinates']
        missing_keys = [key for key in required_keys if key not in aoi.keys()]
        if missing_keys:
            raise KeyError(f'Missing keys in AOI definition: {missing_keys}')
        
        # check if the 'shape' key has a valid value
        if aoi['shape'] not in ['rectangle', 'circle']:
            raise ValueError(f"Unsupported AOI shape: {aoi['shape']}")
        
        # check if the 'coordinates' key corresponds to a tuple, list, or numpy array
        if not isinstance(aoi['coordinates'], (tuple, list, np.ndarray)):
            raise ValueError('AOI coordinates should be a tuple, list, or numpy array')
        
        # check if the 'coordinates' key has the correct number of elements
        if aoi['shape'] == 'rectangle' and len(aoi['coordinates']) != 4:
            raise ValueError('Rectangle coordinates should have four elements')
        
        elif aoi['shape'] == 'circle' and len(aoi['coordinates']) != 3:
            raise ValueError('Circle coordinates should have three elements')
        
        # check if the 'coordinates' key has valid values (non-negative integers)
        if not all(isinstance(coord, (numbers.Integral)) for coord in aoi['coordinates']):
            raise ValueError('All coordinates must be integers')
        
        if any(coord < 0 for coord in aoi['coordinates']):
            raise ValueError('Coordinates cannot be negative')
        
        # check if the 'coordinates' key is within the screen boundaries
        screen_height, screen_width = screen_dimensions

        if aoi['shape'] == 'rectangle':
            x1, x2, y1, y2 = aoi['coordinates']
            
            if x1 > x2:
                raise ValueError('x1 cannot be greater than x2')
            if y1 > y2:
                raise ValueError('y1 cannot be greater than y2')
            
            if x2 > screen_width or y2 > screen_height:
                raise ValueError('AOI exceeds screen boundaries')
                    
        elif aoi['shape'] == 'circle':
            x_center, y_center, radius = aoi['coordinates']
    
            if (x_center - radius < 0 or x_center + radius > screen_width or
                y_center - radius < 0 or y_center + radius > screen_height):
                raise ValueError('AOI exceeds screen boundaries')
        
    return None

def screen_dimensions_validation(screen_dimensions):
    
    '''
    Validate the screen dimensions
    
    Parameters:
    -----------
    screen_dimensions : tuple, list, or numpy.ndarray
        Screen dimensions (height, width)
        
    Returns:
    --------
    None
    '''
    
    # check if screen_dimensions is a tuple
    if not isinstance(screen_dimensions, (tuple, list, np.ndarray)):
        raise ValueError('Screen dimensions should be a tuple, list, or numpy array')
    
    # check if screen_dimensions has two elements
    if len(screen_dimensions) != 2:
        raise ValueError('Screen dimensions should have two elements.')
    
    # check if screen_dimensions has positive integer values
    if not all(isinstance(dim, numbers.Integral) for dim in screen_dimensions):
        raise ValueError('Screen dimensions should have positive integer values.')
    
    if any(dim <= 0 for dim in screen_dimensions):
        raise ValueError('Screen dimensions should have positive integer values.')
    
    return None
