import eyelinkio
import pandas as pd
import numpy as np

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


def dataframe_validation(df):
        
    '''
    Validate the input dataframe and return the x and y coordinates if the dataframe is valid
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    x_coord : numpy.ndarray
        x coordinates
    y_coord : numpy.ndarray
        y coordinates
    '''
    x_coord = None
    y_coord = None
    
    # check if input data is a pd.DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError('input data should be a pandas dataframe')
    
    # check if the dataframe contains either 'axp' and 'ayp' or 'xpos' and 'ypos' columns
    if not (('axp' in df.columns and 'ayp' in df.columns) or ('xpos' in df.columns and 'ypos' in df.columns)):
        raise ValueError('missing x and y coordinates')
    
    else:
        # pull the x and y coordinates
        if 'axp' in df.columns and 'ayp' in df.columns:
            x_coord = df['axp']
            y_coord = df['ayp']
        else:
            x_coord = df['xpos']
            y_coord = df['ypos']
            
    return (x_coord.values, y_coord.values)