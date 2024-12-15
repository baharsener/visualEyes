"""Test the define_aoi function."""

import os
from visualeyes import define_aoi
import eyelinkio
import pytest
import numpy as np

def test_run_correctly():
    """
    Smoke test of whether the function runs without errors for all edf files in the data folder
    """

    data_path = os.path.join('data')
    
    # for all edf files in the data folder
    for file in os.listdir(data_path):
        if file.endswith('.edf'):
            
            # read the edf file
            edf_path = os.path.join(data_path, file)
            edf = eyelinkio.read_edf(edf_path)
            
            # specify the screen coordinates and the AOI definitions
            screen_coords = edf['info']['screen_coords']
            aoi_definitions = [{'shape':'rectangle', 'coordinates': ('462', '590', '563', '691')}]
            
            # define the AOI
            _ = define_aoi(screen_coords, aoi_definitions)
    
    return None
           
def test_define_aoi_rectangle():
    """
    One shot test of whether the function correctly defines a rectangular AOI
    """
    screen_dimensions = (10, 10)
    aoi_definitions = [{'shape': 'rectangle', 'coordinates': (2, 5, 2, 5)}]
    
    mask = define_aoi(screen_dimensions, aoi_definitions)
    
    # Check dimensions
    assert mask.shape == screen_dimensions, 'Mask dimensions mismatch.'
    
    # Check values in AOI
    assert np.all(mask[2:5, 2:5] == 1), 'Rectangle AOI is incorrect.'
    assert np.all(mask[:2, :] == 0) and np.all(mask[:, :2] == 0), 'Outside AOI values are incorrect.'

def test_define_aoi_circle():
    """
    One shot test of whether the function correctly defines a circular AOI
    """
    screen_dimensions = (10, 10)
    aoi_definitions = [{'shape': 'circle', 'coordinates': (5, 5, 2)}]
    
    mask = define_aoi(screen_dimensions, aoi_definitions)
    
    # Check dimensions
    assert mask.shape == screen_dimensions, 'Mask dimensions mismatch.'
    
    # Check that circle center is included
    assert mask[5, 5] == 1, 'Circle center is not included.'
    
    # Check that outside circle is excluded
    assert mask[0, 0] == 0, 'Outside AOI values are incorrect.'

def test_define_aoi_invalid_input():
    """
    One shot test of whether the function raises errors for invalid input
    """
    # Test with invalid screen dimensions
    with pytest.raises(ValueError, match='Screen dimensions should have two elements.'):
        define_aoi((10,), [{'shape': 'rectangle', 'coordinates': (2, 5, 2, 5)}])
    
    # Test with unsupported AOI shape
    with pytest.raises(ValueError, match='Unsupported AOI shape: triangle'):
        define_aoi((10, 10), [{'shape': 'triangle', 'coordinates': (2, 5, 2, 5)}])
    
    # Test with invalid rectangle coordinates
    with pytest.raises(ValueError, match='Rectangle coordinates should have four elements'):
        define_aoi((10, 10), [{'shape': 'rectangle', 'coordinates': (2, 5)}])