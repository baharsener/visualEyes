"""Test the percent_data_in_aoi function."""

import pandas as pd
import numpy as np
from visualeyes import percent_data_in_aoi

def test_run_correctly():
    """
    Smoke test of whether the function runs without errors for valid input
    """
    
    # Define input
    df = pd.DataFrame({'xpos': [1, 2, 3], 'ypos': [1, 2, 3]})  # 3 points total
    aoi_mask = np.array([[0, 1, 0, 0],  # Only (2, 2) and (3, 3) are inside the AOI
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 0]])
    screen_dimensions = (4, 4)  # Matches aoi_mask

    # Run the function
    result = percent_data_in_aoi(df, aoi_mask, screen_dimensions)

    # Verify the result
    expected_result = (2 / 3) * 100  # Two out of three points are inside AOI
    assert np.isclose(result, expected_result, atol=1e-6), f'Expected {expected_result}, got {result}'

    return None
    
def test_no_data_in_aoi():
    
    # Define input
    df = pd.DataFrame({'xpos': [0, 3, 3, 0], 'ypos': [0, 3, 3, 0]})
    aoi_mask = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    screen_dimensions = (4, 3)

    # Run the function
    result = percent_data_in_aoi(df, aoi_mask, screen_dimensions)

    # Check the result
    assert result == 0.0, f'Expected 0.0, got {result}'
    
    return None
    
def test_all_data_in_aoi():
    
    # Define input
    df = pd.DataFrame({'xpos': [1, 2, 1, 2], 'ypos': [1, 2, 2, 1]})
    aoi_mask = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 0]])
    screen_dimensions = (4, 3)

    # Run the function
    result = percent_data_in_aoi(df, aoi_mask, screen_dimensions)

    # Check the result
    assert result == 100.0, f'Expected 100.0, got {result}'
    
    return None