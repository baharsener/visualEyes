"""Test the epoch_data function."""

import pytest
import pandas as pd
from visualeyes import epoch_data

def test_run_correctly():
    """
    Smoke test of whether the function runs without errors for valid input
    """
    eye_data = pd.DataFrame({'time': [0, 1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50, 60]})
    window_start = [0, 3]
    window_duration = [2, 2]
    
    _ = epoch_data(eye_data, window_start, window_duration)
    
    return None

def test_epoch_data_valid_input():
    """
    One shot test of whether the function runs without errors for valid input
    """
    eye_data = pd.DataFrame({'time': [0, 1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50, 60]})
    window_start = [0, 3]
    window_duration = [2, 2]
    
    epochs, epoched_data = epoch_data(eye_data, window_start, window_duration)
    
    print(epoched_data)
    
    # Check epochs
    assert epochs == [(0, 2), (3, 5)], 'Epoch start and end times mismatch.'
    
    # Check epoched data
    assert len(epoched_data) == 4, 'Epoched data length mismatch.'
    assert all(epoched_data['epoch_index'].isin([0, 1])), 'Epoch indices are incorrect.'
    
    return None

def test_epoch_data_invalid_input():
    """
    One shot test of whether the function raises errors for invalid input
    """
    # Test with missing 'time' column
    eye_data = pd.DataFrame({'value': [10, 20, 30]})
    with pytest.raises(ValueError, match='data should contain a time column'):
        epoch_data(eye_data, 0, 2)
    
    # Test with invalid window_start type
    eye_data = pd.DataFrame({'time': [0, 1, 2]})
    with pytest.raises(ValueError, match='window_start should be a list, a numpy array, or a single value'):
        epoch_data(eye_data, 'invalid', 2)
    
    # Test with mismatched window_start and window_duration lengths
    with pytest.raises(ValueError, match='window_duration should have the same length as window_start'):
        epoch_data(eye_data, [0, 2], [1])
        
    return None    