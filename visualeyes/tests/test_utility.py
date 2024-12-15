"""Testing functions in visualeyes.core.utility.py"""

import numpy as np
import pandas as pd
import pytest
from visualeyes.core._utility import (aoi_mask_validation, dataframe_validation,
                                        aoi_definitions_validation, screen_dimensions_validation)

def test_aoi_mask_validation():
    
    """
    One shot tests for aoi_mask_validation function.
    """
    
    # Valid case
    aoi_mask = np.array([[1, 0], [0, 1]])
    screen_dimensions = (2, 2)
    assert aoi_mask_validation(aoi_mask, screen_dimensions) is None

    # Test invalid aoi_mask type
    with pytest.raises(ValueError, match='AOI mask should be a numpy array'):
        aoi_mask_validation([[1, 0], [0, 1]], screen_dimensions)

    # Test non-2D aoi_mask
    with pytest.raises(ValueError, match='AOI mask should be a 2D array'):
        aoi_mask_validation(np.array([1, 0, 1]), screen_dimensions)

    # Test non-binary mask
    with pytest.raises(ValueError, match='AOI mask should be a binary mask'):
        aoi_mask_validation(np.array([[2, 0], [0, 1]]), screen_dimensions)

    # Test mismatched dimensions
    with pytest.raises(ValueError, match='AOI mask should have the same shape as the screen dimension'):
        aoi_mask_validation(np.array([[1, 0, 1], [0, 1, 0]]), screen_dimensions)

def test_dataframe_validation():
    """
    One shot tests for dataframe_validation function.
    """
    # Valid case
    df = pd.DataFrame({'xpos': [10, 20], 'ypos': [30, 40]})
    screen_dimensions = (50, 50)
    (x, y), outliers = dataframe_validation(df, screen_dimensions)
    assert list(x) == [10, 20]
    assert list(y) == [30, 40]
    assert len(outliers) == 0

    # Missing coordinates columns
    invalid_df = pd.DataFrame({'axp': [10], 'another_column': [20]})
    with pytest.raises(ValueError, match='Missing x and y coordinates'):
        dataframe_validation(invalid_df, screen_dimensions)

    # Outliers
    df = pd.DataFrame({'xpos': [-10, 20, 60], 'ypos': [30, 40, 60]})
    (x, y), outliers = dataframe_validation(df, screen_dimensions)
    assert list(outliers) == [0, 2]

    # Drop NaN
    df = pd.DataFrame({'xpos': [10, None], 'ypos': [30, 40]})
    (x, y), outliers = dataframe_validation(df, screen_dimensions, drop_nan=True)
    assert list(x) == [10]
    assert list(y) == [30]
    
def test_aoi_definitions_validation():
    screen_dimensions = (100, 100)

    # Valid rectangle AOI
    aoi_def = {'shape': 'rectangle', 'coordinates': (10, 20, 10, 20)}
    assert aoi_definitions_validation(aoi_def, screen_dimensions) is None

    # Valid circle AOI
    aoi_def = {'shape': 'circle', 'coordinates': (50, 50, 10)}
    assert aoi_definitions_validation(aoi_def, screen_dimensions) is None

    # Missing keys
    aoi_def = {'shape': 'rectangle'}
    with pytest.raises(KeyError, match='Missing keys in AOI definition'):
        aoi_definitions_validation(aoi_def, screen_dimensions)

    # Invalid shape
    aoi_def = {'shape': 'triangle', 'coordinates': (10, 20, 10, 20)}
    with pytest.raises(ValueError, match='Unsupported AOI shape'):
        aoi_definitions_validation(aoi_def, screen_dimensions)

    # AOI exceeding boundaries
    aoi_def = {'shape': 'rectangle', 'coordinates': (10, 110, 10, 20)}
    with pytest.raises(ValueError, match='AOI exceeds screen boundaries'):
        aoi_definitions_validation(aoi_def, screen_dimensions)

    # Negative coordinates
    aoi_def = {'shape': 'circle', 'coordinates': (-10, 50, 10)}
    with pytest.raises(ValueError, match='Coordinates cannot be negative'):
        aoi_definitions_validation(aoi_def, screen_dimensions)


def test_screen_dimensions_validation():
    # Valid case
    screen_dimensions = (100, 200)
    assert screen_dimensions_validation(screen_dimensions) is None

    # Invalid type
    with pytest.raises(ValueError, match='Screen dimensions should be a tuple, list, or numpy array'):
        screen_dimensions_validation('100, 200')

    # Invalid size
    with pytest.raises(ValueError, match='Screen dimensions should have two elements'):
        screen_dimensions_validation((100,))

    # Non-integer values
    with pytest.raises(ValueError, match='Screen dimensions should have positive integer values'):
        screen_dimensions_validation((100, 200.5))

    # Non-positive values
    with pytest.raises(ValueError, match='Screen dimensions should have positive integer values'):
        screen_dimensions_validation((100, -200))

