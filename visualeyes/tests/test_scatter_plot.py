"""
Tests for the plot as scatter function
"""
import pytest
import pandas as pd
import eyelinkio
import os 
import matplotlib
import matplotlib.collections as collections

from visualeyes import plot_as_scatter

datapath = os.path.join(os.path.dirname(__file__),'..', 'data')
filename = 'TG_2017.EDF'
data = os.path.join(datapath, filename)
data = eyelinkio.read_edf(data)
screen_dimensions = (data['info']['screen_coords'][1], data['info']['screen_coords'][0]) #(height, width)
df = data.to_pandas()

def test_smoke():

    """
    Simple smoke test to make sure 'plot_as_scatter' runs
    """
    
    fig, ax = plot_as_scatter(df, screen_dimensions=screen_dimensions)

    return None

def test_aspect_ratios():
    """
    Test to check the ratio of the figure is the same as the ratio of the screen
    """
    ratio_screen = data['info']['screen_coords'][0]/data['info']['screen_coords'][1] #width/height
    fig, ax = plot_as_scatter(df, screen_dimensions=screen_dimensions)
    ratio_fig = fig.get_size_inches()[0]/fig.get_size_inches()[1]

    assert ratio_screen == ratio_fig, "the aspect ratios of the screen and the fig don't match"

    return None

def test_fig_scatterplot():
    """
    Test to check if the fig returned by the function is a scatterplot
    """
    fig, ax = plot_as_scatter(df, screen_dimensions=screen_dimensions)
    assert any(isinstance(coll, collections.PathCollection) for coll in ax.collections), "No scatter plot found in ax.collections"

    return None