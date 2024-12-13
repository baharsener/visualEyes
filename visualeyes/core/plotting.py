import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ._utility import (dataframe_validation, aoi_definitions_validation, screen_dimensions_validation)
import numbers

def plot_as_scatter(data, screen_dimensions, aoi_definitions=None, save_png=None, save_path=None, marker_size=60):
    """
    Plot the data on the AOI mask, and optionally save the plot to the working directory as a PNG file.

    Parameters:
    ----------
    data: pd.DataFrame
        The data to be plotted. Must contain 'xpos' and 'ypos' columns or 'axp' and 'ayp' columns.
    screen_dimensions: tuple
        The dimensions of the screen in pixels (height, width).
    aoi_definitions: dict, list of dict, or None
        The AOI definitions to overlay on the plot. Each dict should contain:
        - 'shape': 'rectangle' or 'circle'.
        - 'coordinates': tuple, list or np.ndarray of coordinates.
            - for rectangluar AOI's: (x1, x2, y1, y2), upper-bounds non-inclusive. 
            - for circlular AOI's: (x_center, y_center, radius).
    save_png: bool or None
        Whether to save the plot as a PNG file.
    save_path: str or None
        The path to save the PNG file to.
        
    Returns:
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects of the plot.    
    """

    # check if input data is a pd.DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data should be a pandas dataframe")
    
    # check if screen_dimensions is valid
    screen_dimensions_validation(screen_dimensions)
    
    # validate the input dataframe and save the x and y coordinates if the dataframe is valid
    (x_coord, y_coord), _, _ = dataframe_validation(data, screen_dimensions, drop_outlier=True)

    # Validate the AOI definitions
    if aoi_definitions is not None:
        aoi_definitions_validation(aoi_definitions, screen_dimensions)

    # Initialize the plot
    fig, ax = plt.subplots()
    
  
    # Invert y-axis to match screen coordinates
    ax.invert_yaxis()
    ax.set_xlim(0, screen_dimensions[-1])
    ax.set_ylim(0, screen_dimensions[0])

    # Set the x-axis to the top

   # if the data contains 'axp' and 'ayp' columns, plot the data with varying marker sizes
    if 'axp' in data.columns and 'ayp' in data.columns:
        
        # calculate the fixation duration
        fixation_duration = []
        
        for _, series in data.iterrows():
            single_duration = series['etime'] - series['stime']
            fixation_duration.append(single_duration)
            
        # find a scaling factor for the marker size
        max_duration = round(max(fixation_duration), 2)
        mag_factor = [duration/max_duration for duration in fixation_duration]
        
        # plot the data
        ax.scatter(data['axp'], data['ayp'], color='skyblue', marker='o', 
                   facecolors='none', s=[3 * marker_size * mag for mag in mag_factor]) 

        # plot the data with a fixed marker size
    plt.scatter(y=data['ypos'], x=data['xpos'], color='skyblue', marker='o', s=marker_size) 

    # optionally, overlay aoi
    if aoi_definitions is not None:
        overlay_aoi(aoi_definitions, screen_dimensions, ax)

    if save_png:
        if not save_path:
            file_path = 'scatter_plot.png'
        else:
            file_path = os.path.join(save_path, 'scatter_plot.png')

        fig.savefig(file_path)
        print(f"Saved PNG file to {file_path}")

    return fig, ax

def overlay_aoi(aoi_definitions, screen_dimensions, ax):
    """
    Overlay shape of AOIs on plots.

    Parameters:
    ----------
    aoi_definitions: list of dict
        List of dictionaries defining the AOIs. Each dictionary should contain:
        - 'shape': 'rectangle' or 'circle'.
        - 'coordinates': list, tuple, or np.ndarray of coordinates.
    screen_dimensions: tuple
        The dimensions of the screen in pixels (height, width).
    ax: matplotlib.axes.Axes
        The axes object to overlay the AOIs on.
        
    Returns:
    -------
    ax: matplotlib.axes.Axes
        The axes object with the AOIs overlaid.
    """
    
    # Validate the input AOI definitions
    aoi_definitions_validation(aoi_definitions, screen_dimensions)
    
    # Validate the screen dimensions
    screen_dimensions_validation(screen_dimensions)
    
    screen_height, screen_width = screen_dimensions

    for idx, aoi in enumerate(aoi_definitions): # check for each AOI and make the error specific to that AOI
        shape = aoi['shape'].lower()
        coordinates = aoi['coordinates']

        if shape == 'rectangle':
            x1, x2, y1, y2 = map(int, coordinates)
            
            # Plot rectangle boundary on the axes
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red', lw=1)
        
        elif shape == 'circle':
            
            x_center, y_center, radius = coordinates
            
            # Calculate circle boundary points
            theta = np.linspace(0, 2 * np.pi, 100)
            x = x_center + radius * np.cos(theta)
            y = y_center + radius * np.sin(theta)
            
            # Plot circle boundary on the axes
            ax.plot(x, y, color='red', lw=1)

    return ax
def plot_heatmap(data, screen_dimensions, aoi_definitions=None, bins=None):
    """
    Plots a heatmap of eye-tracking data and overlays AOIs if defined.

    Parameters:
    - data: DataFrame containing 'xpos' and 'ypos' for plotting.
    - screen_dimensions: Tuple of (screen_height, screen_width).
    - aoi_definitions: List of dictionaries defining the AOIs (optional).
    - bins: Either an integer specifying the number of bins for both dimensions,
            or a tuple (bins_x, bins_y) for separate bin sizes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import numbers

    # Validate screen dimensions
    screen_height, screen_width = screen_dimensions
    screen_dimensions_validation(screen_dimensions)

    # Validate and preprocess the data
    (x_coord, y_coord), _, _ = dataframe_validation(data, screen_dimensions, drop_outlier=True)

    # Validate the AOI definitions
    if aoi_definitions is not None:
        aoi_definitions_validation(aoi_definitions, screen_dimensions)

    # Determine bins
    if bins is None:  # Default bins: 20 px bins
        bins_x = int(screen_width / 10)
        bins_y = int(screen_height / 10)
    elif isinstance(bins, numbers.Integral):  # User-defined equal bins
        bins_x = bins_y = bins
    elif isinstance(bins, (tuple, list, np.ndarray)) and len(bins) == 2:  # User-defined separate bins
        bins_x, bins_y = bins
    else:
        raise ValueError("`bins` must be an integer or a tuple of two integers.")

    # Create 2D histogram
    heatmap, x_edges, y_edges = np.histogram2d(
        x_coord, y_coord, bins=[bins_x, bins_y], 
        range=[[0, screen_width], [0, screen_height]]
    )

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap with fixed extent and reversed y-axis for top-left origin
    heatmap_img = ax.imshow(
        heatmap.T, interpolation='nearest', origin='upper', cmap='viridis',
        extent=[0, screen_width, 0, screen_height]
    )

    # Add a colorbar
    fig.colorbar(heatmap_img, ax=ax, label='Count')

    # Set axis limits and labels
    ax.set_xlim(0, screen_width)
    ax.set_ylim(screen_height, 0)  # Reverse y-axis for top-left origin
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')

    # Overlay AOI boundaries if defined
    if aoi_definitions is not None:
        ax = overlay_aoi(aoi_definitions, screen_dimensions, ax)

    # Return the figure and axis
    return fig, ax

def plot_heatmap(data, screen_dimensions, aoi_definitions=None, bins=None):
    """
    Plots a heatmap of eye-tracking data and overlays AOIs if defined.

    Parameters:
    - data: DataFrame containing 'xpos' and 'ypos' for plotting.
    - screen_dimensions: Tuple of (screen_height, screen_width).
    - aoi_definitions: List of dictionaries defining the AOIs (optional).
    - bins: Either an integer specifying the number of bins for both dimensions,
            or a tuple (bins_x, bins_y) for separate bin sizes.
    """

    # Get screen width and height
    screen_height, screen_width = screen_dimensions
    screen_dimensions_validation(screen_dimensions)

    # Validate the data
    (x_coord, y_coord), _, _ = dataframe_validation(data, screen_dimensions, drop_outlier=True)

    
    # Validate the AOI definitions
    if aoi_definitions is not None:
        aoi_definitions_validation(aoi_definitions, screen_dimensions)

    # Determine bins (depends a bit on screen)
    if bins is None:  # Default bins, 20 px bins here if nothing else is given
        bins_x = int(screen_width / 10)
        bins_y = int(screen_height / 10)
    elif isinstance(bins, numbers.Integral):  # User-defined, if bins for x and y are the same
        bins_x = bins_y = bins
    elif isinstance(bins, (tuple, list, np.ndarray)) and len(bins) == 2:  # User-defined, if different bins for x and y are desired
        bins_x, bins_y = bins
    else:
        raise ValueError("`bins` must be an integer or a tuple of two integers.")

    # Initialize the plot
    fig, ax = plt.subplots()

   
    heatmap,  xedges, yedges = np.histogram2d(
        x_coord, y_coord, bins=[bins_x, bins_y],
        range=[[0, screen_width], [0, screen_height]]
    )

    # Plot the heatmap
    heatmap_img = ax.imshow(heatmap.T, interpolation='nearest', origin='upper', cmap='viridis',
        extent=[0, xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=20)

    fig.colorbar(heatmap_img, ax=ax, label='Number of trials spent looking at screen location')

    #axis limits
    ax.set_xlim(0, screen_width)
    ax.set_ylim(screen_height, 0)
        
    # draw the AOI boundaries if defined
    if aoi_definitions is not None:
        ax = overlay_aoi(aoi_definitions, screen_dimensions, ax)
        
    #labels
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
        
    return fig, ax

