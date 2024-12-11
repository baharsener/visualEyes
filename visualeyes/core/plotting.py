import pandas as pd
import numpy as np
from eyelinkio import read_edf
import matplotlib.pyplot as plt
import os
from .utility import dataframe_validation

def plot_as_scatter(data, screen_dimensions, aoi_definitions=None, save_png=None, save_path=None):
    """
    plot the data on the AOI mask, and optionally save the plot to the working directory as a PNG file.

    Inputs:
    - data: pd.DataFrame
    - screen dimensions: tuple
        (height, width)
    - aoi_definitions: list of dict or None
        Each dict defines one AOI (so we can have multiple) with keys:
        - 'shape': 'rectangle' or 'circle'.
        - 'coordinates': tuple of coordinates:
            - for rectangluar AOI's: (x1, x2, y1, y2), upper-bounds non-inclusive. 
            - for circlular AOI's: (x_center, y_center, radius).
    - save_png: bool or None
        If True, the results will be saved as png file(s)
    - save_path: str or None 
        
    Outputs:
    - fig, ax

    """

    # check if input data is a pd.DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data should be a pandas dataframe")
    
    # validate the input dataframe and save the x and y coordinates if the dataframe is valid
    x_coord, y_coord = dataframe_validation(data)

    # check if there are data outside screen dimension
    out_of_bounds_x = np.where((x_coord < 0) | (x_coord >= screen_dimensions[1]))[0] # list of indices
    out_of_bounds_y = np.where((y_coord < 0) | (y_coord >= screen_dimensions[0]))[0] # list of indices

    out_of_bounds = np.unique(np.concatenate([out_of_bounds_x, out_of_bounds_y]))

    clean_data = data.drop(index=out_of_bounds)

    fig, ax = plt.subplots()

    ax.set_xlim(0, screen_dimensions[1]-1)

    ax.set_ylim(0, screen_dimensions[0]-1)

    ax.set_ylim(ax.get_ylim()[::-1])

    ax.xaxis.tick_top()

    # plot scatter plot
    if 'axp' in clean_data.columns and 'ayp' in clean_data.columns:
        fixation_duration = []
        for _, series in clean_data.iterrows():
            single_duration = series['etime'] - series['stime']
            fixation_duration.append(single_duration)
        max_duration = round(max(fixation_duration), 2)
        mag_factor = [duration/max_duration for duration in fixation_duration]
        ax.scatter(clean_data['axp'], clean_data['ayp'], color='skyblue', marker='o', facecolors='none', s=[160 * mag for mag in mag_factor]) 
    else:
        ax.scatter(clean_data['xpos'], clean_data['ypos'], color='skyblue', marker='o', s=60)
    
    # optionally, overlay aoi
    if aoi_definitions is not None:
        # ax = visualeyes.aoi_contour
        pass

    if save_png:
        if not save_path:
            file_path = 'scatter_plot.png'
        else:
            file_path = os.path.join(save_path, 'scatter_plot.png')

        fig.savefig(file_path)
        print(f"Saved PNG file to {file_path}")

    return fig, ax

def aoi_overlay(aoi_definitions, screen_width, screen_height):
    """
    Overlay shape of AOIs on plots.

    Parameters:
    - aoi_definitions (list): A list of AOI dictionaries, containing:
        - shape (str): "rectangle" or "circle" AOI's.
        - coordinates (list): Coordinates defining the AOI.
          For "rectangle": [x1, x2, y1, y2].
          For "circle": [x_center, y_center, radius].
    - screen_width (int): Width of the screen in pixels.
    - screen_height (int): Height of the screen in pixels.

    Returns:
    - list: A list of AOI boundaries as lists of (x, y) tuples.
    """
    aoi_boundaries = []

    for idx, aoi in enumerate(aoi_definitions): #check for each AOI and make the error specific to that AOI
        shape = aoi['shape'].lower()
        coordinates = aoi['coordinates']
        
        # Check shape
        if shape not in ['rectangle', 'circle']:
            raise ValueError(f"Unsupported AOI shape '{shape}' in AOI {idx + 1}. Must be 'rectangle' or 'circle'.")
        
        # Check coordinates
        if not all(isinstance(coord, (int, float)) for coord in coordinates):
            raise ValueError(f"All coordinates must be numbers in AOI {idx + 1}.")
        
        if any(coord < 0 for coord in coordinates):
            raise ValueError(f"Coordinates cannot be negative in AOI {idx + 1}.")
        
        if shape == 'rectangle':
            if len(coordinates) != 4:
                raise ValueError(f"Rectangle AOI {idx + 1} must have 4 coordinates: [x1, x2, y1, y2].")
            
            x1, x2, y1, y2 = coordinates
            
            if x1 > x2:
                raise ValueError(f"x1 cannot be greater than x2 in AOI {idx + 1}.")
            if y1 > y2:
                raise ValueError(f"y1 cannot be greater than y2 in AOI {idx + 1}.")
            
            if x2 > screen_width or y2 > screen_height:
                raise ValueError(f"AOI {idx + 1} exceeds screen boundaries (Width: {screen_width}, Height: {screen_height}).")
            
            #Add rectangle boundaries
            aoi_boundaries.append([(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)])
        
        elif shape == 'circle':
            if len(coordinates) != 3:
                raise ValueError(f"Circle AOI {idx + 1} must have 3 coordinates: [x_center, y_center, radius].")
            
            x_center, y_center, radius = coordinates
            
            if radius <= 0:
                raise ValueError(f"Radius must be positive in circle AOI {idx + 1}.")
            
            if (x_center - radius < 0 or x_center + radius > screen_width or
                y_center - radius < 0 or y_center + radius > screen_height):
                raise ValueError(f"Circle AOI {idx + 1} exceeds screen boundaries (Width: {screen_width}, Height: {screen_height}).")
            
            #Add circle boundary
            theta = np.linspace(0, 2 * np.pi, 100)
            x = x_center + radius * np.cos(theta)
            y = y_center + radius * np.sin(theta)
            aoi_boundaries.append(list(zip(x, y)))

    return aoi_boundaries

def plot_heatmap(data, screen_coords, aoi_boundaries=None, bins=None):
    """
    Plots a heatmap of eye-tracking data and overlays AOIs if defined.

    Parameters:
    - data: DataFrame containing 'xpos' and 'ypos' for plotting.
    - screen_coords: Tuple of (screen_width, screen_height,).
    - aoi_definitions: List of dictionaries defining the AOIs (optional).
    - bins: Either an integer specifying the number of bins for both dimensions,
            or a tuple (bins_x, bins_y) for separate bin sizes.
    """
    # Filter NaN values in the gaze columns
    data = data.dropna(subset=['xpos', 'ypos'])

    # Get screen width and height
    screen_width, screen_height = screen_coords

    # Filter gaze data outside the screen coordinates
    data = data[(data['xpos'] >= 0) & (data['xpos'] <= screen_width) & 
            (data['ypos'] >= 0) & (data['ypos'] <= screen_height)]

    # Determine bins (depends a bit on screen)
    if bins is None:  # Default bins, 20 px bins here if nothing else is given
        bins_x = int(screen_width / 20)
        bins_y = int(screen_height / 20)
    elif isinstance(bins, int):  # User-defined, if bins for x and y are the same
        bins_x = bins_y = bins
    elif isinstance(bins, (tuple, list)) and len(bins) == 2:  # User-defined, if different bins for x and y are desired
        bins_x, bins_y = bins
    else:
        raise ValueError("`bins` must be an integer or a tuple of two integers.")

    heatmap = np.histogram2d(
    data['xpos'], data['ypos'], bins=[bins_x, bins_y])[0]

    # Plot the heatmap:
    plt.imshow(heatmap.T, origin='lower', cmap='viridis', extent=[0, screen_width, 0, screen_height])
    plt.colorbar(label='Count')

    # Adjust x and y axis limits
    plt.xlim(0, screen_width)
    plt.ylim(0, screen_height)

    # If AOIs are defined, draw them
    if aoi_boundaries:
        for boundary in aoi_boundaries:
            # Outline the AOI by drawing its boundary
            boundary = np.array(boundary)
            plt.plot(boundary[:, 0], boundary[:, 1], color='red', lw=1)

    # Invert y-axis to set (0, 0) at the top-left
    plt.gca().invert_yaxis() 

    # Add title and show
    plt.title('Heatmap of Gaze Data')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()