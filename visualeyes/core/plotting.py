import pandas as pd
import numpy as np
from visualeyes.processing import define_aoi
from eyelinkio import read_edf
import matplotlib.pyplot as plt
import os

def plot_fixations_aoi(filepath, dataframe, savepath):
    """
    calculate the ratio of fixations within the Area of Interest (AOI) for each trial
    plot all fixations on the binary image and save as svg file

    Inputs:
    - filepath: string
        path to the edf file
    - dataframe: pd.DataFrame
        generated by edf.to_pandas() 
    - savepath: string
        a path to save the output svg (one svg per trial)

    Outputs:
    - ratios: list (returned)
        ratios of fixations within the Area of Interest (one ratio per trial)
    - plots: svg
        plots of fixations ('cross') on the binary image (one svg per trial)

    """

    # check if input data is a pd.DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('input data should be a pandas dataframe')
    
    # check if the 'discrete' column has both 'trials' and 'fixations' sub-columns
    if 'trials' not in dataframe.columns:
        raise ValueError('missing trials information')
    if 'fixations' not in dataframe.columns:
        raise ValueError('missing fixations information')

    # pull the 'trials' and 'fixations' columns
    df_trials = dataframe['discrete']['trials']
    df_fixations = dataframe['discrete']['fixations']

    # start an empty list to save ratio of fixations within the aoi for each trial
    ratios_fixations_within = []

    for row_idx, trial in df_trials.iterrows():
        print(f'trial indx: {row_idx}')
        trial_stime = trial['stime']
        trial_etime = trial['etime']

        # locate the fixations within trial start and end time
        fixations_in_trial = df_fixations[(df_fixations['stime'] >= trial_stime) & (df_fixations['etime'] <= trial_etime)]

        # find the x and y coordinates of fixations 
        x_coor = fixations_in_trial ['axp']
        y_coor = fixations_in_trial ['ayp']

        # count the number of fixations
        num_fixations = x_coor.shape[0]

        # generate the binary image as aoi mask
        edf_file = read_edf(filepath)
        
        # extract screen coordinates from the 'info' column of the edf file
        screen_width = edf_file['info']['screen_coords'][0]
        screen_height = edf_file['info']['screen_coords'][1]

        # define boundaries of the aoi (assume rectangle shape)
        SHAPE = 'rectangle'
        PIXELS = 100
        x1 = screen_width/2 - PIXELS
        x2 = screen_width/2 + PIXELS
        y1 = screen_height/2 - PIXELS
        y2 = screen_height/2 + PIXELS
        
        # initiate aoi parameters dict
        aoi_parameters = [{'shape': SHAPE, 'coordinates': (x1, y1, x2, y2)}]

        # generate aoi mask
        aoi_mask = define_aoi(screen_width, screen_height, aoi_parameters)

        plt.imshow(aoi_mask, cmap='gray')
        plt.title("Fixations per trial on the AOI mask")
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1]) # invert the y-axis to align with eyelink
        plt.gca().xaxis.tick_top()
        plt.scatter(x_coor, y_coor, color='skyblue', marker='x', s=15) # plot fixations within trial
        plt.savefig(os.path.join(savepath, f'Fixations_in_AOI_{row_idx+1}.svg')) # save the svg for each trial in the savepath

        # calculate ratio of fixations within the AOI for the current trial
        fixations_within_aoi = 0 

        for x,y in zip(x_coor, y_coor):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                fixations_within_aoi += 1
                ratio = 100* fixations_within_aoi/num_fixations

        ratios_fixations_within.append(ratio)

    return ratios_fixations_within
