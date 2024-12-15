# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:vizeyes]
#     language: python
#     name: conda-env-vizeyes-py
# ---

import os
import eyelinkio
import pandas as pd
import sys
import matplotlib.pyplot as plt
from visualeyes.core import plot_heatmap
import argparse


# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id',default='2015') 
parser.add_argument('--project_id', default='TG') #default for testing; 
parser.add_argument('--data_path', default='data') 

#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])

subject_id = args.subject_id
project_id = args.project_id
data_path = args.data_path

# -

edf_path = os.path.join(data_path, f'{project_id}_{subject_id}.EDF') 

edf = eyelinkio.read_edf(edf_path)


edf.to_pandas()['discrete']['messages']['stime'].iloc[-1]

sum(edf.to_pandas()['discrete']['blinks']['etime'] - edf.to_pandas()['discrete']['blinks']['stime'])

edf['info']

#Get screen coordinates
screen_coordinates = edf['info']['screen_coords']
print(screen_coordinates)

# +
samples = edf['samples']
df = pd.DataFrame(edf['samples'].T, columns=['xpos', 'ypos', 'pupil_area'])

#Define an AOI
screen_width, screen_height = screen_coordinates
aoi_definitions = [{'shape': 'rectangle', 'coordinates': (400, 590, 500, 691)}]

#Gaze data (just x and y)
data = df.dropna(subset=['xpos', 'ypos'])

#Plot
plot_heatmap(data, screen_coordinates, aoi_definitions, save_png='heatmap.png')

# +
with open("output.html", "w") as f: 
    f.write(f'<h>Quality report for participant {project_id} {subject_id}</h>') 
    f.write('\n')
    f.write('\n')

    f.write(f'<p>{screen_coordinates}</p>') 

    f.write('<p>Placeholder: participant spent X amount of time lookiong at AOI<p>')

    f.write(f'<p>Total time of task: {edf.to_pandas()['discrete']['messages']['stime'].iloc[-1]}</p>')
    f.write(f'<img src="heatmap.png">')

    

# -

import webbrowser
res = webbrowser.open_new_tab('output.html')


res


