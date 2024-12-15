# visualEyes
CSE 583 project working on a Python package to do quality control on eye tracking data while running experiments. 
This is a work in process!

# Project Description
Eye fixation directly impacts data quality in psychophysics and MRI experiments, making it a critical variable for vision scientists to control. Currently, most data quality inspections happen after the experiments, often leading to the exclusion of low-quality data. If fixation could be monitored in real time during the experiment, researchers could enhance data quality by communicating directly with participants. However, a user-friendly toolbox for this purpose is currently lacking. In this project, we aim to develop a Python package that provides clear and intuitive visualizations of eye-tracking data, helping both researchers and participants better understand and improve the quality of data being collected.

# Setup Instructions
1. Downlaod and install Eyelink Developers Kit by signing up for a (free) [Eyelink account](https://www.sr-research.com/support/member.php?action=register) and following the [install instructions for your OS](https://www.sr-research.com/support/thread-13.html). 

2. Clone our repository.
`git clone git@github.com:baharsener/visualEyes.git`

3. Create a Conda environment with our dependencies.
`conda env create -f environment.yml`

# Repository 
We provide a few example Eyelink Data File (EDF) files in `/data`. 

Demonstration scripts can be found in `/demo`

Core functions and tests can be found in `/visualeyes`. 

Other materials can be found in `/docs`. 

# Group Members
- Mckenzie Hagen (mphagen@uw.edu)
- Lydia Zhang (yzh22@uw.edu)
- Bahar Sener (sbsener@uw.edu)
- Brenda (Siyue) Qiu (siyueq@uw.edu)


