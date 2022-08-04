# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 23:32:43 2022

@author: Sverre Hassing
"""

import obspy
import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import matplotlib.pyplot as plt
import glob

# File to create figure 4-1 in the thesis

station = "6384" # The station to use
# The time to clip the data to
time_start = "2021-07-27T22:12:50" 
time_end = "2021-07-27T22:13:00"
# The component to use
component = 'Z'
# Which line to plot
line = '0'
# Path to the raw data
path_base = os.path.normpath('E:/Thesis/clip_data/')
# Path to coordinate information
path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

# The file to open for this data
file = '2021.07.27.22.00.00.Z.mseed'
day = '27'

#%% The plotting

# Convert dates to UTCDateTime
time_start = obspy.core.UTCDateTime(time_start)
time_end = obspy.core.UTCDateTime(time_end)

# Set up the path of the file
path_file = os.path.join(path_base,day,file)

# Read the file
record = obspy.read(path_file)
# Select only one of the lines
record = th.coord.select_line(record, line, path_info)
# Trim the record down to the right time
record = record.trim(time_start,time_end)
# Attach distance information to each trace
record = th.coord.attach_distances(record, 51, line, path_info)

# Plot the section
fig = record.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            handle = True)
ax = fig.gca()
ax.set_xlabel('Distance along line [km]')
fig.suptitle('')
ax.set_title(f'Noise panel on main line - {len(record)} traces')