# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:47:11 2022

@author: Sverre Hassing
"""

import obspy
import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import winsound
import time

# This file is used to cut up the original data in new chunks. The original
# data is stored with a .mseed file of 24 hours for one component and one 
# station. This is changed to be one file for 0.5 hours of data for one 
# component for all stations. Makes it significantly faster to work with.

# These are the limits of the amount of data that is processed at once. Keep
# in mind that in principle this should all go in the RAM. In practice, a large
# part will be written to the pagefile or something. It can get quite large
time_start = "2021-07-30T00:00:00"
time_end = "2021-07-31T00:00:00"

# Because the program takes a long time to run, there are various sound alerts
# built into the code. 

# Three short beeps means the program has crashed somewhere
# One long beep means the program has finished opening the files
# Three beeps repeated three times means the program has finished

# Which component to process
component = 'N'
# Path to the original data
base_path = os.path.normpath('E:\\Thesis\\raw_data')
# Output path for the new, clipped files
path_out = 'E:\\Thesis\\clip_data'

# The length of the new files in seconds
length_of_file = 30*60

# The list of all stations
stations = np.load('./Arrays/Stations.npy')

# Format to use when printing dates
date_format = '%Y:%m:%d %H:%M:%S'
# Format to use when saving the files
date_format_write = '%Y.%m.%d.%H.%M.%S'

#%%

# See when the program started
timestamp_start = obspy.core.UTCDateTime()

# Start by opening the files
try:
    record = th.of.open_diff_stat(stations, time_start, component, base_path, time_end = time_end)
except Exception:
    for i in range(3):
        winsound.Beep(1200, 500)
        time.sleep(0.2)
else:
    # If this succeeds continue with this
    timestamp_end = obspy.core.UTCDateTime()
    print(f"Opening lasted from {timestamp_start.strftime(date_format)} to {timestamp_end.strftime(date_format)}")
    print(f"Or a duration of {th.util.tfs_string(timestamp_end-timestamp_start)}")
    
    # Checked for masked arrays and convert them to filled arrays
    for trace in record:
        if isinstance(trace.data, np.ma.masked_array):
            trace.data = trace.data.filled(fill_value = 0)
    
    winsound.Beep(200,1000)
    
    print("Writing files:\n0/48", end = "")
    counter = 0
    
    # Now try cutting up the data and saving each chunk
    try:
        for window in record.slide(window_length = length_of_file, step = length_of_file):
            # Name of the new file
            name = f'{window[0].stats.starttime.strftime(date_format_write)}.{component}.mseed'
            # Write the file as .mseed
            window.write(os.path.join(path_out,f'{window[0].stats.starttime.day}',name),format='MSEED')
            
            counter += 1
            print(f'\r{counter}/48', end = "")
    except Exception:
        for i in range(3):
            winsound.Beep(1200, 500)
            time.sleep(0.2)
    else:
        # If the program finished, give three bursts of beeps
        duration = 500
        freq = 800
        # winsound.Beep(freq, duration)
        wait = 0.5
        
        amt = 3
        
        for i in range(amt):
            for i in range(amt):
                winsound.Beep(freq,duration)
                time.sleep(wait)
            time.sleep(3)