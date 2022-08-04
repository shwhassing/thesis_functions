# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:21:48 2022

@author: Sverre Hassing
"""

import obspy
import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
# working_dir = os.path.normpath('D:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import glob

# File used to transform the CMP gathers into a stacked section 

# Path to saved CMP gathers
path_cmp = 'E:\\Thesis\\Arrays\\CMP 5000\\'
# Path to coordinate information
path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

# Which line to process
line = '0'

# AGC parameters
oper_len = 1.
type_scal = 'mean'
basis = 'centred'

# Muting parameters
method = 'ramp'
vel_mute = 400
shift = -0.15
len_ramp = 0.1

# Velocity model
test_times = [0.0,0.35,0.5,1.1,2.]
test_vels = [1800,2200,2500,4000,6000]

#%%

# Load information about CMP locations
offset_mat = np.load(f'./Arrays/offset_mat{line}.npy')
CMP_locs = np.load(f'./Arrays/CMP_locs{line}.npy')

# Get the full list of CMP gathers as files
file_list = glob.glob(os.path.join(path_cmp,line,'*.mseed'))

# Open an example file
record = obspy.read(file_list[0])
times = record[0].times()
# Interpolate the velocity model
vel = np.interp(times,test_times,test_vels)

print(f'Progress:\n0/{len(file_list)}',end='')

# Initialise a new stream
record_stack = obspy.Stream()

# Go through each CMP gather
for counter,file in enumerate(file_list):
    
    # Extract the CMP position from the filename
    filename = os.path.split(file)[-1]
    start = filename.find('CMP ') + 4
    end = filename.find('.mseed')
    cmp_pos = float(filename[start:end])
    
    # Find the index of the closest CMP location in the saved array
    close_idx = th.util.find_closest(CMP_locs,cmp_pos)
    # Find the list of offsets that belong to this gather
    offset = offset_mat[close_idx,:].squeeze()
    
    # Read the file
    record = obspy.read(file)
    # and attach the offsets
    for i,trace in enumerate(record):
        trace.stats.distance = offset[i]
        
    # Do some processing
    record = th.proc.normalise_section(record)
    record = th.proc.AGC(record,oper_len,type_scal,basis)
    record = th.proc.mute_cone(record, method, vel_mute, shift, len_ramp=len_ramp)
    
    # Perform the NMO correction with the velocity model
    NMO_data = th.proc.NMO_corr(record,vel)
    
    # Copy information over to the new, stacked trace
    new_trace = obspy.Trace()
    new_trace.data = NMO_data.sum(axis=1)
    new_trace.stats.sampling_rate = record[counter].stats.sampling_rate
    new_trace.stats.starttime = record[counter].stats.starttime
    new_trace.stats.channel = record[counter].stats.channel
    new_trace.stats.distance = cmp_pos
    new_trace.stats.station = str(cmp_pos)
    
    # Add it to the stacked section
    record_stack += new_trace
    
    print(f'\r{counter+1}/{len(file_list)}',end='')

CMP_folder = path_cmp.split('\\')[-2]
record_stack.write(f'Crosscorr {CMP_folder[-4:]} - line {line} - stack.mseed')

#%%

# Plot the resulting section if you want
fig = record_stack.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            recordlength=2.,
            recordstart=0,
            handle = True,
            title=f'Section line {line} - {len(record_stack)} traces'
            )

ax = fig.gca()
ax.set_xlabel("Distance along line [km]")
fig.suptitle("")
ax.set_title(f'Section line {line} - {len(record_stack)} traces')