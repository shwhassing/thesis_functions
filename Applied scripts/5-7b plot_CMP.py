# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:49:41 2022

@author: Sverre Hassing
"""

import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th

# File used to create figure 5-7b in the thesis

# Path to CMP gathers
path_cmp = 'E:\\Thesis\\Arrays\\CMP 5000\\'
# Path to coordinate information
path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

# Line to use
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

# Velocity model - linearly interpolated between these values
test_times = [0.0,0.35,0.5,1.1,2.]
test_vels = [1800,2200,2500,4000,6000]

# Which CMP position to open, finds the nearest CMP
cmp_pos = 575


#%%

record = th.of.load_cmp(path_cmp,line,cmp_pos)

# Perform some processing on the record
record = th.proc.normalise_section(record)
record = th.proc.AGC(record,oper_len,type_scal,basis)
record = th.proc.mute_cone(record, method, vel_mute, shift, len_ramp=len_ramp)

# Get the offset at each trace
dists = []
for trace in record:
    dists.append(trace.stats.distance)
dists = np.sort(np.array(dists))

# Plot the original CMP gather
fig = record.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            recordlength=2.,
            recordstart=0,
            handle = True)
ax = fig.gca()
fig.suptitle('')
ax.set_title(f'Common Midpoint Gather with midpoint at {int(cmp_pos)} m along line - fold {len(record)}')

# Add the hyperbolae from velocity picks
for test_vel,test_time in zip(test_vels,test_times):
    hyperbola = np.sqrt(np.square(dists)/np.square(test_vel)+test_time**2)
    ax.plot(dists/1000,hyperbola,c='r')

# Interpolate the velocity model
times = record[0].times()
vel = np.interp(times,test_times,test_vels)

# Perform the NMO correction
record_shift = th.proc.recreate_stream_NMO(th.proc.NMO_corr(record,vel),record)

# Plot the shifted record
fig = record_shift.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            recordlength=2.,
            recordstart=0,
            handle = True)
ax = fig.gca()

# Add the corrected hyperbolae
for test_time in test_times:
    ax.hlines(test_time,dists.min()/1000,dists.max()/1000, color='r')
fig.suptitle('')
ax.set_title(f'NMO corrected CMP gather with midpoint at {int(cmp_pos)} m along line - fold {len(record)}')