# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 00:22:03 2022

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

# File used to create figure 5-3, 5-6, and A-1 to A-3

path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

record0 = obspy.read('./Arrays/Autocorr - line 0 - vel10000.mseed')
record1 = obspy.read('./Arrays/Autocorr - line 1 - vel10000.mseed')
# Alternatively
record0_filt = obspy.read('./Arrays/Autocorr - line 0 - vel10000 - filtered.mseed')
record1_filt = obspy.read('./Arrays/Autocorr - line 1 - vel10000 - filtered.mseed')

# Attach the distance along the line to each trace
record0 = th.coord.attach_distances(record0, 51, '0', path_info)
record1 = th.coord.attach_distances(record1, 37, '1', path_info)
record0_filt = th.coord.attach_distances(record0_filt, 51, '0', path_info)
record1_filt = th.coord.attach_distances(record1_filt, 37, '1', path_info)

# Which record to use for the plotting
inp_rec = record1_filt.copy()
oper_len = 1.0 # Operator length for AGC
type_scal = 'mean' # Method for AGC
basis = 'centred' # Which part to use
power_constant = 1.6 # Power constant for TAR

# Process the section
new_rec = th.proc.TAR(inp_rec,power_constant)
new_rec = th.proc.AGC(new_rec,oper_len,type_scal,basis)
# Plot the unfiltered section
fig = new_rec.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            handle = True,
            recordlength=2.)
ax = fig.gca()
ax.set_xlabel('Distance along line [km]')
fig.suptitle('')
ax.set_title(f'Zero-offset section from autocorrelations - {len(record0)} traces')

# Filter and then process the section
rec_filt = inp_rec.filter('bandpass',freqmin=5,freqmax=40,corners=5)
rec_filt = th.proc.TAR(rec_filt,power_constant)
rec_filt = th.proc.AGC(rec_filt,oper_len,type_scal,basis)
# Plot the filtered section
fig = rec_filt.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            handle = True,
            recordlength=2.)
ax = fig.gca()
ax.set_xlabel('Distance along line [km]')
fig.suptitle('')
ax.set_title(f'Zero-offset section from autocorrelations - {len(record0)} traces')