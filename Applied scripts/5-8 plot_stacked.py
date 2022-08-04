# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:51:20 2022

@author: Sverre Hassing
"""

import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import obspy

# File used to create figures 5-8, A-4, A-7 and A-8

path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

record0 = obspy.read('./Arrays/Crosscorr 5000 - line 0 - stack.mseed')
record1 = obspy.read('./Arrays/Crosscorr 5000 - line 1 - stack.mseed')

#%%

# Attach the CMP positions to the section
records = [record0,record1]
for line,record in enumerate(records):
    CMP_locs = np.load(f'./Arrays/CMP_locs{line}.npy')
    for trace,CMP_loc in zip(record,CMP_locs):
        trace.stats.distance = CMP_loc

# Which section to plot
inp_rec = record1.copy()

fig = inp_rec.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            handle = True,
            recordlength=2.)
ax = fig.gca()
ax.set_xlabel('Distance along line [km]')
fig.suptitle('')
ax.set_title(f'Stacked section from crosscorrelations - {len(inp_rec)} traces')

fig = inp_rec.plot(type='section',
            time_down = True,
            fillcolors = ([0.5,0.5,0.5],None),
            grid_color='white',
            handle = True,
            # recordlength=2.,
            size=(1000,2000*0.6))
ax = fig.gca()
ax.set_xlabel('Distance along line [km]')
fig.suptitle('')
ax.set_title(f'Stacked section from crosscorrelations - {len(inp_rec)} traces')