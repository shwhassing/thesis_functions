# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:54:15 2022

@author: Sverre Hassing
"""

import obspy
import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th

path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

stations = np.load('./Arrays/Stations.npy')
line_id = np.load('./Arrays/line_id.npy')

line = '0'

# Path to where deconvolved virtual shot gathers are saved
path_saved = 'E:/Thesis/Arrays/Crosscorr 5000 - decon'

station = '6847'

# AGC parameters
oper_len = 1.
type_scal = 'mean'
basis = 'centred'

# Muting parameters
method = 'ramp'
vel_mute = 400
shift = -0.15
len_ramp = 0.1

#%%

stat_id = np.argwhere(stations == station)[0][0]

filename = f'Line {line} - shot {station}.mseed'
path_full = os.path.join(path_saved,line,filename)
record = obspy.read(path_full)
record = th.coord.attach_distances(record,stat_id,line,path_info)

record_AGC = th.proc.AGC(record,oper_len,type_scal,basis)
record_mute = th.proc.mute_cone(record_AGC,method,vel_mute,shift,len_ramp)

fig = record_mute.plot(type='section',
                       time_down = True,
                       fillcolors = ([0.5,0.5,0.5],None),
                       grid_color='white',
                       handle = True,
                       recordlength=2.)
ax = fig.gca()
fig.suptitle('')
ax.set_xlabel('Distance to virtual source [km]')
ax.set_title(f'Virtual shot gather at station {station} - {len(record_mute)} traces')