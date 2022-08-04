# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:46:16 2022

@author: Sverre Hassing
"""

import obspy
import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import glob

# File used to deconvolve virtual shot gathers. 

# Path to coordinate information
path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

stations = np.load('./Arrays/Stations.npy')
line_id = np.load('./Arrays/line_id.npy')

# Which line is used
line = '0'

# Path to the virtual shot gathers
path_saved = 'E:/Thesis/Arrays/Crosscorr 5000'
# Path to where deconvolved virtual shot gathers are saved
path_out = 'E:/Thesis/Arrays/Crosscorr 5000 - decon'

# Deconvolution parameter
n = 300 # lag times of autocorrelation used


#%%

# Check if the output folder exists and make it otherwise
if not os.path.isdir(os.path.join(path_out)):
    os.mkdir(os.path.join(path_out))

file_list = glob.glob(os.path.join(path_saved,line,f'Line {line} - shot *.mseed'))

fragment = f'Line {line} - shot '

for i,file in enumerate(file_list):
    record = obspy.read(file)
    
    str_idx_stat = file.find(fragment)+len(fragment)
    station = file[str_idx_stat:str_idx_stat+4]
    stat_id = np.argwhere(stations[line_id==int(line)]==station)[0][0]
    
    record = th.coord.attach_distances(record,i,line,path_info)
    record_filt = record.filter('bandpass',freqmin=5,freqmax=40,corners=5)
    
    print(f'\r{i} - Opened...',end='')
    
    record_wiener = th.proc.wiener_decon_stream(record_filt,i,n)
    record_w_filt = record_wiener.filter('bandpass',freqmin=5,freqmax=40,corners=5)
    
    filename = os.path.split(file)[-1]
    
    print(f'\r{i} - Processed...',end='')
    
    # Check if the folder exists and otherwise create it
    if not os.path.isdir(os.path.join(path_out,line)):
        os.mkdir(os.path.join(path_out,line))
    
    record_w_filt.write(os.path.join(path_out,line,filename))
    
    print(f"\r{i} - Saved...",end='')