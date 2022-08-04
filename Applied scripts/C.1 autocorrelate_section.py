# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 00:27:51 2022

@author: Sverre Hassing
"""

import os
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th

# File used to generate the autocorrelated zero-offset sections

station = "7149"
component = 'Z'
path_base = os.path.normpath('E:/Thesis/clip_data/')
path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')
path_saved = os.path.normpath("E:/Thesis/Arrays/")

# The minimum apparent velocity to use for the autocorrelations
vel_cut = 10000 # m/s
window_length = 10. # s
mtr_station = '7149'
# Which illumination analysis results to use
added_string = ' - filtered'

# Create the autocorrelation sections
record0, record1 = th.proc.autocorr_section(path_base, path_saved, path_info, mtr_station, component, window_length, vel_cut, added_string)

record0.write(f'./Arrays/Autocorr - line 0 - vel{int(vel_cut)}{added_string}.mseed')
record1.write(f'./Arrays/Autocorr - line 1 - vel{int(vel_cut)}{added_string}.mseed')