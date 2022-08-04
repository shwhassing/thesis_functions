# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:21:46 2022

@author: Sverre Hassing
"""

import os
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import numpy as np

# File used to extract some basic information that will be used more often

path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

# If this folder does not exist yet, create a folder
if not os.path.isdir(os.path.dirname('./Arrays/')):
    os.mkdir(os.path.dirname('./Arrays/'))

stakes, stations, coords = th.coord.read_coords(path_info)

np.save('./Arrays/Stations.npy',stations)

line_id = np.zeros(len(stakes))
for i,stake in enumerate(stakes):
    line_id[i] = int(stake[1])
    
np.save('./Arrays/line_id.npy',line_id)