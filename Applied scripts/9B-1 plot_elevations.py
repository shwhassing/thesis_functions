# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:21:18 2022

@author: Sverre Hassing
"""
import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import matplotlib.pyplot as plt

# Plot used to create figure B-1 in the thesis

# Path to coordinate information
path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')
# The two lines
lines = ['0','1']

# Plotting parameters
plot_size = 10
fsize = 15 # font size
x_pos = [[600,320],[600,270]] # position of equation label

#%%

# Read in the coordinate information
__, stations, coords = th.coord.read_coords(path_info)
line_id = th.coord.open_line_id(path_info)

# Determine the ratio the plots should have to keep the y-axis on the same scale
# as the x-axis
ratios = []
dhs = []
dxs = []
for line in lines:
    coords_sel = coords[line_id==int(line),:]
    
    dx_mat = th.coord.adapt_distances(path_info, line)
    bottom_idx = np.argwhere(dx_mat == dx_mat.max())[0][0]
    dists = dx_mat[bottom_idx,:]
    
    dh = coords_sel[:,2].max() #- coords_sel[:,0].min()
    dx = dists.max()
    # dhs.append(dh)
    dxs.append(dx)
    ratios.append(dh/dx)

ratio = max(ratios)

# Initialise the plot
fig, ax = plt.subplots(1,2,dpi=300,figsize=(20,3), gridspec_kw={'width_ratios': dxs})

# Go over each line and add the right plot
for i,line in enumerate(lines):
    
    # Select only the coordinates belonging to this line
    coords_sel = coords[line_id==int(line),:]
    
    # Get the distance along line from the bottom of the line
    dx_mat = th.coord.adapt_distances(path_info, line)
    bottom_idx = np.argwhere(dx_mat == dx_mat.max())[0][0]
    dists = dx_mat[bottom_idx,:]
    
    # Fit a line through the elevations
    coef, intercept = th.coord.fit_line(dists,coords_sel[:,2])
    # Predict the elevation
    # pred_elevs = dists*coef+intercept
    
    
    # Plot the results
    ax[i].scatter(dists,coords_sel[:,2])
    ax[i].plot([0,dx],[intercept,dx*coef+intercept], ls='dotted')
    
    # Set extra labels
    ax[i].set_title(f"Line {line}", fontsize=fsize)
    colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    ax[i].text(x_pos[i][0],x_pos[i][1],f'y={round(coef[0],3)}x+{round(intercept,1)}',c=colour, fontsize=fsize)
    ax[i].grid()
    ax[i].set_xlabel(f"Distance along line {line} [m]")
    ax[i].set_xlim([0,dxs[i]])
    ax[i].set_ylim([200,430])
    ax[i].set_ylabel("Elevation [m]")

# Setting the fontsize for all axs, can also unravel this to modify each part apart
for i in range(len(ax)):
    for item in ([ax[i].xaxis.label, ax[i].yaxis.label] +
                 ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        item.set_fontsize(fsize)