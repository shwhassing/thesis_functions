# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:17:18 2022

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

# File used to resort CSP gathers into CMP gathers and save some necessary
# distance information

# Which line to use
line = '1'
# The binning distance from the CMP location [m]
tolerance = 5

# Path to coordinate information
path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')
# Path to deconvolved virtual CSP gathers
path_shot = os.path.normpath('E:\\Thesis\\Arrays\\Crosscorr 5000 - decon\\')
# Output path where CMP gathers are stored
path_out = os.path.normpath('E:\\Thesis\\Arrays\\CMP 5000\\')


#%% Open all CSG on line

# Find all common shot gathers on this line
file_list = glob.glob(os.path.join(path_shot,line,'*.mseed'))

# Open them and attach their distance to virtual source point
streams = []
for i,file in enumerate(file_list):
    stream = obspy.read(file)
    stream = th.coord.attach_distances(stream,i,line,path_info)
    streams.append(stream)
    
#%% Set up CMP locations

# Get the distance from the bottom of the line to each receiver
dx_mat = th.coord.adapt_distances(path_info, line)
edges = th.coord.find_outer_stats(line, path_info)
distances = dx_mat[edges[0]]

# Initialise arrays
midpoints = np.zeros([len(distances),len(distances)])
offsets = np.zeros([len(distances),len(distances)])

# Go over each common shot gather
for i,stream in enumerate(streams):
    # And then each trace in there
    for j,trace in enumerate(stream):
        # Determine the offset (half of the distance to the source point)
        offsets[i,j] = trace.stats.distance/2
        # Determine the distance of the midpoint along the line 
        # Same as the distance of the station along the line + the offset
        midpoints[i,j] = offsets[i,j] + distances[i]

# The CMP points that will be used are set up
new_mps = np.arange(tolerance,distances.max(),2*tolerance)

# And some of the information is saved
np.save(f'./Arrays/CMP_locs{line}',new_mps)
np.save(f'./Arrays/Midpoints{line}',midpoints)
np.save(f'./Arrays/offsets{line}',offsets)


#%% Resort CSP into CMP gathers

# Now set up gathers for each CMP location
cmp_gathers = []
for new_mp in new_mps:
    # Make a new stream
    CMP_stream = obspy.Stream()
    
    # Find every midpoint that falls within the binning distance of the new
    # midpoint location
    test = np.logical_and(
        midpoints >= new_mp - tolerance,
        midpoints <  new_mp + tolerance
        )
    
    # Go through every CSP gather
    for i,stream in enumerate(streams):
        # And every trace
        for j,trace in enumerate(stream):
            # And if they fall within the binning distance
            if test[i,j]:
                # Add their offset and add them to the gather
                trace.stats.distance = offsets[i,j]
                CMP_stream += trace
    # The full gather is then added to the list
    cmp_gathers.append(CMP_stream)

#%% Save offset information

# Get the maximum amount of traces in a CMP gather
max_fold = 0
for stream in cmp_gathers:
    max_fold = max(max_fold,len(stream))

# Initialise an array that will contain the offset of each trace in their 
# gather
offset_mat = np.zeros([len(new_mps),max_fold])

# Now go over each binned midpoint
for k,new_mp in enumerate(new_mps):
    # See how many traces there are in the stream
    trc_in_strm = 0
    
    # See which traces fall within the binning range
    test = np.logical_and(
        midpoints >= new_mp - tolerance,
        midpoints <  new_mp + tolerance
        )
    
    # Go over the same loop as for the resorting, but now save the offset
    for i,stream in enumerate(streams):
        for j,trace in enumerate(stream):
            if test[i,j]:
                offset_mat[k,trc_in_strm] = offsets[i,j]
                trc_in_strm += 1

# Save the result
np.save(f'./Arrays/offset_mat{line}',offset_mat)

#%% Save results

# Check if the output folder exists and make it otherwise
if not os.path.isdir(path_out):
    os.mkdir(path_out)

# Add the line id to the path
path_full = os.path.join(path_out,line)

# Check if the output folder exists and make it otherwise
if not os.path.isdir(path_full):
    os.mkdir(path_full)

# Go over each CMP gather
for i,stream in enumerate(cmp_gathers):
    # Determine the filename
    filename = f'CMP {int(np.round(new_mps[i]))}.mseed'
    
    # And save the result as a .mseed
    stream.write(os.path.join(path_full,filename))

#%%

# The fold can be plotted for QC
plt.figure(dpi=300)
plt.hist(midpoints.flatten(),range=[0,distances.max()],bins=int(distances.max()/10),zorder=4)
plt.xlabel('CMP location [m]')
plt.ylabel('Fold')
plt.ylim([0,100])
plt.grid(zorder=10)