# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:28:48 2022

@author: sverr
"""
import numpy as np
import os
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\Scripts')
os.chdir(working_dir)
# import thesis_functions.TauP as TauP
# import thesis_functions.open_files as of
import matplotlib.pyplot as plt
import thesis_functions as th
import obspy

# Input values

amt_stations = 500

dx = 2 # Distance between stations [m]
dt = 1/10000 # Time step of the data [s]
# Info for the lines and hyperbolae in the order
# starttime, velocity, amplitude
info_lines = [[0.05,1100,0.9],[0.1,1000,1.0],[0.2,900,1.4],[0.4,800,0.6]]
info_hyperbolae = [[0.15,900,1.3],[0.5,1100, 1.6]]
depth = 100

ricker_width = 50 # Width of the ricker wavelet in elements, multiply by dt for time
record_length = 1.5 # Total length of the record

# The evaluated slowness values in p_range
min_vel = 200 # minimum velocity [m/s]
amt_p_vals = 1200 # amount of slowness values tested, see it as the resolution
mtr_idx = 0
p_range = np.linspace(-1/min_vel,1/min_vel,amt_p_vals)

#%%

# Calculate the location of each station
distances = np.linspace(0,amt_stations,amt_stations)*dx
amt_samples = int(record_length / dt) + 1 

# Initiate the data array
test_linear = np.zeros([amt_stations,amt_samples])

# Add some data to the array
for start_time, velocity, amplitude in info_lines:
    test_linear = th.TauP.add_line(test_linear, distances, dt, ricker_width, start_time, velocity, amplitude)
for start_time, velocity, amplitude in info_hyperbolae:
    test_linear = th.TauP.add_hyperbola(test_linear, distances, dt, ricker_width, start_time, depth, velocity, amplitude)

v_lims = max(abs(test_linear.max()),abs(test_linear.min()))

# Plot the data
plt.figure(figsize=(8,8),dpi=300)
plt.imshow(test_linear.T,
           aspect='auto',
           origin='upper',
           extent=[0,dx*amt_stations,record_length,0],
           cmap='seismic',
           vmin=-v_lims,
           vmax=v_lims
           )
plt.ylabel('Time [s]')
plt.xlabel("Distance [m]")
plt.show()


#%%

# Compute the Tau-P transform (also for a single slice to test if that function
# works)

taup_t0 = th.TauP.TauP_slice(test_linear, p_range, distances, dt, (np.shape(test_linear)[1] - 1)/2)

taup_test = th.TauP.TauP_batch(test_linear, p_range, distances, dt)

#%%

v_lims = max(abs(taup_test.max()),abs(taup_test.min()))

# Plot the results of the Tau-P transform
plt.figure(figsize=(8,8),dpi=300)
plt.imshow(taup_test[600:840,:].T,
            aspect='auto',
            origin = 'upper',
            extent = [p_range[600], p_range[840], record_length, 0],
            cmap='seismic',
            vmin=-v_lims,
            vmax=v_lims
            )
plt.xlabel("Slowness [s/m]")
plt.ylabel("Time [s]")