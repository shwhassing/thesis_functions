# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:00:01 2022

@author: Sverre Hassing
"""
import glob
import numpy as np
import os
import obspy
import matplotlib.pyplot as plt
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import matplotlib.dates as mdates

# File used to create figures 5-1 and 5-4 in the thesis. Uses the second cell,
# other cells are not used in the thesis itself

# The master trace of the illumination analysis
master_trace = '7149'
# The component for the illumination analysis
component = 'Z'
# The path to the results of the illumination analysis
path_res = "E:\\Thesis\\Arrays\\"
# A possible added string for the results. Can be '' for the original files and
# ' - filtered' for the second analysis
added_string = ' - filtered'
fsize = 16 # font size

# Extract the results of the illumination analysis from the files
start_time, end_time, dom_slow0, dom_slow1 = th.proc.extract_results(path_res, 
                                                                     master_trace, 
                                                                     component, 
                                                                     added_string)
# Convert the timestamps to a more useful format in matplotlib
times = th.proc.convert_date(start_time,'plt')

#%%
msize = 22 # marker size, does not really need to be changed

# Set up tick locations at every half day, but only a label at every day for
# the major grid
tick_locs = np.arange(np.floor(times[0]),np.ceil(times[-1]),0.5)
tick_labels = []
for date in tick_locs:
    if date % 1 == 0:
        datetime = mdates.num2date(date)
        date_str = f'{datetime.year}-{datetime.month}-{datetime.day}'
        tick_labels.append(date_str)
    else:
        tick_labels.append('')

# Start the plot
fig, ax = plt.subplots(dpi=300, figsize=(20,10))
ax.grid()
ax.grid(which='minor',c=(0.9,0.9,0.9)) # the minor grid is a bit lighter

# Apply the tick labels 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.set_zorder(0.5)
ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_labels)
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

# Plot the data
ax.scatter(times,dom_slow0, marker='*', s=msize, label='Main line',zorder=2.)
ax.scatter(times,dom_slow1, marker='*', s=msize, label='Crossline',zorder=2.01)
# Add lines around the noisy zone
ax.axvline(18832+15/24,c='lime',ls='--',zorder=2.5,lw=5)
ax.axvline(18834,c='lime',ls='--',zorder=2.5,lw=5)

# Set the axis labels
ax.set_xlabel('Date')
ax.set_ylabel("Slowness [s/m]")
ax.set_xlim([times[0]-0.1, times[-1]+0.1])
ax.set_ylim([-.005,.005])
ax.legend(fontsize=fsize)

# Setting the fontsize for all axs, can also unravel this to modify each part apart
for item in ([ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fsize)

plt.show()

#%%

# Histogram with slowness values for each line
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.hist([dom_slow0,dom_slow1], 20, label=['Main line', 'Crossline'])
ax.set_xlabel("Slowness [s/m]")
ax.set_ylabel("Amount")
ax.legend(fontsize = 12)
ax.grid()
for item in ([ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(12)
plt.show()

#%%

# Plot showing how many panels are available with each minimum apparent 
# velocity
vel_checks = np.linspace(10000,200,2400)
slow_checks = 1/vel_checks
dom_slow0_abs = abs(dom_slow0)
dom_slow1_abs = abs(dom_slow1)
amt_panels = len(dom_slow0)

amounts = []
for slow in slow_checks:
    amount = np.sum(np.logical_and(dom_slow0_abs <= slow, dom_slow1_abs <= slow))
    amounts.append(amount/amt_panels*100)
    # amounts.append(amount)
    
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.plot(vel_checks, amounts)
ax.set_xlim([0,max(vel_checks)])
ax.set_ylim([0,100])
ax.set_xlabel('Velocity [m/s]')
ax.set_ylabel('% panels used')
ax.grid()

#%%

# A moving average applied to the data. Play a bit with the window, the lines
# seem interesting

avg_window = 24*36
mov_avg0 = np.convolve(dom_slow0,np.ones(avg_window), 'same')/avg_window
mov_avg1 = np.convolve(dom_slow1,np.ones(avg_window), 'same')/avg_window
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.plot(times,mov_avg0, label='Main line')
ax.plot(times,mov_avg1, label='Crossline')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.set_xlabel("Date (UTC)")
ax.set_ylabel("Slowness [s/m]")
ax.grid()
ax.legend()

maxlim = 0.005
ax.set_ylim([-maxlim, maxlim])
ax.set_xlim(min(times), max(times))