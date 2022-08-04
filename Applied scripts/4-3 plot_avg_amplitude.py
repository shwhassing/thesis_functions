# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:37:54 2022

@author: Sverre Hassing
"""

import os
import numpy as np
working_dir = os.path.normpath('H:\Onderwijs\TU Delft\\2-3 Master\'s thesis\\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import matplotlib.pyplot as plt
import glob
import obspy
from matplotlib.ticker import MultipleLocator

# This file was used to create figure 4-3 in the thesis

# base_path = os.path.normpath('G:/Seismic data Iceland 2021/Seismic data Iceland summer 2021/')
# path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')

# Path to the raw data
path_base = os.path.normpath('E:/Thesis/clip_data/')

stations = np.load('./Arrays/Stations.npy')
line_id = np.load('./Arrays/line_id.npy')

component = 'Z'

#%%

# This determines the average spectrum of 8 hours of data. It is a bit hacked
# together as the file takes LONG to process. The best is to let it run for a
# few hours and save the resulting array with
# np.save('./Arrays/Avg_amplitude.npy',specgram)

# Set up filters, comment it out inside the loop for the unfiltered average spectrum
f_ranges = [[0.0001,1]]
width = 2
centres = [21, 42, 63]
for centre in centres:
    f_ranges.append([centre - width, centre + width])

# The folders to look through for files, here capped for a single folder
folder_list = glob.glob(os.path.join(path_base,'*'))[4:5]

# Read a single file for some starting information
record = obspy.read(glob.glob(os.path.join(folder_list[0],f'*.{component}.mseed'))[0])
specgram,f,t = th.filt.spectrogram(record)
specgram = specgram.sum(axis=1)

for folder in folder_list:
    file_list = glob.glob(os.path.join(folder,f'*.{component}.mseed'))
    
    # Go through the files in the folder, here capped for 8 hours
    for file in file_list[:16]:
        # Read the file
        record = obspy.read(file)
        
        # Comment this out to get the unfiltered spectrum
        for f_range in f_ranges:
            record = record.filter('bandstop',
                                   freqmin=f_range[0],
                                   freqmax=f_range[1],
                                   corners=4)
        
        for trace in record:
            new_spec,f,t = th.filt.spectrogram(trace)
            specgram += new_spec.sum(axis=1)
        print(f'\r{file}',end="")
        
#%%

# Here, I load the filtered and unfiltered saved arrays
specgram_raw = np.load("./Arrays/Avg_amplitude2.npy")
specgram_filt = np.load("./Arrays/Avg_amplitude2_filt.npy")
plot_max_f = 100 # Maximum frequency to plot [Hz]

# Normalise both arrays with the unfiltered spectrum
specgram_filt = specgram_filt/specgram_raw.max()
specgram_raw = specgram_raw/specgram_raw.max()

# Ugly way to get the frequency axis of the data. The highest frequency is 
# based on the Nyquist frequency of the data. It is already defined if the 
# previous cell was ran
f = np.linspace(0,500,len(specgram_raw))

# Plot the two spectra
fig,ax = plt.subplots(figsize=(12,8),dpi=300)
ax.plot(f[f<=100],specgram_raw[f<=100],zorder=3)
ax.plot(f[f<=100],specgram_filt[f<=100],zorder=2)
ax.grid()
ax.set_xlim([f[0],f[f<=100][-1]])
ylims = ax.get_ylim()
ax.set_ylim([0,ylims[1]])
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(which='minor',c=(0.9,0.9,0.9))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Normalised amplitude")