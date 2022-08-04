# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:57:58 2022

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

# File used to create figure 4-3 in the thesis

components = ['N','Z','E']

# Open the record for all three components
record = obspy.Stream()
for component in components:
    file = f'E:/Thesis/clip_data/26/2021.07.26.18.30.00.{component}.mseed'
    record += obspy.read(file)

#%%

# Trim the record to the right time
start=obspy.UTCDateTime('2021-07-26T18:54:30')
end=obspy.UTCDateTime('2021-07-26T18:55:30')

# Select only a single station
new_rec = record.select(station='07149')
new_rec = new_rec.trim(starttime=start,endtime=end)

# Add the traces to a list instead for the function
records = []

for trace in new_rec:
    temp = obspy.Stream()
    temp += trace
    records.append(temp)
    
del temp

#%%

# As an alternative, only select a single component, used in the final thesis
records = [new_rec.select(component='Z')]

#%%

# Set the maximum frequency for plotting
plot_max_f = 100

# Plotting parameters

# To what values the limits are rounded for the shown trace
rounding_brackets = np.array([0,100,250,500,1000,10000,100000]) 
mult_factor = 5 # 
fsize = 20 # font size
dpi = 200 # dots per inch of the image
size_fac = 10 # A size factor for the image
fname = '' # Name of an output plot
savefig = False # Whether to actually save the plot

# Labels of the plot in the order of the list records. Remove the 'Z' if you 
# use all components
ylabels = ['Z','N','Z','E']


# Setting up some values for the plotting
ampl_range = []
avg_max = 0

print("Computing spectra...")

# Initialising arrays to contain the spectrum information and the range of values for the colormap
spectrums = []
freqs = []
times = []
c_range = np.zeros([len(records),2])

# At first the frequency spectrum for each record is determined
for i, rec in enumerate(records):
    specgram, f, t = th.filt.spectrogram(rec[0])
    spectrums.append(specgram)
    freqs.append(f)
    times.append(t)
    c_range[i,:] = [specgram.min(),specgram.max()]

# If no maximum frequency is set, the maximum frequency from the last record is used
if plot_max_f == None:
    plot_max_f = f[-1]
# The limits of the colorbar are set the same for each record by taking the min and max
# for all spectra
clims = [c_range[:,0].min(), c_range[:,1].max()]

print("Plotting...")

# Initialising a plot with on the left column the traces, on the right the spectrum
fig, axs = plt.subplots(len(records),3,
                        dpi=dpi, 
                        sharex='col',
                        figsize = (3*size_fac,len(records)*2/3*size_fac), 
                        gridspec_kw={'width_ratios': [5,5,3]}
                        )
axs = axs[np.newaxis,:]

# Now going through the records and plotting the right things
for i,rec in enumerate(records):
    spectrum = spectrums[i]
    
    # First plot the trace itself
    axs[i,0].plot(rec[0].times(),rec[0].data,c='black')
    axs[i,0].set_xlim([rec[0].times()[0], rec[0].times()[-1]])
    axs[len(records)-1,0].set_xlabel("Time [s]")
    axs[i,0].set_ylabel(ylabels[i], rotation=0)

    # The maximum absolute amplitudes are gotten to determine the limits later
    ampl_range.append(abs(min(rec[0].data)))
    ampl_range.append(abs(max(rec[0].data)))
    
    # Plot the frequency spectrum
    t = times[i]
    f = freqs[i]
    
    # Plot the spectrum
    axs[i,1].imshow(spectrum[np.where(f<=plot_max_f,True,False)],
               origin = 'lower', 
               extent = [t[0],t[-1],f[0],f[np.where(f<plot_max_f,True,False)][-1]],
               aspect = 'auto',
               vmin = clims[0],
               vmax = clims[1]
               )
    axs[len(records)-1,1].set_xlabel("Time [s]")
    axs[i,1].set_ylabel("Frequency [Hz]")
    axs[i,1].set_xlim([rec[0].times()[0], rec[0].times()[-1]])
    
    avg_spectrum = spectrum.sum(axis=1)/spectrum.shape[1]
    avg_max = max(avg_max,avg_spectrum.max())
    axs[i,2].plot(f[f<=plot_max_f],avg_spectrum[f<=plot_max_f])
    axs[i,2].set_xlim(f[0],plot_max_f)
    axs[len(records)-1,2].set_xlabel("Frequency [Hz]")
    axs[i,2].grid(c=(0.6,0.6,0.6))

# Figure out to which amount the limits are rounded by multiplying rounding_brackets by
# mult_factor, then seeing when the maximum absolute amplitude from all traces is larger than
# the values in the resulting array. The largest of these options is taken.
rounding = rounding_brackets[max(np.where(max(ampl_range) >= rounding_brackets*mult_factor)[0])]
# And actually rounding upwards to get the limits for the trace y-axes
max_amp = np.ceil(max(ampl_range)/rounding)*rounding

for i in range(len(records)):
    axs[i,0].set_ylim([-max_amp,max_amp])
    axs[i,2].set_ylim([0,avg_max])

# Setting the fontsize for all axs, can also unravel this to modify each part apart
for ax in axs.flatten():
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)
        
for ax in axs[:,0]:
    ax.yaxis.label.set_fontsize(2*fsize)

# Getting some information for the title
station = records[0][0].stats.station[1:]
component = records[0][0].stats.channel[2]
time_start = records[0][0].stats.starttime
time_end = records[0][0].stats.endtime

# Setting the title
fig.suptitle(f"Station {station}{component} from {time_start.datetime} to {time_end.datetime}", fontsize = 2*fsize)
if savefig:
    print("Saving figure...")
    plt.savefig(f'./Images/Filtering/{time_start.year}-{time_start.month}-{time_start.day} {time_start.hour}-{time_start.minute}-{time_start.second} - {time_end.year}-{time_end.month}-{time_end.day} {time_end.hour}-{time_end.minute}-{time_end.second} {station}{component} {fname}')

plt.show()