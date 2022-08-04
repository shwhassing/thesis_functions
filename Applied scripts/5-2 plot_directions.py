import glob
import numpy as np
import os
import obspy
import matplotlib.pyplot as plt
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.colors as colors

# Used to create images 5-2 and 5-5 in the thesis

# The master trace for the illumination analysis
master_trace = '7149'
# The component used for the illumination analysis
component = 'Z'
# The path where the results of the illumination analysis are found
path_res = "E:\\Thesis\\Arrays\\"
# Possible added string, ' - filtered' for the second analysis
added = ''

# Extract the results of the illumination analysis from the files
start_time, end_time, dom_slow0, dom_slow1 = th.proc.extract_results(path_res, 
                                                                     master_trace, 
                                                                     component,
                                                                     added)
# Convert the times to a format for matplotlib
times = th.proc.convert_date(start_time,'plt')

#%%

# At which apparent velocities the relevant boundaries are found (v=1/s)
vel_cut0 = 200 # The lowest velocity used in the velocity analysis
vel_cut1 = 5000 # The lower boundary for panels used in the crosscorrelations
vel_cut2 = 10000 # The lower boundary for panels used in the autocorrelations
fsize = 14 # Font size of the plot

# Mask for selected panels in the second plot
mask = th.proc.select_panels(dom_slow0,dom_slow1,vel_cut1)
dom_slow = np.stack([dom_slow0[mask],dom_slow1[mask]])

# Initiate the plot with two subplots
fig,ax = plt.subplots(1,2,figsize=(20,8),dpi=300)
# Set up the first plot
h1 = ax[0].hist2d(dom_slow1
                  ,dom_slow0,
                  bins=200,
                  cmap='OrRd',
                  range=[[-1/vel_cut0,1/vel_cut0],[-1/vel_cut0,1/vel_cut0]],
                  norm=colors.LogNorm())
rectangle = patches.Rectangle((-1/vel_cut1,-1/vel_cut1),
                              2/vel_cut1,
                              2/vel_cut1,
                              facecolor='none',
                              edgecolor='b')
ax[0].set_xlabel("Detected slowness along crossline [s/m]")
ax[0].set_ylabel("Detected slowness along main line [s/m]")
ax[0].add_patch(rectangle)

fig.colorbar(h1[3], ax=ax[0]).set_label(label="Amount",fontsize=fsize)

# Set up the second plot
h2 = ax[1].hist2d(dom_slow[1,:],
                  dom_slow[0,:],
                  bins=20,
                  cmap='OrRd',
                  range=[[-1/vel_cut1,1/vel_cut1],[-1/vel_cut1,1/vel_cut1]])
ax[1].ticklabel_format(axis='both',scilimits=(0,0))
ax[1].yaxis.offsetText.set_fontsize(fsize)
ax[1].xaxis.offsetText.set_fontsize(fsize)
ax[1].set_xlim([-1/vel_cut1,1/vel_cut1])
ax[1].set_ylim([-1/vel_cut1,1/vel_cut1])

ax[1].set_xlabel("Detected slowness along crossline [s/m]")
ax[1].set_ylabel("Detected slowness along main line [s/m]")
rectangle = patches.Rectangle((-1/vel_cut2,-1/vel_cut2),
                              2/vel_cut2,
                              2/vel_cut2,
                              facecolor='none',
                              edgecolor='g')
ax[1].add_patch(rectangle)
fig.colorbar(h2[3], ax=ax[1]).set_label(label="Amount",fontsize=fsize)

# Set the font size on multiple places
for axis in ax:
    for item in ([axis.xaxis.label, axis.yaxis.label] +
                  axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(fsize)