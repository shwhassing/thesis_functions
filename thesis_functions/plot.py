# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:31:56 2023

@author: shwhassing
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from thesis_functions.proc import normalise_section, select_panels
from thesis_functions.util import stream_to_array

def dom_plot(record, **kwargs):
    """
    Simple plotting function that uses obspy.core.stream.Stream.plot() and 
    already sets some values to properly plot a seismic section. This means
    that the keyword arguments type, time_down, fillcolors and grid_color 
    cannot be used

    Parameters
    ----------
    record : obspy.core.stream.Stream
        The section to plot.
    **kwargs : TYPE
        The keyword arguments for the plotting routine.

    Returns
    -------
    None.

    """
        
    record.plot(type='section',
                time_down = True,
                fillcolors = ([0.5,0.5,0.5],None),
                grid_color='white',
                **kwargs)

def get_name_dist(dist_size):
    num_dist_list = [1,1000,0.3048,1609.344]
    name_dist_list = ['m','km','ft','mi']
    
    if isinstance(dist_size,float) or isinstance(dist_size,int):
        if dist_size in num_dist_list:
            return name_dist_list[num_dist_list.index(dist_size)], dist_size
        else:
            return f"m/{dist_size}", dist_size
    elif dist_size in name_dist_list:
        dist_size = num_dist_list[name_dist_list.index(dist_size)]
        return name_dist_list[num_dist_list.index(dist_size)], dist_size
    else:
        raise ValueError(f"{dist_size} is not a valid option for dist_size. Can be float, int or {name_dist_list}")
        
def plot_section_map(input_record,
                     figure=None,
                     figsize=(10,6),
                     dpi=300,
                     recordlength=None,
                     orient=None,
                     intsect=None,
                     intsect_len = 0.1,
                     fs=11,
                     dist_size=1,
                     save=False,
                     out_file=None,
                     **kwargs):
    """
    Function that plots a supplied stream as a seismic section with a 
    colourmap. All extra keyword arguments are supplied to the 
    matplotlib.pyplot.imshow function. 

    Parameters
    ----------
    input_record : obspy.core.stream.Stream
        DESCRIPTION.
    figsize : tuple, optional
        Size in inches of the resulting figure. The default is (10,6).
    dpi : float, optional
        Dots per inch of the resulting figure. The default is 300.
    recordlength : float, optional
        [s] Length of the record that is plotted. The default is None.
    orient : list, optional
        [A,B] List containing strings that are plotted at the top left and right
        of the plot to indicate the orientation of the line. String A is plotted
        left and B right. The default is None.
    intsect : float, optional
        [m] Intersection point of seismic lines indicated with red lines at the 
        top and bottom of the plot. A label with 'Intersection' is added at the
        bottom. A negative number can be supplied to count from the end of the
        line. The default is None.
    intsect_len : float, optional
        [inch] Length of the red lines used for the intersection line. Length
        required is based on the size of the plot, so can require some 
        experimentation. The default is 0.1.
    fs : float, optional
        Size of the font. The default is 11.
    dist_size : str or float, optional
        Conversion factor of the distances from metres used for the unit in the
        x-axis label. For example, if kilometres are used, dist_size is 1000. 
        For some values, it will get the name, or when a string is provided it 
        will fill in a number. Options are:
            1 - 'm'
            1000 - 'km'
            0.3048 - 'ft'
            1609.344 - 'mi'
        The default is 1.
    save : bool, optional
        Whether to save the plot. If True, a path for the new file needs to be
        given with out_file. The default is False.
    out_file : str, optional
        Path and filename of the output image if the plot is saved. The default
        is None.
    **kwargs
        Extra keyword arguments are given to the matplotlib.pyplot.imshow 
        function that plots the image.

    Raises
    ------
    ValueError
        If out_file is not defined, while save is True, so if no filename is 
        given while the image is saved an error is raised.

    Returns
    -------
    fig2, ax2 : matplotlib.pyplot.figure.Figure, matplotlib.pyplot.axes._subplots.AxesSubplot
        Returns the Matlotlib figure and axis used to create the plot

    """
    
    # Determine which unit of distance is used
    name_dist,dist_size = get_name_dist(dist_size)
    
    # Make a deep copy of the stream
    record = input_record.copy()
    
    # Enforce the record length, use the length of the first trace
    if recordlength == None:
        recordlength = input_record[0].stats.endtime - input_record[0].stats.starttime 
    record = record.trim(endtime=record[0].stats.starttime+recordlength)
    
    # Extract the distance information
    dists = []
    for trace in record:
        dists.append(trace.stats.distance/dist_size)
    dists = np.array(dists)
    
    # Normalise the section
    rec = normalise_section(record)
    
    # Convert to numpy array
    raw_data = stream_to_array(rec)
    
    # Take the maximum and minimum values
    maxima = [raw_data.max(),abs(raw_data.min())]
    
    t_max = record[0].stats.delta*(record[0].stats.npts-1)
    
    if figure == None:
        # Create the plot
        fig2,ax2 = plt.subplots(figsize=figsize,dpi=dpi)
    else:
        fig2,ax2 = figure
    
    # And plot the image
    ax2.imshow(raw_data.T,
               extent=[dists.min(),dists.max(),t_max,0],
               origin='upper',
               aspect='auto',
               # cmap='seismic',
               vmin=-np.max(maxima),
               vmax=np.max(maxima),
               **kwargs
               )
    
    # Add the intersection point of the two lines as red lines outside the plot
    if intsect != None:
        # If the supplied distance is negative, take it from the end of the
        # line
        if intsect < 0:
            intsect_point = dists.max() + intsect/dist_size
        else:
            intsect_point = intsect/dist_size
        
        # Set up the two lines
        line0 = Line2D([intsect_point,intsect_point],[0-intsect_len,0],color='r')
        line1 = Line2D([intsect_point,intsect_point],[t_max,t_max+intsect_len],color='r')
        
        # Add the two lines
        line0.set_clip_on(False)
        ax2.add_line(line0)
        
        line1.set_clip_on(False)
        ax2.add_line(line1)
        
        # Add a descriptor
        ax2.text(intsect_point,
                 t_max+1.7*intsect_len, 
                 "Intersection", 
                 color='r', 
                 rotation=0, 
                 horizontalalignment='center',
                 rotation_mode='anchor')
    
    # Add the orientation of the line above the plot, generally wind directions
    if orient != None:
        ax2.text(dists.min(), 
                 0,
                 orient[0],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 fontsize=1.6*fs)
        ax2.text(dists.max(), 
                 0,
                 orient[1],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 fontsize=1.6*fs)
    
    # Set labels
    ax2.set_ylabel('Two-way traveltime [s]')
    ax2.set_xlabel(f'Distance along line [{name_dist}]')
    
    # Enforce the same boundaries as for the wiggle plot
    max_dist = dists.max() - dists.min()
    ax2.set_xlim([dists.min()-0.05*max_dist, dists.max()+0.05*max_dist])
    
    # Set the ticks for both axes
    # XXX bit too hardcoded now
    ax2.xaxis.set_major_locator(MultipleLocator(250/dist_size))
    ax2.xaxis.set_minor_locator(MultipleLocator(50/dist_size))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    if save:
        if out_file == None:
            raise ValueError("File is saved without a filename")
        plt.savefig(out_file, transparent=True)
        
    return fig2,ax2
    
def plot_section_wiggle(input_record,
                        figsize=(10,6),
                        dpi=300,
                        lc=(0,0,0),
                        la=0.5,
                        lw=0.4,
                        fill_color=None,
                        orient=None,
                        intsect=None,
                        intsect_len = 0.1,
                        dist_size=1,
                        recordlength=None,
                        tr_scale = 1.,
                        fs=11,
                        save=False,
                        out_file=None):
    """
    Creates a wiggle plot of a section based on the supplied stream. Positive 
    amplitudes can be filled by a rgb colour supplied with fill_color.

    Parameters
    ----------
    input_record : obspy.core.stream.Stream
        Stream containing the seismic data. If no recordlength is supplied,
        the length of the first trace in the stream is taken as the length
        of the section
    figsize : tuple, optional
        Size of the figure in inches. The default is (10,6).
    dpi : float, optional
        Dots per inch of the figure. The default is 300.
    lc : tuple, optional
        RGB colour of the trace lines. The default is (0,0,0).
    la : float, optional
        Transparency (alpha) of the seismic traces. The default is 0.5.
    lw : float, optional
        Width of the trace lines. The default is 0.4.
    fill_color : tuple, optional
        RGB colour filling the space between the zero-point of each trace and
        positive amplitudes. Should be supplied as (r,g,b). The default is None.
    orient : list, optional
        [A,B] List containing strings that are plotted at the top left and right
        of the plot to indicate the orientation of the line. String A is plotted
        left and B right. The default is None.
    intsect : float, optional
        [m] Intersection point of seismic lines indicated with red lines at the 
        top and bottom of the plot. A label with 'Intersection' is added at the
        bottom. A negative number can be supplied to count from the end of the
        line. The default is None.
    intsect_len : float, optional
        [inch] Length of the red lines used for the intersection line. Length
        required is based on the size of the plot, so can require some 
        experimentation. The default is 0.1.
    dist_size : str or float, optional
        Conversion factor of the distances from metres used for the unit in the
        x-axis label. For example, if kilometres are used, dist_size is 1000. 
        For some values, it will get the name, or when a string is provided it 
        will fill in a number. Options are:
            1 - 'm'
            1000 - 'km'
            0.3048 - 'ft'
            1609.344 - 'mi'
        The default is 1.
    recordlength : float, optional
        [s] Length of the record that is plotted. The default is None.
    tr_scale : float, optional
        Amplitude scaling of the traces. The default is 1..
    fs : float, optional
        Font size on the plot. The default is 11.
    save : bool, optional
        Whether to save the plot. If True, a path for the new file needs to be
        given with out_file. The default is False.
    out_file : str, optional
        Path and filename of the output image if the plot is saved. The default
        is None.

    Raises
    ------
    ValueError
        If out_file is not defined, while save is True, so if no filename is 
        given while the image is saved an error is raised.

    Returns
    -------
    fig2, ax2 : matplotlib.pyplot.figure.Figure, matplotlib.pyplot.axes._subplots.AxesSubplot
        Returns the Matlotlib figure and axis used to create the plot

    """
    
    # Determine which unit of distance is used
    name_dist,dist_size = get_name_dist(dist_size)
    
    # Create a deep copy of the stream
    record = input_record.copy()
    
    # Enforce the record length, if it is not set, use length of the first trace
    if recordlength == None:
        recordlength = record[0].stats.endtime - record[0].stats.starttime
    record = record.trim(endtime=record[0].stats.starttime+recordlength)
    
    # Extract distance information
    dists = []
    for trace in record:
        dists.append(trace.stats.distance/dist_size)
    dists = np.array(dists)
    
    # Determine how much room is available for each trace, scaled by input parameter
    # tr_scale
    plot_ampl = (dists.max()-dists.min())/(1.5*(len(dists)))*tr_scale
    
    # Normalise each trace
    for trace in record:
        trace.data = trace.data/abs(trace.data).max()
    
    # Get the max value of the data, does nothing with the current normalisation
    # but is left in for others
    max_data = abs(np.array(record)).max()
    
    # Initialise the figure
    fig3,ax3 = plt.subplots(figsize=figsize,dpi=dpi)
    
    if intsect != None or orient != None:
        max_dist = dists.max()
    
    # Go over each trace
    for trace in record:
        data = trace.data
        # Get the location of this trace
        centre = trace.stats.distance/dist_size
        # Array with time information
        times = trace.times()
        
        # Normalise the data for the plot around the location of the trace
        plot_normalised_data = data/max_data*plot_ampl + centre
        # Plot the result
        ax3.plot(plot_normalised_data,
                 times,
                 color=lc,
                 lw=1.,
                 alpha=la)
        
        # Fill in the wiggles with the right colour
        if fill_color != None:
            ax3.fill_betweenx(times,
                              centre,
                              plot_normalised_data,
                              where=plot_normalised_data>centre,
                              facecolor=fill_color)
    
    # Add the intersection point as red lines with text
    if intsect != None:
        # If the supplied distance is negative, subtract it from the end of the
        # line
        if intsect < 0:
            intsect_point = max_dist + intsect/dist_size
        else:
            intsect_point = intsect/dist_size
        
        # Set up the lines above and below the figure
        line0 = Line2D([intsect_point,intsect_point],[times[0]-intsect_len,times[0]],color='r')
        line1 = Line2D([intsect_point,intsect_point],[times[-1],times[-1]+intsect_len],color='r')
        
        # Add the two lines
        line0.set_clip_on(False)
        ax3.add_line(line0)
        
        line1.set_clip_on(False)
        ax3.add_line(line1)
        
        # Add the text
        ax3.text(intsect_point,
                 times[-1]+1.7*intsect_len, 
                 "Intersection", 
                 color='r', 
                 rotation=0, 
                 horizontalalignment='center',
                 rotation_mode='anchor')
    
    # Add the orientation of the line as wind directions on top of the plot
    if orient != None:
        ax3.text(0, 
                 times[0],
                 orient[0],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 fontsize=1.6*fs)
        ax3.text(max_dist, 
                 times[0],
                 orient[1],
                 color='black',
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 fontsize=1.6*fs)
    
    # Set the limits of the time axis
    ax3.set_ylim([times[-1],times[0]])
    # Set labels
    ax3.set_xlabel(f"Distance along line [{name_dist}]")
    ax3.set_ylabel('Two-way traveltime [s]')
    
    # Set some space around the traces to be the same as the map plot
    max_dist = dists.max() - dists.min()
    ax3.set_xlim([dists.min()-0.05*max_dist, dists.max()+0.05*max_dist])
    
    # Set up the ticks
    ax3.xaxis.set_major_locator(MultipleLocator(250/dist_size))
    ax3.xaxis.set_minor_locator(MultipleLocator(50/dist_size))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    # Save the figure if requested
    if save:
        if out_file == None:
            raise ValueError("File is saved without a filename")
        plt.savefig(out_file, transparent=True)
        
    # Return the resulting figure
    return fig3,ax3

def histplot2d(dom_slow0, dom_slow1, vel_cut0, vel_cut1, vel_cut2, fsize):
    """
    Plot a square 2D histogram based on the provided slowness values

    Parameters
    ----------
    dom_slow0 : np.ndarray
        Slowness values along main line, used as y-axis.
    dom_slow1 : np.ndarray
        Slowness values along crossline, used as x-axis.
    vel_cut0 : float
        Apparent velocity used as minimum for illumination analysis.
    vel_cut1 : float
        Apparent velocity used as minimum for stacked image.
    vel_cut2 : float
        Apparent velocity used as minimum for zero-offset image.
    fsize : float
        Font size in plot.

    Returns
    -------
    None.

    """
    
    # Mask for selected panels in the second plot
    mask = select_panels(dom_slow0,dom_slow1,vel_cut1)
    dom_slow = np.stack([dom_slow0[mask],dom_slow1[mask]])
    
    # Initiate the plot with two subplots
    fig,ax = plt.subplots(1,2,figsize=(20,8),dpi=300)
    # Set up the first plot
    h1 = ax[0].hist2d(dom_slow1,
                      dom_slow0,
                      bins=200,
                      cmap='OrRd',
                      range=[[-1/vel_cut0,1/vel_cut0],[-1/vel_cut0,1/vel_cut0]],
                      norm=LogNorm())
    # Add the stacked image limits
    rectangle = plt.Rectangle((-1/vel_cut1,-1/vel_cut1),
                                  2/vel_cut1,
                                  2/vel_cut1,
                                  facecolor='none',
                                  edgecolor='b')
    ax[0].add_patch(rectangle)
    
    # Set up labels
    ax[0].set_xlabel("Dominant slowness along crossline [s/m]")
    ax[0].set_ylabel("Dominant slowness along main line [s/m]")

    # Plot colorbar
    fig.colorbar(h1[3], ax=ax[0]).set_label(label="Amount of panels",fontsize=fsize)
    
    # Set up the second plot
    h2 = ax[1].hist2d(dom_slow[1,:],
                      dom_slow[0,:],
                      bins=20,
                      cmap='OrRd',
                      range=[[-1/vel_cut1,1/vel_cut1],[-1/vel_cut1,1/vel_cut1]],
                      norm=LogNorm())
    # Set axes
    ax[1].ticklabel_format(axis='both',scilimits=(0,0))
    ax[1].yaxis.offsetText.set_fontsize(fsize)
    ax[1].xaxis.offsetText.set_fontsize(fsize)
    ax[1].set_xlim([-1/vel_cut1,1/vel_cut1])
    ax[1].set_ylim([-1/vel_cut1,1/vel_cut1])
    # and labels
    ax[1].set_xlabel("Dominant slowness along crossline [s/m]")
    ax[1].set_ylabel("Dominant slowness along main line [s/m]")
    
    # Add zero-offset image limits
    rectangle = plt.Rectangle((-1/vel_cut2,-1/vel_cut2),
                                  2/vel_cut2,
                                  2/vel_cut2,
                                  facecolor='none',
                                  edgecolor='g')
    ax[1].add_patch(rectangle)
    # Plot the colorbar
    fig.colorbar(h2[3], ax=ax[1]).set_label(label="Amount of panels",fontsize=fsize)
    
    # Set the font size on multiple places
    for axis in ax:
        for item in ([axis.xaxis.label, axis.yaxis.label] +
                      axis.get_xticklabels() + axis.get_yticklabels()):
            item.set_fontsize(fsize)
    plt.plot()
    
def get_2dhistogram(x_data, x_lims, x_bins, y_data, y_lims, y_bins):
    """
    Get the histogram data 

    Parameters
    ----------
    x_data : np.ndarray
        X-values of the data.
    x_lims : np.ndarray
        Limits of the bins along the x-axis.
    x_bins : int
        Amount of bins along the x-axis.
    y_data : np.ndarray
        Y-values of the data.
    y_lims : np.ndarray
        Limits of the bins along the y-axis.
    y_bins : int
        Amount of bins along the y-axis.

    Returns
    -------
    hist : np.ndarray
        2D array containing the histogram counts for each bin.
    X : np.ndarray
        2D array containing the x-coordinates of each bin.
    Y : np.ndarray
        2D array containing the y-coordinates of each bin.

    """
    
    # Define the binning
    xbins = np.linspace(x_lims[0],x_lims[1], x_bins)
    ybins = np.linspace(y_lims[0],y_lims[1], y_bins)
    
    # Get the histogram and mesh
    hist, __, __ = np.histogram2d(x_data, y_data, bins=(xbins,ybins))
    X, Y = np.meshgrid(xbins, ybins)
    
    return hist, X, Y

def polar_histplot2d(azims, lengths, vel_cut0, vel_cut1, vel_cut2, rmax = None, n_abins = 360, n_lbins = 1200, alims = [-np.pi,np.pi], fsize=12, use_circle = False):
    """
    Get a 2D histogram as a polar plot. Data must be defined in polar coordinates,
    so containing an azimuth and length. As this function is meant to plot
    slowness data, specific limits for the selection of noise panels must be 
    defined with vel_cut0 to 2. 

    Parameters
    ----------
    azims : np.ndarray
        Array containing the azimuth of each data point.
    lengths : np.ndarray
        Array containing the length of each data point.
    vel_cut0 : float
        Apparent velocity used as a minimum for the illumination analysis.
    vel_cut1 : float
        Apparent velocity used as a minimum to select panels for the virtual
        common-shot gathers.
    vel_cut2 : float
        Apparent velocity used as a minimum to select panels for the zero-offset
        section.
    rmax : float, optional
        Maximum length used for the plot As a default, the maximum from the 
        data is used. The default is None.
    n_abins : int, optional
        Number of azimuth bins used for the plot. The default is 360.
    n_lbins : int, optional
        Number of length bins used for the plot. The default is 1200.
    alims : list, optional
        Limits of azimuth used for the plot. The default is [-np.pi,np.pi].
    fsize : float, optional
        Font size. The default is 12.
    use_circle : bool, optional
        Whether to plot the selection criteria as a circle (True) or as a 
        square (False). The default is False.

    Returns
    -------
    None.

    """
    
    # If the maximum slowness is not set, use max value of illumination analysis
    if rmax == None:
        rmax = 1/vel_cut0
    
    # Set up the histogram data
    hist, A, R = get_2dhistogram(azims, [alims[0], alims[1]], n_abins,
                                 lengths, [0,rmax], n_lbins)
    
    # Set up the figure as polar plot
    fig, ax = plt.subplots(1,2,subplot_kw=dict(projection="polar"),dpi=300,figsize=(11,5))
    fig.tight_layout(pad=2)
    
    # Set the ticks and labels of the first plot
    ax[0].set_xticks(np.arange(0,2*np.pi,0.25*np.pi))
    ax[0].set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])
    ax[0].set_yticks(np.arange(0,1/vel_cut0,0.001))
    ax[0].set_rlabel_position(85)
    ax[0].grid(False)
    
    # Put the radial axis label outside the plot
    pos1 = ax[0].get_rlabel_position()
    ax[0].text(np.radians(pos1-6),1.04*rmax,'Slowness [s/m]')
    
    # Set the limits of the stacked section
    if use_circle:
        circle_cross = plt.Circle((0,0), 
                                  1/vel_cut1, 
                                  transform=ax[0].transData._b, 
                                  fill=False,
                                  color='blue',
                                  zorder = 5)
        ax[0].add_artist(circle_cross)
    else:
        rect_cross = plt.Rectangle((-1/vel_cut1,-1/vel_cut1),
                                   2/vel_cut1,
                                   2/vel_cut1,
                                   transform=ax[0].transData._b,
                                   facecolor='none',
                                   edgecolor='b')
        ax[0].add_artist(rect_cross)
    
    # Plot the histogram
    pc = ax[0].pcolormesh(A, R, hist.T, cmap="magma_r",norm=LogNorm())
    # Set up the colorbar
    cbar1 = fig.colorbar(pc, ax=ax[0], pad=0.08, fraction = 0.046)
    cbar1.set_label(label="Amount of panels",fontsize=fsize)
    ax[0].grid(True)
    
    # Now set up the second histogram
    hist2, A2, R2 = get_2dhistogram(azims[lengths<=1/vel_cut1],  [alims[0], alims[1]], 90,
                                    lengths[lengths<=1/vel_cut1], [0,1/vel_cut1], 20)
    
    # Do the axis labels
    ax[1].set_xticks(np.arange(0,2*np.pi,0.25*np.pi))
    ax[1].set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])
    ax[1].set_yticks(np.arange(0,1/vel_cut1,1/(4*vel_cut1)))
    
    normalised_ytickmax = 1/10**(np.floor(np.log10(1/vel_cut1)))/vel_cut1
    ax[1].set_yticklabels(np.arange(0,normalised_ytickmax,normalised_ytickmax/4))
    ax[1].set_rlabel_position(85)
    
    # And the radial axis label
    pos2 = ax[1].get_rlabel_position()
    ax[1].text(np.radians(pos2 - 6),1.04/vel_cut1,'Slowness [1ᴇ-4 s/m]')
    
    # Set the limits of the zero-offset image
    if use_circle:
        circle_auto = plt.Circle((0,0), 
                                  1/vel_cut2, 
                                  transform=ax[1].transData._b, 
                                  fill=False,
                                  color='green',
                                  lw=1,
                                  zorder = 5)
        ax[1].add_artist(circle_auto)
    else:
        rect_auto = plt.Rectangle((-1/vel_cut2,-1/vel_cut2),
                                   2/vel_cut2,
                                   2/vel_cut2,
                                   transform=ax[1].transData._b,
                                   facecolor='none',
                                   edgecolor='green')
        ax[1].add_artist(rect_auto)
    
    ax[1].grid(False)
    
    # Plot the histogram
    pc2 = ax[1].pcolormesh(A2, R2, hist2.T, cmap='magma_r',norm=LogNorm())
    # Do the colorbar
    cbar2 = fig.colorbar(pc2, ax=ax[1], pad=0.08, fraction=0.046)
    cbar2.set_label(label="Amount of panels",fontsize=fsize)
    cbar2.ax.set_yticks([1,2,3,4])
    cbar2.ax.set_yticklabels(['1','2','3','4'])
    
    ax[1].grid(True)
    plt.show()
    
def plot_avg_spectrum(record, plot_max_f = 60):
    """
    Function that plots the absolute frequency spectrum of all traces in a 
    stream and then the average on top of it.

    Parameters
    ----------
    record : obspy.Stream
        Stream containing the traces for which the frequency spectrum is plotted.
        It is assumed that all traces have the same time step and amount of 
        samples

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis on which the plot is drawn.

    """
    # Compute the Fourier transform, take the positive frequencies and then the absolute value
    spectrum =  np.abs(np.fft.rfft(stream_to_array(record), axis=1))
    # Get the frequencies belonging to this spectrum
    freq = np.fft.rfftfreq(record[0].stats.npts, record[0].stats.delta)
    
    fig, ax = plt.subplots(dpi=300,figsize=(10,6))
    # Plot the spectrum of each trace separately
    ax.plot(freq, spectrum.T, alpha = 0.1, c='r')
    # Plot the average spectrum
    ax.plot(freq, np.average(spectrum.T, axis=1), alpha=1, c='black')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    ax.set_xlim([0,plot_max_f])
    ylim = ax.get_ylim()
    ax.set_ylim([0,ylim[1]])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid()
    
    return fig, ax