# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:36:26 2022

@author: sverr
"""
import obspy
import numpy as np
import matplotlib.pyplot as plt
from thesis_functions.spectrogram_alt import spectrogram as spectr_alt
from thesis_functions.of import open_cont_record

def plot_spectrum(record, plot_max_f = None):
    """
    Plots the frequency spectrum of the first trace found in the provided Stream.
    Frequencies can be cut off at a maximum value.

    Parameters
    ----------
    record : obspy Stream object
        Stream containing the trace whose frequency spectrum will be plotted.
        The first trace is always used.
    plot_max_f : int or float, optional
        Maximum frequency that can be plotted. If no value is provided, the standard
        output is used. The default is None.

    Returns
    -------
    None.

    """
    # Calculate the frequency spectrum and the axis values.
    specgram, f, t = spectrogram(record[0])
    
    # If no maximum plotting frequency is provided, the maximum frequency is used
    if plot_max_f == None:
        plot_max_f = f[-1]
    
    # Some information on the record to make the plot title
    station = record[0].stats.station[1:]
    component = record[0].stats.channel[2]
    time_start = record[0].stats.starttime
    time_end = record[0].stats.endtime
    
    # Plot the figure
    plt.figure(figsize=(20,8),dpi=200)
    plt.imshow(specgram[np.where(f<=plot_max_f,True,False)], 
               origin = 'lower', 
               extent = [t[0],t[-1],f[0],f[np.where(f<plot_max_f,True,False)][-1]],
               aspect = 'auto'
               )
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title(f'Station {station}{component} from {time_start.datetime} to {time_end.datetime}')
    plt.colorbar()
    plt.show()
    
def compare_spectrum(records, plot_max_f = None, rounding_brackets = np.array([0,100,250,500,1000]), mult_factor = 5, fsize = 20, dpi = 200, size_fac = 10, fname = '', savefig = False):
    """
    Plotting function that plots the trace of the provided records next to their frequency spectrum.
    It adapts itself to the amount of records provided

    Parameters
    ----------
    records : list with Obspy Stream objects or Obspy Stream object
        A list containing all of the streams that need to be plotted or a single stream.
    plot_max_f : int or float, optional
        The maximum frequency that is plotted. If not provided, the base output is used. The default is None.
    rounding_brackets : array with ints or floats, optional
        The limits of the amplitude axis is set equal for all traces. To have the edges a nice
        distance away from the line, the max absolute amplitude is rounded up to the nearest multiple of
        one of the values in this array. 
        Which one is determined by the value of the max absolute amplitude
        The values in this array are multiplied by a mult_factor. If the max amplitude is bigger than a value 
        in the resulting array, the highest option is used for the rounding.
        The default is np.array([0,100,250,500,1000]).
    mult_factor : int or float, optional
        The rounding_brackets array is multiplied by this amount to determine the rounding amount. See the
        description of rounding_brackets. The default is 5.
    fsize : int, optional
        Font size for the figure for all text. The default is 20.
    dpi : int, optional
        Dots per inch to determine the figure resolution. The default is 200.
    size_fac : int or float, optional
        A factor to control the size of the figure. Size is dependent on the amount of records provided, if
        this is num_rec, then the size of the figure is: 
            3*size_fac by num_rec*2/3*size_fac
        Giving a default size of 30 by num_rec*6.6.
        The default is 10.
    fname : string
        Extra information to be added to the filename.
    savefig : boolean
        Whether or not to save the figure.

    Returns
    -------
    None.

    """
    # To determine the range of absolute amplitudes, the maximums of each record are saved in a list.
    ampl_range = []    
    
    print("Computing spectra...")
    
    # If records contains multiple streams in a list, the first block is used, if it is
    # a single stream, the second block is used.
    if isinstance(records,list):
        # Initialising arrays to contain the spectrum information and the range of values for the colormap
        spectrums = []
        freqs = []
        times = []
        c_range = np.zeros([len(records),2])
        
        # At first the frequency spectrum for each record is determined
        for i, rec in enumerate(records):
            specgram, f, t = spectrogram(rec[0])
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
        fig, axs = plt.subplots(len(records),2,
                                sharex=True, 
                                dpi=dpi, 
                                figsize = (3*size_fac,len(records)*2/3*size_fac), 
                                gridspec_kw={'width_ratios': [5,5]}
                                )
        
        # Now going through the records and plotting the right things
        for i,rec in enumerate(records):
            # First plot the trace itself
            plt.subplot(len(records),2,2*i+1)
            plt.plot(rec[0].times(),rec[0].data,c='black')
            plt.xlim([rec[0].times()[0], rec[0].times()[-1]])
            plt.xlabel("Time [s]")
            # The maximum absolute amplitudes are gotten to determine the limits later
            ampl_range.append(abs(min(rec[0].data)))
            ampl_range.append(abs(max(rec[0].data)))
            
            # Plot the frequency spectrum
            plt.subplot(len(records),2,2*(i+1))
            t = times[i]
            f = freqs[i]
            
            plt.imshow(spectrums[i][np.where(f<=plot_max_f,True,False)],
                       origin = 'lower', 
                       extent = [t[0],t[-1],f[0],f[np.where(f<plot_max_f,True,False)][-1]],
                       aspect = 'auto',
                       vmin = clims[0],
                       vmax = clims[1]
                       )
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
        
        # Figure out to which amount the limits are rounded by multiplying rounding_brackets by
        # mult_factor, then seeing when the maximum absolute amplitude from all traces is larger than
        # the values in the resulting array. The largest of these options is taken.
        rounding = rounding_brackets[max(np.where(max(ampl_range) >= rounding_brackets*mult_factor)[0])]
        # And actually rounding upwards to get the limits for the trace y-axes
        max_amp = np.ceil(max(ampl_range)/rounding)*rounding
        
        for i in range(len(records)):
            axs[i,0].set_ylim([-max_amp,max_amp])
                
    elif isinstance(records,obspy.core.stream.Stream):
        # Determining the spectrum for the single record
        specgram, f, t = spectrogram(records[0])
        # XXX Setting the colorbar limits is not really necessary if I just take the min and max
        clims = [specgram.min(), specgram.max()]
        
        print("Plotting...")
        
        # Setting up the plot
        fig,axs = plt.subplots(1,2,
                               sharex=True, 
                               dpi=dpi, 
                               figsize = (3*size_fac,2/3*size_fac), 
                               gridspec_kw={'width_ratios': [5,5]}
                               )
        # Plotting the trace
        plt.subplot(1,2,1)
        plt.plot(records[0].times(), records[0].data,c='black')
        plt.xlim(records[0].times()[0], records[0].times()[-1])
        plt.xlabel("Time [s]")
        ampl_range = [abs(min(records[0].data)), abs(max(records[0].data))]
        
        # Already setting the rounding, because there are no other records to worry about
        rounding = rounding_brackets[max(np.where(max(ampl_range) >= rounding_brackets*mult_factor)[0])]
        max_amp = np.ceil(max(ampl_range)/rounding)*rounding
        axs[0].set_ylim([-max_amp,max_amp])
        
        # Plotting the spectrum        
        plt.subplot(1,2,2)
        plt.imshow(specgram[np.where(f<=plot_max_f,True,False)],
                   origin = 'lower',
                   extent = [t[0],t[-1],f[0],f[np.where(f<plot_max_f,True,False)][-1]],
                   aspect = 'auto',
                   vmin = clims[0],
                   vmax = clims[1]
                   )
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
    else:
        # If no list or stream is provided, give an error
        raise TypeError("records must either be a list or an obspy Stream object")
    
    # Setting the fontsize for all axs, can also unravel this to modify each part apart
    for ax in axs.flatten():
        for item in ([ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fsize)
    
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
    
def average_spectrum(record, plot_max_f = None, ax = None, savefig = False, label = '', show = True):
    spec, f, t = spectrogram(record)
    if plot_max_f == None:
        plot_max_f = f[-1]
    
    spec = spec.sum(axis=1)/len(t)
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(10,7),dpi = 200)
    ax.plot(f, spec, label=label)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_xlim([0,plot_max_f])
    __, top_y = ax.get_ylim()
    ax.set_ylim([0,max([top_y,max(spec)])])
    ax.set_ylabel("Energy")
    ax.grid()
    ax.legend()
    
    time_start = record[0].stats.starttime
    time_end = record[0].stats.endtime
    station = record[0].stats.station[1:]
    component = record[0].stats.channel[2]
    
    if savefig:
        plt.savefig(f'./Images/Filtering/{time_start.year}-{time_start.month}-{time_start.day} {time_start.hour}-{time_start.minute}-{time_start.second} - {time_end.year}-{time_end.month}-{time_end.day} {time_end.hour}-{time_end.minute}-{time_end.second} {station}{component} Average spectrum')
    
    if show:
        plt.show()

def test_filter(record, method, **kwargs):
    """
    Testing different filters on a provided record and immediately plotting them.
    Different filtering methods are available. There is no other output than the plot.
    
    "low/high"
    Applies a lowpass and highpass Butterworth filter at the same corner frequency. 
    Parameters:
    freq : int or float
        Corner frequency for the filter
    
    Parameters
    ----------
    record : Obspy Stream object
        The record that will be filtered.
    method : string
        String determining which filtering method is used.
    **kwargs
        Depend on the filtering method used, see the description. Apart from these, there is:
            plot_max_f : int or float
                Maximum plotting frequency used for plotting the spectra

    Returns
    -------
    None.

    """
    # Setting up the records that must be plotted
    records = []
    records.append(record.copy())
    
    # This method applies a lowpass and highpass filter at the same corner frequency
    if method == "low/high":
        freq = kwargs['freq']
        rec_filt_low = record.copy().filter('lowpass',freq=freq,corners=5)
        records.append(rec_filt_low)
        
        rec_filt_high = record.copy().filter('highpass',freq=freq,corners=5)
        records.append(rec_filt_high)
    elif method == "bandstop":
        freq_ranges = kwargs['freq_ranges']
        
        rec_filt = record.copy()
        rec_filt_opp = record.copy()
        # To get the opposite filter start from 0 (or just above)
        last_f = 0.01
        # Go through all of the bandstop ranges
        for f_range in freq_ranges:
            rec_filt.filter('bandstop', freqmin = f_range[0], freqmax = f_range[1], corners = 4)
            rec_filt_opp.filter('bandstop', freqmin = last_f, freqmax = f_range[0], corners = 4)
            last_f = f_range[1]
        # Apply two more filters to the opposite filter to suppress the high
        # frequencies
        rec_filt_opp.filter('bandstop', freqmin = last_f, freqmax = last_f+30, corners = 8)
        rec_filt_opp.filter('bandpass', freqmin = 0.00001, freqmax = last_f+5, corners = 8)
        
        # And add the results
        records.append(rec_filt)
        records.append(rec_filt_opp)
    # Setting the default for plot_max_f
    plot_max_f = kwargs.get('plot_max_f',None)
    savefig = kwargs.get('savefig',False)
    fname = kwargs.get('fname','')
    
    # Plotting the results
    compare_spectrum(records, plot_max_f, fname = fname, savefig = savefig)
    
def spectrogram(record):
    """
    A very simple function that uses the spectr_alt function and inputs the 
    information found in the record. Saves some typing and is easier to read.
    spectr_alt is a slightly modified version of the spectrogram function in 
    obspy that returns the actual values of the spectrogram instead of only plotting
    them. 

    Parameters
    ----------
    record : obspy Stream object
        DESCRIPTION.

    Returns
    -------
    specgram : numpy array
        Matrix containing the spectrogram values. Time changes along the x-axis,
        frequency along the y-axis. 
    f : numpy array
        Frequency values used for the y-axis of the spectrogram
    t : numpy array
        Time values along the x-axis of the spectrogram.

    """
    if isinstance(record, obspy.Stream):
        return spectr_alt(record[0].data, record[0].stats.sampling_rate)
    elif isinstance(record, obspy.Trace):
        return spectr_alt(record.data, record.stats.sampling_rate)

def compare_avg_spectrum(station,time_ranges,component,base_path,labels = None, savefig = False, title = ""):
    """
    Calculates the spectrogram of a certain duration and then determines the 
    average over time. This gives a 2D plot, which might be easier to interpret
    for general features. Multiple time ranges can be given, which will be plotted
    as separate lines.

    Parameters
    ----------
    station : string
        The station number for which the record is opened.
    time_ranges : array [amt of ranges, 2]
        Array containing the time ranges for which the average is determined.
        Each time range gets a separate line. 
        First row along 0 axis contains the start times as would be put into 
        the function open_cont_record and the second row the end times. Durations 
        are not supported.
    component : string
        The component for which the record is opened.
    base_path : string / path
        The path to the base folder for the data files, see open_seis_file for 
        more information.
    labels : list, optional
        List with labels for the different time ranges. The default is None.
    savefig : bool, optional
        Whether or not to save the resulting plot to a file. The default is False.
    title : string, optional
        String with title for saving the plot, is applied at the end of the extra
        information. The default is "".

    Returns
    -------
    None.

    """
    # Initialise figure
    fig, ax = plt.subplots(figsize=(10,7), dpi = 200)
    
    # If there are no labels provided, set them to be empty
    if labels == None:
        labels = []
        for i in range(len(time_ranges)):
            labels.append('')
    
    # For each time range, set up the plot
    for i, time_range in enumerate(time_ranges):
        # First open the right file
        record = open_cont_record(station,time_range[0],component,base_path, time_end = time_range[1])
        # Then use average spectrum to add a line
        average_spectrum(record, 100, ax=ax, label=labels[i], show = False)
    
    # Save the figure if it is set to True
    if savefig:
        plt.savefig(f'./Images/Filtering/Compare spectra {station}{component} {title}')
    
    plt.show()

def apply_filters(record):
    
    f_ranges = []
    width = 2
    centres = [0,21,42,63,84]
    for centre in centres:
        f_ranges.append([max(0.0001,centre-width), centre+width])
    
    for f_range in f_ranges:
        record.filter('bandstop', freqmin = f_range[0], freqmax = f_range[1], corners = 4)
    
    return record