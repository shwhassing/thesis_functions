# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:41:52 2022

@author: Sverre
"""

import numpy as np
from scipy.signal import ricker#, correlate
# from os.path import join, split
import obspy
# import csv
# from thesis_functions.of import open_cont_record, open_diff_stat
# from thesis_functions.coord import calcCoord, open_line_id
from thesis_functions.util import cross_corr, tfs_string
from thesis_functions.proc import normalise_trace
# import matplotlib.pyplot as plt

############## Core functions

def TauP_batch(data, p_range, distances, dt):
    """
    Calculates the Tau-P transform (see the function TauP_calc) with raw data.
    This method performs the whole dataset in one batch, so all stations need
    to be included. 

    Parameters
    ----------
    data : array [amt stations, amt samples]
        Array containing the recorded data as in a reflection survey. For the
        noise data, this means that autocorrelations have to done first.
    p_range : array [amt slowness vals, 1]
        Array containing all of the slowness values that are evaluated.
    distances : array [amt stations, 1]
        Array containing the distance from each station to the master trace. 
        This array should be sorted in the same order as the data is.
    dt : float
        Time between each sample.

    Returns
    -------
    taup_data : array [amt slowness vals, amt samples]
        Tau-P transform of the data for the specified slowness values.

    """
    # Initialise the data
    taup_data = np.zeros([len(p_range), data.shape[1]])
    
    # Calculate for each slowness value
    for i, p in enumerate(p_range):
        # Determine the linear move-out shifts in time units
        shift = np.around(distances * p / dt)
        
        data_shifted = np.zeros(np.shape(data))
        # Now for each record, shift the data by the right amount
        for j in range(data.shape[0]):
            data_shifted[j,:] = shift_row(data[j,:], -shift[j], 0)
        
        # Sum up the data to get a new row of the Tau-P data
        taup_data[i,:] = np.sum(data_shifted, axis = 0)
        
    return taup_data

def TauP_add(data, p_range, distance, dt):
    """
    Calculates the contribution of a single station to the Tau-P transform 
    (see also the function TauP_calc). This allows the program to use less 
    memory. The contributions from every stations should be added up to get the
    full transform.

    Parameters
    ----------
    data : array [amt samples, 1]
        The data recorded at this station. Should be as a reflection gather,
        for noise data this means cross-correlation with the master trace.
    p_range : array [amt slowness vals, 1]
        Array containing all of the slowness values that are evaluated.
    distance : float
        Distance from the station to the master trace.
    dt : TYPE
        Time between each sample.

    Returns
    -------
    data_contr : array [amt slowness vals, amt samples]
        The contribution of this station to the Tau-P transform.

    """
    # The normal move-out shift of the data for the distance and each slowness
    # value. This is converted to an index shift
    shifts = np.around(distance * p_range / dt)

    # Initialising the data
    data_contr = np.zeros([len(p_range), len(data)])
    # Shifting the data by each value in shift and then adding that to the right
    # row
    for i in range(data_contr.shape[0]):
        data_contr[i, :] += shift_row(data, -shifts[i], 0)

    return data_contr

def shift_row(array, amount, new_vals = None):
    """
    Function that gets a 1D array and shifts the values by a certain amount along
    the array axis. This means that new values are added on one of the sides.
    These are None defaultly, but can be set to other values.

    Parameters
    ----------
    array : numpy array [N,1]
        Array which should be shifted.
    amount : int
        By how many locations the values should be shifted.
    new_vals : any type that fits in numpy arrays, optional
        When the values are shifted, new values are added, which value they
        have is decided by this parameter. The default is None.

    Returns
    -------
    new_array : array [N,1]
        The shifted version of the array, has the same size as the original.

    """
    # Ensure amount is an integer
    amount = int(amount)
    
    # Make a deep copy of the input array to put the changes into
    new_array = array.copy()
    # Different indexing is needed for positive and negative shifts
    if amount > 0:
        new_array[amount:] = array[:-amount]
        new_array[:amount] = new_vals
    elif amount < 0:
        new_array[:amount] = array[-amount:]
        new_array[amount:] = new_vals
    # In the case that amount is zero, the original array can be provided back
    return new_array

def TauP_slice(data, p_range, distances, dt, slice_idx):
    """
    Determine the Tau-P transform for a single intercept (tau) value. This 
    intercept is determined by the slice_idx. 

    Parameters
    ----------
    data : np.ndarray
        Array containing the crosscorrelated panel on which the Tau-P transform
        is applied.
    p_range : np.ndarray
        Array containing the slowness values that are tested.
    distances : np.ndarray
        Array containing the distance of each station to the master station.
    dt : float
        Time step of the data.
    slice_idx : int
        Index indicating at which position the Tau-P transform is used.

    Returns
    -------
    values : np.ndarray
        The resulting sum of amplitudes for each slowness value at the 
        specified intercept value.

    """
    # Determine the linear move-out shift for each trace
    shifts = np.around(distances[:,np.newaxis] * p_range[np.newaxis,:] / dt)
    
    # We will index each location twice to get the result
    
    # First an index for time is created by determining the actual move-out 
    # from the starting position
    index_time = (shifts + slice_idx).astype(int)
        
    # Determine indices that fall out of the range of the data
    nt = np.shape(data)[1]
    test1 = index_time >= nt
    test2 = index_time < 0
    
    # Create a mask for them and set them to zero as a placeholder
    mask_zeroes = np.logical_or(test1, test2)
    index_time[mask_zeroes] = 0
    
    # As the second index use the indices of each station and repeat for 
    # every slowness value
    index_space = (np.arange(0, np.shape(data)[0])[:,np.newaxis] + np.zeros(len(p_range))[np.newaxis,:]).astype(int)
        
    # Index the data at the two indices to get the LMO corrected data
    values = data[index_space, index_time]
    # Set the right values to zero
    values[mask_zeroes] = 0
    # And sum over the first axis to get the Tau-P transform
    values = values.sum(axis=0)
    
    return values

############## Calculation functions

def process_window(window,
                   amt_stations,
                   lines,
                   mtr_idx,
                   window_length,
                   p_range,
                   distances,
                   line_id):
    """
    Process a single window for the illumination analysis. 

    Parameters
    ----------
    window : obspy.core.stream.Stream
        The noise panel as a stream.
    amt_stations : int
        How many stations there should be in a complete panel.
    lines : list
        A list containing the possible line identifiers.
    mtr_idx : int
        The index of the master trace in window.
    window_length : float
        The length of the noise panel in seconds.
    p_range : np.ndarray
        Array containing all slowness values that should be evaluated [s/m]
    distances : np.ndarray
        Array containing the distance to the master station.
    line_id : np.ndarray
        Array containing the line identifier for each station.

    Returns
    -------
    None
        If the panel is not complete in some sense, the function returns nothing
    result : dict
        Dictionary containing the results of the illumination analysis. Entries
        are:
            Start - UTCDateTime start of the noise panel
            End - UTCDateTime end of the noise panel
            Slow0 - The dominant slowness found on line 0 [s/m]
            Slow1 - The dominant slowness found on line 1 [s/m]

    """
    # If the panel is not complete or the master trace is not long enough, 
    # skip the panel
    if len(window) != amt_stations:
        return None
    elif window[mtr_idx].stats.npts != window_length * window[mtr_idx].stats.sampling_rate + 1:
        return None
    
    # First normalise the panel
    norm_wind = obspy.Stream()
    for trace in window:
        norm_wind += normalise_trace(trace)

    # norm_wind = normalise_section(window)

    # Then crosscorrelate the panel with the master trace 
    # XXX Use 2dconvolve
    record_corr = obspy.Stream()
    for trace in norm_wind:
        if trace.stats.npts != window_length * trace.stats.sampling_rate + 1:
            return None

        record_corr += cross_corr(norm_wind[mtr_idx], trace)

    # Get some extra information
    dt = window[0].stats.delta
    # This ensures that the Tau-P transform is evaluated at t=0 in the 
    # crosscorrelation
    slice_idx = window_length * window[0].stats.sampling_rate
    
    # A quick check to see if there are any results
    amt_results = 0
    slownesses = []
    for line in lines:
        # Select only the right line
        data_line = record_corr.select(location=line)


        cont = False
        for trace in data_line:
            # If one of the traces does not have the right length, add nothing
            # as a result
            if trace.stats.npts != 2 * window_length * trace.stats.sampling_rate + 1:
                cont = True
                break
        if cont:
            slownesses.append(None)
        else:
            # Set up the data 
            data = np.array(data_line)
            # Perform the Tau-P transform at the right location
            TauP_data = TauP_slice(data, p_range, distances[line_id == float(line)].squeeze(), dt, slice_idx)

            amt_results += 1
            # Add the slowness to the results
            slownesses.append(p_range[np.argwhere(TauP_data == TauP_data.max())[0][0]])

    if amt_results == 0:
        return None
    
    # Set up the results as a dictionary
    result = {'Start': window[0].stats.starttime,
              'End': window[0].stats.endtime,
              'Slow0': slownesses[0],
              'Slow1': slownesses[1]
              }

    return result

############## Utility functions

def add_line(data, distances, dt, ricker_width, start_time, velocity, amplitude):
    """
    Used to generate test data. Takes the input data and adds a straight line,
    described by a certain velocity. The signal itself is a Ricker wavelet from
    scipy. No data is overwritten.

    Parameters
    ----------
    data : array [amt of stations, amt of time samples]
        The data for which a line should be added.
    distances : array [amt of stations, 1] [m]
        Array containing the distance between the stations and the source location
    dt : float [s]
        The timing between the time samples.
    ricker_width : int
        Width of the Ricker wavelet used by scipy.
    start_time : float [s]
        Time at which the signal starts.
    velocity : float [m/s]
        The velocity with which the signal travels.

    Returns
    -------
    data : array
        The original data with a straight line added.

    """
    # Get information from the data
    amt_stations = data.shape[0]
    amt_samples = data.shape[1]
    
    # For each station trace, add the Ricker wavelet at the right time
    for i in range(amt_stations):
        # Determine the index that belongs to the timing of the start of the Ricker
        # wavelet
        index_start = min([int(round(start_time/dt + distances[i]/(velocity*dt))) - 5*ricker_width, amt_samples - 1])
        # The end index for the wavelet
        index_end = min([index_start + 5*ricker_width, amt_samples - 1])
        
        # Add the wavelet to the data
        data[i, index_start:index_end] += ricker(10*ricker_width,ricker_width)[0:max(0,index_end-index_start)] * amplitude
    
    return data

def add_hyperbola(data, distances, dt, ricker_width, start_time, depth, velocity, amplitude):
    """
    Used to generate test data. Takes the input data and adds a hyperbola. Could
    simulate reflection data. The signal is a Ricker wavelet. The new part is
    added, so that no data is overwritten.

    Parameters
    ----------
    data : array [amt of stations, amt of time samples]
        The data for which a line should be added.
    distances : array [amt of stations, 1] [m]
        Array containing the distance between the stations and the source location
    dt : float [s]
        The timing between the time samples.
    ricker_width : int
        Width of the Ricker wavelet used by scipy.
    start_time : float [s]
        Time at which the signal starts.
    depth : float
        The hyperbola can describe reflection data in a model with only one layer.
        This is the depth of this layer.
    velocity : float
        The velocity of this layer.

    Returns
    -------
    data : array
        The array with a hyperbola added.

    """
    # Get information from the data
    amt_stations = data.shape[0]
    amt_samples = data.shape[1]
    
    # Go over each trace
    for i in range(amt_stations):
        # Determine the start and end index for the signal according to the 
        # equation for a hyperbola
        index_start = min([int(round(2*np.sqrt( (distances[i]/2)**2 + depth**2 ) / (velocity*dt) + start_time/dt)), amt_samples - 1])
        index_end = min([index_start + 10*ricker_width, amt_samples - 1])
        
        # Add the Ricker wavelet to the trace
        data[i, index_start:index_end] += ricker(10*ricker_width,ricker_width)[0:max(0,index_end-index_start)] * amplitude
    return data

def write_results(results, master_stat, csv_writer, date_format='%Y-%m-%d - %H-%M-%S'):
    """
    Write the results of the illumination analysis to a csv/txt file. 

    Parameters
    ----------
    results : list
        List with dictionaries for each noise panel containing the results.
    master_stat : str
        Name of the master station.
    csv_writer : csv.writer
        An object that can write the results to a specific file.
    date_format : str, optional
        The format in which the date is written inside the file. 
        The default is '%Y-%m-%d - %H-%M-%S'.

    Returns
    -------
    None.

    """
    for result in results:
        if result is not None:
            time_start = result.get('Start')
            time_end = result.get('End')
            dom_slow0 = result.get('Slow0')
            dom_slow1 = result.get('Slow1')

            csv_writer.writerow(
                [time_start.strftime(date_format),
                 time_end.strftime(date_format),
                 master_stat,
                 dom_slow0,
                 dom_slow1])