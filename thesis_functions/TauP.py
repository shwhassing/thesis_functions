# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:41:52 2022

@author: sverr
"""

import numpy as np
from scipy.signal import ricker#, correlate
from os.path import join, split
import obspy
import csv
from obspy.signal.cross_correlation import correlate_template
from thesis_functions.of import open_cont_record, open_diff_stat
from thesis_functions.coord import calcCoord, open_line_id
from thesis_functions.util import tfs_string
import matplotlib.pyplot as plt

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
    
    shifts = np.around(distances[:,np.newaxis] * p_range[np.newaxis,:] / dt)
    index_time = (shifts + slice_idx).astype(int)
        
    kut = np.shape(data)[1]
    test1 = index_time >= kut
    test2 = index_time < 0
    
    mask_zeroes = np.logical_or(test1, test2)
    index_time[mask_zeroes] = 0
    # Repeat the index from 0 to the amount of stations for every p-value
    
    index_space = (np.arange(0, np.shape(data)[0])[:,np.newaxis] + np.zeros(len(p_range))[np.newaxis,:]).astype(int)
        
    values = data[index_space, index_time]
    values[mask_zeroes] = 0
    values = values.sum(axis=0)
    
    return values

############## Calculation functions

def TauP_at_T0(record, p_range, mtr_idx, f_ranges, window_length, path_info, csv_writer):
    
    date_format = '%Y-%m-%d - %H-%M-%S'
    master_stat = record[mtr_idx].stats.station[1:] + record[mtr_idx].stats.channel[2]
    
    # Attach a line identifier to the traces
    line_id = open_line_id(path_info)
    lines = np.unique(line_id).astype('int32').astype('U') # Get all lines
    for trace, line_no in zip(record, line_id):
        trace.stats.location = str(int(line_no))
    
    for f_range in f_ranges:
        record = record.filter('bandstop',
                               freqmin=f_range[0],
                               freqmax=f_range[1],
                               corners=4)
    
    dx_mat = calcCoord(path_info)
    distances = dx_mat[mtr_idx,:]
    
    dt = record[0].stats.delta
    slice_idx = window_length*record[0].stats.sampling_rate
    
    for window in record.slide(window_length = window_length, step = window_length):
        record_corr = obspy.Stream()
        
        if len(window) != len(record):
            continue
        
        for trace in window:
            record_corr += cross_corr(window[mtr_idx], trace)
        
        for line in lines:
            data_line = record_corr.select(location = line)
            
            cont = False
            for trace in data_line:
                if trace.stats.npts != 2 * window_length * trace.stats.sampling_rate + 1:
                    cont = True
            if cont:
                continue
            
            data = np.array(data_line)
            
            TauP_data = TauP_slice(data, p_range, distances[line_id == float(line)].squeeze(), dt, slice_idx)
            
            time_start = window[0].stats.starttime
            time_end = window[0].stats.endtime
            min_val = TauP_data.min()
            max_val = TauP_data.max()
            dom_slow = p_range[np.argwhere(TauP_data == max_val)[0][0]]
            
            csv_writer.writerow([time_start.strftime(date_format),time_end.strftime(date_format),master_stat,line,dom_slow,min_val,max_val])

def TauP_calc(stations, time_start, component, base_path, time_end, p_range, mtr_idx, freq_ranges, window_length, method = 'additive', path_info = None, distances = None, print_progress = True):
    """ A Tau-P transform (also called slant-stack or linear radon) performs a 
    linear move-out correction on the data and stacks over the distance axis.
    This can be interpreted as a plane wave decomposition, showing from which
    distances more energy is arriving.
    A list of stations to open is provided, together with information on the 
    time duration and component. Together this is enough to open all of the 
    relevant records. 
    There are two methods of calculation. Batch first opens all of the records
    and then performs the transformation, additive opens each file separately
    and adds the contributions to get the final result. The first is more 
    memory intensive, but has a shorter computation time.
    The station taken as the zero position is the master trace and indicated by
    the index mtr_idx. The slowness values for which this is evaluated are given 
    by p_range. Distance data can be provided directly as an array or as the 
    path to a csv file with coordinate information.

    Parameters
    ----------
    record : obspy Stream object
        Stream containing all of the traces that are used for the evaluation. 
        It is assumed that these all start at the same time. The order of the 
        traces should be the same as the distance info so that distances[i] is 
        the distance from the master trace station to the the trace of record[i].
    p_range : array
        The slowness values for which the Tau-P transform is evaluated.
    mtr_idx : int
        Index of the master trace. 
    path_info : string / path, optional
        Path to the file containing coordinate information. Will be read by 
        read_coords. The informatin can also be provided directly by distances,
        so can be left empty, but one of them has to be provided or an error is
        raised. The default is None.
    distances : array, optional
        Array containing the distances between all of the stations. These 
        distances should be the distance to the master station from the station
        indicated by that index. The default is None.

    Raises
    ------
    ValueError
        If neither of the options for the distances (path_info or distances) are
        provided, an error is raised.

    Returns
    -------
    taup_data : array
        Gives the Tau-P transform of the data with a linear transform.

    """
    if print_progress:
        print("Opening master trace...")
    
    # Open master trace
    master_trace = open_cont_record(stations[mtr_idx],
                                    time_start,
                                    component,
                                    base_path,
                                    time_end=time_end,
                                    print_progress=print_progress)

    # Filter the master trace
    for freq_range in freq_ranges:
        master_trace = master_trace.filter('bandstop',
                                           freqmin=freq_range[0],
                                           freqmax=freq_range[1],
                                           corners=4)

    # Get the file name to save the array
    time_start = master_trace[0].stats.starttime
    time_end = master_trace[0].stats.endtime
    str_dur = f'{time_start.year}-{time_start.month}-{time_start.day} {time_start.hour}-{time_start.minute}-{time_start.second} - {time_end.year}-{time_end.month}-{time_end.day} {time_end.hour}-{time_end.minute}-{time_end.second}'
    arr_name = f'./Arrays/TauP data {str_dur} - master {stations[mtr_idx]}{component}'

    # If neither path_info, nor distances are provided, an error is raised.
    if path_info == None and distances == None:
        raise ValueError("No distance information is provided, use either path_info or distances...")
    # If path_info is provided, distances is calculated form the information
    elif path_info != None:
        gp_dists = calcCoord(path_info)
        distances = gp_dists[mtr_idx,:]
    
    if method == 'batch':
        # Open a stream with all of the data in it
        record = open_diff_stat(stations, time_start, component, base_path, time_end = time_end)
        
        # Filter the data
        for freq_range in freq_ranges:
            record = record.filter('bandstop', freqmin = freq_range[0], freqmax = freq_range[1], corners = 4)
        
        # Initialise some information from the record data.
        dt = record[0].stats.delta
        
        # Now cross-correlate the data with the master trace and create a new
        # stream
        record_corr = obspy.Stream()
        for i in range(len(record)):
            record_corr += cross_corr(master_trace, record) #cross_corr(master_trace, record[i])
        
        # Transform the data into an array
        data = np.array(record_corr)
        
        # Perform the Tau-P transformation
        taup_data = TauP_batch(data, p_range, distances, dt)
    
    # The other method opens the files one by one and then adds their 
    # contributions
    elif method == 'additive':
        # Initialise the array for the Tau-P data
        taup_data = np.zeros([len(p_range), 2*int(round(window_length/master_trace[0].stats.delta)) + 1])
        
        if print_progress:
            print(f'Progress of stations\n0/{len(stations)}', end='')
            counter = 0
        
        # Start going over all of the stations
        for i, station in enumerate(stations):
            # Open the record
            record = open_cont_record(station,time_start,component,base_path,time_end = time_end, print_progress = False)
            
            # Filter it
            for freq_range in freq_ranges:
                record = record.filter('bandstop', freqmin = freq_range[0], freqmax = freq_range[1], corners = 4)
            
            # Cross-correlate with the master trace
            record_corr = cross_corr(master_trace[0], record[0])
            # window_sum(record, master_trace, window_length = window_length)
                        
            # Determine the relevant data
            dt = record_corr.stats.delta
            data = record_corr.data
            distance = distances[i]
            
            # Add the contribution of this station to the total
            taup_data += TauP_add(data, p_range, distance, dt)
            
            if print_progress:
                counter += 1
                print(f'\r{counter}/{len(stations)}', end = '')
    if print_progress:
        print("")
    # Save the result
    np.save(arr_name, taup_data)
    
    return taup_data

def TauP_1stat(stations, time_start, component, base_path, time_end, p_range, stat_idx, mtr_idx, freq_ranges, window_length, arr_names = [], path_info = None, distances = None, path_out = './Arrays/'):
    
    # Open master trace
    master_trace = open_cont_record(stations[mtr_idx],
                                    time_start,
                                    component,
                                    base_path,
                                    time_end=time_end,
                                    print_progress = False)

    line_id = open_line_id(path_info)
    line = line_id[stat_idx]

    # Filter the master trace
    for freq_range in freq_ranges:
        master_trace = master_trace.filter('bandstop',
                                           freqmin=freq_range[0],
                                           freqmax=freq_range[1],
                                           corners=4)

    # If neither path_info, nor distances are provided, an error is raised.
    if path_info == None and distances == None:
        raise ValueError("No distance information is provided, use either path_info or distances...")
    # If path_info is provided, distances is calculated form the information
    elif path_info != None:
        gp_dists = calcCoord(path_info)
        distances = gp_dists[mtr_idx,:]
        
    # Open the record
    record = open_cont_record(stations[stat_idx],time_start,component,base_path, time_end = time_end, print_progress = False)
    
    # Filter it
    for freq_range in freq_ranges:
        record = record.filter('bandstop', freqmin = freq_range[0], freqmax = freq_range[1], corners = 4)
    
    # Get the last part of the file name
    name = f'line {int(line)} - master {stations[mtr_idx]}{component}'
    
    # And go over the calculation
    arr_names = slide_trcs(record, master_trace, window_length, p_range, distances[stat_idx], name, arr_names, path_out = path_out)
    
    return arr_names

def TauP_allstat(stations, 
                 time_start, 
                 component, 
                 base_path, 
                 time_end, 
                 p_range, 
                 mtr_idx, 
                 f_ranges, 
                 window_length, 
                 path_info = None, 
                 distances = None, 
                 print_progress = True, 
                 save_arr = False,
                 path_out = './Arrays/'):
    # The format to use for all dates
    date_format = '%Y-%m-%d - %H-%M-%S'
    
    # Indicate at which timestamp the program starts running
    if print_progress:
        timestamp_start = obspy.core.UTCDateTime()
        print("Starting at ", timestamp_start.strftime(date_format))
    
    # Open all of the relevant records
    records = open_diff_stat(stations, time_start, component, base_path, time_end = time_end, print_progress = print_progress)
    
    if print_progress:
        timestamp_opened = obspy.core.UTCDateTime()
        print("All stations opened at", timestamp_opened.strftime(date_format))
        # dur = timestamp_opened - timestamp_start
        print("Or after a duration of", tfs_string(timestamp_opened-timestamp_start))
    
    # Attach a line identifier to the traces
    line_id = open_line_id(path_info)
    lines = np.unique(line_id).astype('int32').astype('U') # Get all lines
    
    for trace, line_no in zip(records, line_id):
        trace.stats.location = str(int(line_no))
    
    # Filter the traces
    for f_range in f_ranges:
        records = records.filter('bandstop', freqmin = f_range[0], freqmax = f_range[1], corners = 4)
        
    # If neither path_info, nor distances are provided, an error is raised.
    if path_info == None and distances == None:
        raise ValueError("No distance information is provided, use either path_info or distances...")
    # If path_info is provided, distances is calculated form the information
    elif path_info != None:
        gp_dists = calcCoord(path_info)
        distances = gp_dists[mtr_idx,:]
    
    arr_names = []
    
    if print_progress:
        npanels = int(np.floor(records[0].stats.npts*records[0].stats.delta / window_length))
        print(f"Progress of panels\n0/{npanels}", end = "")
        counter = 0
    
    time_start = obspy.core.UTCDateTime(time_start)
    time_end = obspy.core.UTCDateTime(time_end)
    log_name = join(path_out, f'Log {time_start.strftime(date_format)} - {time_end(date_format)} - master {stations[mtr_idx]}{component}.txt')
    file = open(log_name, 'w', newline='')
    writer = csv.writer(file)
    header = ['Start', 'End', 'Master', 'Line', 'Dom. vel', 'Min. val', 'Max val.']
    writer.writerow(header)
    
    for windows in records.slide(window_length = window_length, step = window_length):
        record_corr = obspy.Stream()
        for window in windows:
            record_corr += cross_corr(windows[mtr_idx], window)
        
        dt = record_corr[0].stats.delta
        
        time_start = windows[0].stats.starttime
        time_end = windows[0].stats.endtime
        
        for line in lines:
            data = np.array(record_corr.select(location = line))
            
            taup_data = TauP_batch(data, p_range, distances, dt)
            
            min_val = taup_data.min()
            max_val = taup_data.max()
            dom_vel = 1/p_range[np.argwhere(taup_data == max_val)[0][0]]
            writer.writerow([time_start.strftime(date_format),time_end.strftime(date_format),stations[mtr_idx]+component,line,dom_vel,min_val,max_val])
            
            if save_arr:
                name = f'line {line} - master {stations[mtr_idx]}{component} - batch'
                
                arr_name = join(path_out,f'TauP data {time_start.strftime(date_format)} - {time_end.strftime(date_format)} - {name}')
                arr_names.append(arr_name)
            
                np.save(arr_name, taup_data)
        
        if print_progress:
            counter += 1
            print(f"\r{counter}/{npanels}", end = "")
    
    file.close()
    
    if print_progress:
        print("")
        timestamp_end = obspy.core.UTCDateTime()
        print("Ending at", timestamp_end.strftime(date_format))
        print("For a total duration of", tfs_string(timestamp_end-timestamp_start))
        
    return arr_names

def TauP_range(stations, time_start, component, base_path, time_end, p_range, mtr_idx, f_ranges, window_length, path_info = None, distances = None, print_progress = True, path_out = './Arrays/'):
    
    if print_progress:
        print(f"Progress of stations\n0/{len(stations)}", end = "")
    
    file_list = []
    
    for i in range(len(stations)):
        new_file_list = TauP_1stat(stations, 
                               time_start, 
                               component, 
                               base_path, 
                               time_end, 
                               p_range, 
                               i, 
                               mtr_idx, 
                               f_ranges, 
                               window_length, 
                               file_list,
                               path_info = path_info, 
                               distances = distances, 
                               path_out = path_out)
        
        for filename in new_file_list:
            if filename not in file_list:
                file_list.append(filename)
        
        if print_progress:
            print(f'\r{i+1}/{len(stations)}', end = '')
            
    if print_progress:
        print("")
        
    return file_list

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

def cross_corr(master_trace, trace2):
    mast_trc = master_trace.copy()
    corr = correlate_template(trace2, mast_trc, mode='full', normalize = 'full')
    # corr = correlate(master_trace.data,trace2.data, mode='same')
    trace_corr = obspy.Trace()
    
    trace_corr.data = corr
    trace_corr.stats.starttime = master_trace.stats.starttime - (master_trace.stats.endtime - master_trace.stats.starttime)
    trace_corr.stats.station = trace2.stats.station
    trace_corr.stats.network = master_trace.stats.network
    trace_corr.stats.delta = master_trace.stats.delta
    trace_corr.stats.channel = master_trace.stats.channel
    trace_corr.stats.location = trace2.stats.location
    
    return trace_corr

def slide_trcs(record, master_trace, window_length, p_range, distance, name, arr_names, path_out = './Arrays/'):
    arr_names = []
    for window, mstr_window in zip(record.slide(window_length = window_length, step = window_length), 
                                   master_trace.slide(window_length = window_length, step = window_length)):
        time_start = window[0].stats.starttime
        time_end = window[0].stats.endtime
        
        # str_dur = f'{time_start.year}-{time_start.month}-{time_start.day} {time_start.hour}-{time_start.minute}-{time_start.second} - {time_end.year}-{time_end.month}-{time_end.day} {time_end.hour}-{time_end.minute}-{time_end.second}'
        date_format = '%Y-%m-%d - %H-%M-%S'
        arr_name = join(path_out, f'TauP data {time_start.strftime(date_format)} - {time_end.strftime(date_format)} - {name}')
        
        if arr_name not in arr_names:
            arr_names.append(arr_name)
        
        try:
            taup_data = np.load(arr_name+'.npy')
        except FileNotFoundError:
            taup_data = np.zeros([len(p_range), 2*int(round(window_length/window[0].stats.delta)) + 1])
                
        # Cross-correlate with the master trace
        record_corr = cross_corr(mstr_window[0], window[0])
        # window_sum(record, master_trace, window_length = window_length)
        
        # Determine the relevant data
        dt = record_corr.stats.delta
        data = record_corr.data
        # distance = distances[stat_idx]
        
        # Add the contribution of this station to the total
        taup_data += TauP_add(data, p_range, distance, dt)
                
        # Save the result
        np.save(arr_name, taup_data)
    return arr_names

def plot_file(filename, p_range, window_length, clim = None, savefig = False, show = True, path_out = './Images/TauP/'):
    taup_data = np.load(filename+'.npy')
    
    if clim == None:
        clim = max([abs(taup_data.min()),abs(taup_data.max())])
    
    plt.figure(figsize=(10,10), dpi = 400)
    plt.imshow(taup_data.T,
                aspect='auto',
                origin = 'upper',
                extent = [p_range[0], p_range[-1], window_length, -window_length],
                vmin = -clim,
                vmax = clim
                )
    plt.xlabel("Slowness [s/m]")
    plt.ylabel("Time [s]")
    plt.colorbar()

    dom_idx = np.argwhere(taup_data == taup_data.max())[0]
    dom_vel = 1/p_range[dom_idx[0]]
    print(f"Dominant velocity is {dom_vel} m/s")
    
    if savefig:
        plt.savefig(join(path_out, split(filename)[-1]))
    
    if show:
        plt.show()
    plt.close()

def flip_add(array):
    if len(array) % 2 == 0:
        new_array = array[int(len(array)/2)-1::-1] + array[int(len(array)/2):]
    else:
        new_array = array[int(len(array)/2)::-1] + array[int(len(array)/2):]
        new_array[int(len(array)/2)] /= 2
    return new_array
    
def window_sum(record_in, master_trace, window_length, flip = False, step = None):
    
    if step == None:
        step = window_length
    
    trace_sum = obspy.Trace()
    
    if isinstance(record_in, obspy.Trace):
        record = obspy.Stream()
        record += record_in
    else:
        record = record_in
    
    if flip:
        data_len = int(round(window_length/record[0].stats.delta)) + 1
    else:
        data_len = 2*int(round(window_length/record[0].stats.delta)) + 1
    
    trace_sum.data = np.zeros(data_len)
    
    for window, mstr_window in zip(record.slide(window_length = window_length, step = step), master_trace.slide(window_length = window_length, step = step)):
        if isinstance(master_trace,obspy.Trace):
            correlated = cross_corr(mstr_window, window[0])
        else:
            correlated = cross_corr(mstr_window[0], window[0])
        if flip:
            trace_sum.data += flip_add(correlated)
        else:
            trace_sum.data += correlated
    trace_sum.stats.starttime = record[0].stats.starttime
    trace_sum.stats.delta = record[0].stats.delta
    trace_sum.stats.station = record[0].stats.station
    trace_sum.stats.channel = record[0].stats.channel
    trace_sum.stats.network = record[0].stats.network
    
    return trace_sum

def normalise_trace(trace):
    # Create a new trace
    new_trace = obspy.Trace()

    # If all of the data are zeroes, just return the original to prevent
    # a divide by zero
    if np.all(trace.data == 0.):
        return trace

    # Calculate the rms of the trace
    squares = np.square(trace.data)
    mean_squares = np.mean(squares)
    if mean_squares == 0.:
        return trace

    root_mean_squares = np.sqrt(mean_squares)
    new_trace.data = trace.data / root_mean_squares
    new_trace.stats = trace.stats
    return new_trace

def process_window(window,
                   amt_stations,
                   lines,
                   mtr_idx,
                   window_length,
                   p_range,
                   distances,
                   line_id):
    if len(window) != amt_stations:
        return None
    elif window[mtr_idx].stats.npts != window_length * window[mtr_idx].stats.sampling_rate + 1:
        return None

    norm_wind = obspy.Stream()
    for trace in window:
        norm_wind += normalise_trace(trace)

    # norm_wind = normalise_section(window)

    record_corr = obspy.Stream()
    for trace in norm_wind:
        if trace.stats.npts != window_length * trace.stats.sampling_rate + 1:
            return None

        record_corr += cross_corr(norm_wind[mtr_idx], trace)

    dt = window[0].stats.delta
    slice_idx = window_length * window[0].stats.sampling_rate

    amt_results = 0
    slownesses = []
    for line in lines:
        data_line = record_corr.select(location=line)

        cont = False
        for trace in data_line:
            if trace.stats.npts != 2 * window_length * trace.stats.sampling_rate + 1:
                cont = True
                break
        if cont:
            slownesses.append(None)
        else:
            data = np.array(data_line)

            TauP_data = TauP_slice(data, p_range, distances[line_id == float(line)].squeeze(), dt, slice_idx)

            amt_results += 1
            slownesses.append(p_range[np.argwhere(TauP_data == TauP_data.max())[0][0]])

    if amt_results == 0:
        return None

    result = {'Start': window[0].stats.starttime,
              'End': window[0].stats.endtime,
              'Slow0': slownesses[0],
              'Slow1': slownesses[1]
              }

    return result

def write_results(results, master_stat, csv_writer, date_format='%Y-%m-%d - %H-%M-%S'):
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