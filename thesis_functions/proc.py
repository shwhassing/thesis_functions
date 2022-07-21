import glob
import os
import numpy as np
import obspy
from os.path import join
from scipy.signal import convolve2d
from thesis_functions.TauP import cross_corr
from thesis_functions.coord import attach_distances, select_line, open_line_id, get_unique_lines, attach_line
from thesis_functions.util import auto_filter

def extract_results(path, master_trace, component, added_string=''):
    # Get a list of all of the output files
    output_files = glob.glob(join(path,f"Log day * - master {master_trace}{component}{added_string}.txt"))

    # if component in ['N','E']:
        
    # Initiate lists for all of th relevant information
    start_time = []
    end_time = []
    dom_slow0 = []
    dom_slow1 = []
    
    # Go over each file
    for output_file in output_files:
        # Get all of the lines from th file
        file = open(output_file, 'r')
        lines = file.readlines()[1:]
        file.close()
        
        # If there is only a header in the file, a string is returned,
        # in that case there were no results for the day, so continue
        if isinstance(lines,str):
            continue
        
        # Now go over the lines and extract the relevant information
        for line in lines:
            start, end, master, slow0, slow1 = line.split(',')
            start_time.append(start)
            end_time.append(end)
            dom_slow0.append(slow0)
            dom_slow1.append(slow1)
    
    # Convert the lists to arrays
    start_time = np.array(start_time)
    end_time = np.array(end_time)
    dom_slow0 = np.array(dom_slow0, dtype=float)
    dom_slow1 = np.array(dom_slow1, dtype=float)
    
    return start_time, end_time, dom_slow0, dom_slow1
    # elif component == "Z":
    
    #     start_time = []
    #     end_time = []
    #     dom_slow = []
    #     line_id = []
    #     for output_file in output_files:
    #         file = open(output_file, 'r')
    #         lines = file.readlines()[1:]
    #         file.close()
            
    #         if isinstance(lines,str):
    #             continue
            
    #         for line in lines:
    #             start, end, master, line_no, slow, __, __ = line.split(',')
    #             start_time.append(start)
    #             end_time.append(end)
    #             dom_slow.append(slow)
    #             line_id.append(line_no)
                
    #     start_time = np.array(start_time)
    #     end_time = np.array(end_time)
    #     dom_slow = np.array(dom_slow, dtype=float)
    #     line_id = np.array(line_id, dtype=int)
        
    #     start_time_plt = convert_date(start_time, 'plt')
        
    #     mask = line_id == 0
    #     time0 = start_time_plt[mask]
    #     time1 = start_time_plt[np.logical_not(mask)]
        
    #     # Filter out times where only one of the lines was evaluated
    #     times_in_both = np.intersect1d(time0,time1)
    #     mask2 = np.isin(start_time_plt, times_in_both)
        
    #     start_time_plt = start_time_plt[mask2]
    #     dom_slow = dom_slow[mask2]
    #     line_id = line_id[mask2]
    #     start_time = start_time[mask2]
    #     end_time = end_time[mask2]
        
    #     mask = line_id == 0
        
    #     return start_time[mask], end_time[mask], dom_slow[mask], dom_slow[np.logical_not(mask)]

def convert_date(times, method):
    """
    Convert the date format from the string given from the illumination analysis
    to different date objects. Options are:
        plt:
            Matplotlib date
        obspy:
            Obspy UTCDateTime object

    Parameters
    ----------
    times : np.ndarray
        Array containing the date strings to be converted.
    method : str
        String indicating the format to convert to. Can be 'plt' for 
        matplotlib or 'obspy' for a UTCDateTime object.

    Returns
    -------
    np.ndarray
        Array containing the times in the new form.

    """
    new_times = []
    for time in times:
        if method == 'plt':
            new_times.append(obspy.UTCDateTime(time).matplotlib_date)
        elif method == 'obspy':
            new_times.append(obspy.UTCDateTime(time))
    return np.array(new_times)

def select_panels(dom_slow0, dom_slow1, vel_cut):
    """
    Gives a mask to select all panels where the dominant slowness for both
    lines is lower than the reciprocal of a selected velocity.

    Parameters
    ----------
    dom_slow0 : np.ndarray
        Array containing the dominant slownesses for line 0. It is assumed to
        have the same shape and ordering as dom_slow1
    dom_slow1 : np.ndarray
        Similar to dom_slow0, but for line 1.
    vel_cut : float
        Dominant slownesses are tested to be below 1/vel_cut.

    Returns
    -------
    np.ndarray
        Returns mask (array with booleans) for the locations where the conditions
        are met.

    """
    # Test for both lines
    test0 = abs(dom_slow0) <= 1/vel_cut
    test1 = abs(dom_slow1) <= 1/vel_cut
    # And give back the places where both are selected.
    return np.logical_and(test0, test1)

def date_from_filename(filename):
    """
    Gives an obspy UTCDateTime object from the filename of an output file.

    Parameters
    ----------
    filename : str
        The string containing the filename that was output from the illumination
        analysis.

    Returns
    -------
    obspy UTCDateTime
        The time contained in the filname as UTCDateTime object.

    """
    return obspy.core.UTCDateTime("%s-%s-%sT%s:%s:%s"%tuple(filename.split('.')[:-2]))

def times_mask(times, filename, chunk_len=30*60):
    # Find the start and end times of the chunk
    start_chunk = date_from_filename(filename)
    end_chunk = start_chunk + chunk_len
    
    # Select the times that fit in this chunk
    mask = np.logical_and(times >= start_chunk, times < end_chunk)
    return mask

def find_times_in_chunk(times, filename, chunk_len = 30*60):
    """
    Given a filename and a list of times, finds the times contained in that 
    file and returns them. 

    Parameters
    ----------
    times : np.ndarray
        Array with times as obspy UTCDateTime object or matplotlib date.
    filename : str
        The filename of the data chunk that is analysed.
    chunk_len : float, optional
        The length of the chunk in seconds. Default is the default value from
        the illumination analysis.
        
    Returns
    -------
    np.ndarray
        Array with times as the same type as the input, sliced to only contain
        times fitting in the chunk.

    """
    mask = times_mask(times, filename, chunk_len)
    return times[mask]

def get_panel(record, times, window_length = 10.):
    """
    Generator function that gives each selected time panel in a time section.
    Length of the panel can be adjusted.

    Parameters
    ----------
    record : obspy Stream
        A stream object containing the data.
    times : np.ndarray
        Array containing the times as obspy UTCDateTime objects that should be 
        cut from the data. The time is the start of the new panel.
    window_length : float, optional
        Length of the panels that are cut from the data. The default is 10.

    Yields
    ------
    obspy Stream
        New stream object cut to the desired time and length.

    """
    for time in times:
        yield record.slice(time, time+window_length)

def auto_corr_section(record):
    """
    Autocorrelate all of the traces in a section. Only positive times are taken
    from the correlation, so that t0 from the autocorrelation is at the original
    start time of the section.

    Parameters
    ----------
    record : obspy Stream
        Original section.

    Returns
    -------
    record_corr : obspy Stream
        Autocorrelated section.

    """
    # Create a new stream
    record_corr = obspy.Stream()
    
    # Go over all of the traces
    for trace in record:
        # Correlate the trace with itself
        trace_corr = cross_corr(trace, trace)
        
        # Only take positive times
        trace_corr.data = trace_corr.data[int((trace_corr.stats.npts - 1) / 2):]
        
        # Add it to the stream
        record_corr += trace_corr
    
    return record_corr

def recreate_stream(section, record, line, dist_tr, path_info):
    """
    Recreate a stream after autocorrelating data. Copies different characteristics
    over from original Stream

    Parameters
    ----------
    section : np.ndarray
        Array containing the autocorrelated data.
    record : obspy Stream
        The stream to copy the information from.
    line : str
        Which line is being processed.
    path_info : str or path
        Path to the coordinate information.

    Returns
    -------
    section_stream : obspy Stream
        .

    """
    # Select the right line for the data
    rec = select_line(record, line, path_info)
    # Create a new stream
    section_stream = obspy.Stream()
    
    # Go over each trace
    for i in range(section.shape[0]):
        # Create a new trace
        trace = obspy.Trace()
        # Add the data and extra information
        trace.data = section[i,:]
        trace.stats.station = rec[i].stats.station
        trace.stats.sampling_rate = rec[i].stats.sampling_rate
        trace.stats.starttime = rec[i].stats.starttime
        trace.stats.channel = rec[i].stats.channel
        
        # Add the trace to the stream
        section_stream += trace
    
    # Also attach the distance to a selected station
    # XXX Is hardcoded now, should be changed
    # if line == '0':
    #     mtr_idx = 27
    # else:
    #     mtr_idx = 36

    section_stream = attach_distances(section_stream, dist_tr, line, path_info)
    return section_stream

def normalise_section(record):
    new_record = record.copy()
    
    squares = np.square(np.array(record))
    mean_squares = np.mean(squares, axis=1)
    mean_squares[mean_squares == 0] = 1.
    
    root_mean_squares = np.sqrt(mean_squares)
    new_data = np.array(record) / root_mean_squares[:,np.newaxis]
    
    for i,trace in enumerate(new_record):
        trace.data = new_data[i,:]
    return new_record

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

def autocorr_section(path_base, path_saved, path_info, mtr_station, component, window_length, vel_cut, added_string = '', print_progress=True):
    """
    Function that handles generating an autocorrelated section from the results
    of illumination analysis. 

    Parameters
    ----------
    path_base : str or path
        Path to the location of the raw data.
    path_saved : str or path
        Path to the results of the illumination analysis.
    path_info : str or path
        Path to coordinate information, see thesis_function.coord.read_coords
    mtr_station : str
        The station number of the master trace used for the illumination analysis.
    component : str
        Component for which the section is generated.
    window_length : float
        Window length used for the panels of the autocorrelation.
    vel_cut : float
        Minimum dominant velocity to use to select the panels.
    print_progress : bool, optional
        Whether or not to print the progress of calculations. The default is True.

    Returns
    -------
    section0 : Stream
        Stream containing the autocorrelations for line 0.
    section1 : Stream
        Stream containing the autocorrelations for line 1.

    """
    # Get the results from the illumination analysis
    start_time, __, dom_slow0, dom_slow1 = extract_results(path_saved, mtr_station, component, added_string)
    # Select the times when the events come in roughly vertical
    mask = select_panels(dom_slow0, dom_slow1, vel_cut)
    times_sel = start_time[mask]
    
    # Convert the times to obspy UTCDateTime objects and open the line numbers
    times_sel = convert_date(times_sel, 'obspy')
    line_id = open_line_id(path_info)
    
    # Now go over all files with raw data
    folder_list = glob.glob(os.path.join(path_base,'*'))
    
    # Read one file to get some information
    record = obspy.read(glob.glob(os.path.join(path_base,'*','*.mseed'))[0])
    # Initialise the autocorrelated sections
    section_l0 = np.zeros([np.sum(line_id == 0.), int(window_length*record[0].stats.sampling_rate + 1)])
    section_l1 = np.zeros([np.sum(line_id == 1.), int(window_length*record[0].stats.sampling_rate + 1)])
    
    # Go over all of the folders
    for i,folder in enumerate(folder_list):
        # See all the files in this folder
        file_list = glob.glob(os.path.join(folder,f'*.{component}.mseed'))
        
        # Go over all of the files
        for j,file in enumerate(file_list):
            # See which of the selected times fits in this chunk
            times_chunk = find_times_in_chunk(times_sel,os.path.split(file)[-1])
            
            if print_progress:
                print(f'\r{i}/{len(folder_list)}\t[{j}/{len(file_list)}]\tOpening file...       ', end='')
            
            # If there are not times selected, skip this file
            if len(times_chunk) == 0:
                continue
            
            # Otherwise read in the file
            record = obspy.read(file)
                
            # Go over all of the panels in the file
            for k,panel in enumerate(get_panel(record, times_chunk)):
                if print_progress:
                    print(f'\r{i}/{len(folder_list)}\t[{j}/{len(file_list)}]\t[{k}/{len(times_chunk)}]\tAdding panels...', end='')
                
                # Filter the data
                panel = auto_filter(panel)
                
                # Normalise each trace against itself
                panel = normalise_section(panel)
                
                # Autocorrelate it
                auto_corr = auto_corr_section(panel)
                
                # And add the data of this panel to the total
                section_l0 += np.array(select_line(auto_corr, '0', path_info))
                section_l1 += np.array(select_line(auto_corr, '1', path_info))
    
    if print_progress:
        print(f'\r{i+1}/{len(folder_list)}\t[{j+1}/{len(file_list)}]\t\t\t\t', end = '')
    
    # In the end, add all of the other data to create a stream
    section0 = recreate_stream(section_l0, record, '0', 27, path_info)
    section1 = recreate_stream(section_l1, record, '1', 36, path_info)
    
    return section0, section1

def flip_shot(virt_rec,dom_slow):
    data = np.zeros([len(virt_rec), int((virt_rec[0].stats.npts-1)/2+1)])
    dists = np.zeros(len(virt_rec))
    
    direction = dom_slow/abs(dom_slow)
    for i,trace in enumerate(virt_rec):
        dists[i] = trace.stats.distance
        
    raw_data = np.array(virt_rec)
    data[dists*direction>=0.,:] = raw_data[dists*direction>=0.,int((virt_rec[0].stats.npts-1)/2):]
    data[dists*direction<0.,:] = raw_data[dists*direction<0.,:int((virt_rec[0].stats.npts-1)/2+1)][:,::-1]
    
    return data

def convert_shotdata(virt_shots,record,line,path_info):
    streams = []
    
    for i in range(len(virt_shots)):
        streams.append(recreate_stream(virt_shots[i,...],record,line,i,path_info))
    return streams

def save_shotdata(path_save,shots,line,min_vel):
    new_path = os.path.join(path_save,'Crosscorr '+str(int(min_vel)),line)

    if not os.path.isdir(os.path.dirname(new_path)):
        os.mkdir(os.path.dirname(new_path))

    for i,stream in enumerate(shots):
        filename = f'Line {line} - shot {stream[i].stats.station[1:]}.mseed'

        try:
            stream.write(os.path.join(new_path,filename))
        except FileNotFoundError:
            os.mkdir(new_path)
            stream.write(os.path.join(new_path,filename))

def crosscorr_section(path_base,path_saved,path_info,mtr_station,component,window_length,vel_cut,print_progress=True, return_stream=None):
    start_time, __, dom_slow0, dom_slow1 = extract_results(path_saved, mtr_station, component)
    
    mask = select_panels(dom_slow0, dom_slow1, vel_cut)
    
    times_sel = convert_date(start_time[mask],'obspy')
    dom_slow = np.stack([dom_slow0, dom_slow1]).swapaxes(0,1)
    dom_slow_sel = dom_slow[mask,:]
    
    line_id = open_line_id(path_info)
    
    # Read one file to get some information
    record = obspy.read(glob.glob(os.path.join(path_base,'*','*.mseed'))[0])
    
    virt_shots = [np.zeros([np.sum(line_id==0),np.sum(line_id==0),int(record[0].stats.sampling_rate*window_length+1)]),
                  np.zeros([np.sum(line_id==1),np.sum(line_id==1),int(record[0].stats.sampling_rate*window_length+1)])]
    
    folder_list = glob.glob(os.path.join(path_base,'*'))
    
    if print_progress:
        counter = 0
        print(f"Progress:\n0/{len(times_sel)}\t0/{len(line_id)}",end='')
    
    for folder in folder_list:
        
        file_list = glob.glob(os.path.join(folder,f'*.{component}.mseed'))
        
        for file in file_list:
            mask_chunk = times_mask(times_sel,os.path.split(file)[-1])
            slows = dom_slow_sel[mask_chunk,:]
            times_chunk = times_sel[mask_chunk]
            
            # If there are not times selected, skip this file
            if len(times_chunk) == 0:
                continue
            
            record = obspy.read(file)
            record = attach_line(record,path_info)
            
            for i,panel in enumerate(get_panel(record,times_chunk,window_length)):
                
                
                panel = normalise_section(panel)
                
                for j,master_trace in enumerate(panel):
                    
                    line = master_trace.stats.location
                    mtr_stat = master_trace.stats.station
                    
                    panel_sel = select_line(panel,line,path_info)
                    
                    for idx,trace in enumerate(panel_sel):
                        if trace.stats.station == mtr_stat:
                            new_j = idx
                    
                    record_corr = obspy.Stream()
                    for trace in panel_sel:
                        trace_corr = cross_corr(master_trace, trace)
                        record_corr += trace_corr
                
                    record_corr = attach_distances(record_corr, new_j, line, path_info)
                
                    add_line = flip_shot(record_corr,slows[i,int(line)])
                    
                    virt_shots[int(line)][new_j,:,:] += np.array(add_line)
                    
                    if print_progress:
                        print(f"\r{counter}/{len(times_sel)}\t{j}/{len(record)}      ",end='')
            
            if print_progress:
                counter += 1
                print(f"\r{counter+1}/{len(times_sel)}\t{j+1}/{len(record)}     ",end='')
    
    if print_progress:
        print("Saving...")
    
    if return_stream == 'all':
        results = []
    
    lines = get_unique_lines(path_info)
    for line in lines:
        print(f'\rSaved line {line}', end='')
        streams = convert_shotdata(virt_shots[int(line)], record, line, path_info)
        save_shotdata(path_saved,streams,line)
        
        if return_stream == line:
            results = streams
        elif return_stream == 'all':
            results.append(streams)
    return results
    

def AGC_scaling_val(window, type_scal):
    """
    Return the value with which to scale the data for AGC, see the function
    AGC for more information. This function calculates a separate value for each
    trace in the stream. Mirrors the SeisSpace ProMax AGC function. Determines
    some kind of average for the data and returns the inverse. There are
    three methods:
        mean:
            Uses the mean of the absolute amplitudes of each trace
        median:
            Uses the median of the absolute amplitudes of each trace
        RMS:
            Uses the rms amplitude of each trace.

    Parameters
    ----------
    window : obspy Stream
        Stream object that contains the traces.
    type_scal : str
        The type of scaling to use. Can be 'mean', 'median' or 'RMS'

    Returns
    -------
    float
        Inverse of the selected kind of average to scale the data with.

    """
    # First takes the absolute values of the data
    data = abs(np.array(window))
    
    # Then determine the right kind of average
    if type_scal == 'mean':
        return 1/np.mean(data,axis=1)
    elif type_scal == 'median':
        return 1/np.median(data,axis=1)
    elif type_scal == 'RMS':
        mean_square = np.mean(np.square(data), axis=1)
        return 1/np.sqrt(mean_square)

def get_AGC_window(trace, oper_len, basis):
    """
    Generator function that gives the window for the AGC function. Streams are
    assumed to all contain traces with the same duration in time and sampling
    rate. For each sample, a window is created. The length of the window is set.
    The time window can then be located at three different positions compared 
    to the sample:
        trailing:
            Window starts at the sample
        leading:
            Window ends at the sample
        centred:
            Sample is located at the middle of the sample
    At the start or end, the length of the window can be shorter to accommodate
    the lack of data at the edges of the record.

    Parameters
    ----------
    trace : obspy Stream or obspy Trace
        The data for which the window has to be given.
    oper_len : float
        Length of the window in seconds.
    basis : str
        Indicates how the window is located around the sample location. Can be
        'trailing', 'leading' or 'centred'.

    Yields
    ------
    obspy Stream
        The window to be used for the AGC.

    """
    # Depending on if the input is a trace or stream, the information is gotten
    # from a different location
    if isinstance(trace,obspy.Trace):
        time_start = trace.stats.starttime
        dt = trace.stats.delta
        amt_samples = trace.stats.npts
    elif isinstance(trace, obspy.Stream):
        time_start = trace[0].stats.starttime
        dt = trace[0].stats.delta
        amt_samples = trace[0].stats.npts
    
    # With the index for each sample, the window is given
    for sample_idx in range(amt_samples):
        if basis == 'trailing':
            yield trace.slice(starttime=time_start+dt*sample_idx,
                              endtime=time_start+oper_len+dt*sample_idx)
        elif basis == 'leading':
            yield trace.slice(starttime=time_start-oper_len+dt*sample_idx,
                              endtime=time_start+dt*sample_idx)
        elif basis == 'centred':
            yield trace.slice(starttime=time_start-0.5*oper_len+dt*sample_idx,
                              endtime=time_start+0.5*oper_len+dt*sample_idx)

def AGC_trace(trace, oper_len, type_scal, basis):
    """
    Apply AGC to a single trace. For more information see the function AGC.

    Parameters
    ----------
    trace : obspy.core.Trace
        Input trace.
    oper_len : float
        The length of the AGC window in seconds.
    type_scal : str
        Which type of scaling is used on the data. Can be 'mean', 'median' or
        'RMS'
    basis : str
        Location of the window compared to each sample. Can be 'trailing', 
        'leading', 'centred'.

    Returns
    -------
    trace : obspy.core.Trace
        Trace balanced with AGC.

    """
    # Create a copy of the trace
    new_trace = trace.copy()
    
    # Go over each AGC window
    for i,window in enumerate(get_AGC_window(trace, oper_len, basis)):
        # Get the gain scaling
        scalar = AGC_scaling_val(window, type_scal)
        new_trace.data[i] = trace[i]*scalar
    return trace

def AGC_old(record, oper_len, type_scal, basis):
    """
    Automatic Gain Control balances the gain based on the amplitude in a local
    window. The function is based on the AGC function from SeisSpace ProMAX. 
    Scaling can be done based on the inverse of:
        mean
        median
        RMS 
    The location of the window can be set as:
        trailing:
            Following the sample
        leading:
            Preceding the sample
        centred:
            The sample is located at the centre of the window
    
    See also:
    https://esd.halliburton.com/support/LSM/GGT/ProMAXSuite/ProMAX/5000/5000_8/Help/promax/agc.pdf

    Parameters
    ----------
    record : obspy Stream
        The record that has to be balanced.
    oper_len : float
        Window length in seconds.
    type_scal : str
        Which type of scaling is used on the data. Can be 'mean', 'median' or
        'RMS'
    basis : str
        Location of the window compared to each sample. Can be 'trailing', 
        'leading', 'centred'.

    Returns
    -------
    new_record : obspy Stream
        The new record after application of AGC.

    """
    # Create a copy of the original data
    new_record = record.copy()
    # Get the data as an array
    data = np.array(record)
    
    # Go over each sample and create the window
    for i,window in enumerate(get_AGC_window(record, oper_len, basis)):
        # Get the scaling value for each trace
        scalars = AGC_scaling_val(window, type_scal)
        # Multiply the data with the scalars
        data[:,i] *= scalars
    
    # Now add the new data to the record
    for i,trace in enumerate(new_record):
        trace.data = data[i,:]
    
    return new_record

def AGC(record,oper_len,type_scal,basis):
    """
    Automatic Gain Control balances the gain based on the amplitude in a local
    window. The function is based on the AGC function from SeisSpace ProMAX. 
    Scaling can be done based on the inverse of:
        mean
        median
        RMS 
    The location of the window can be set as:
        trailing:
            Following the sample
        leading:
            Preceding the sample
        centred:
            The sample is located at the centre of the window
    
    See also:
    https://esd.halliburton.com/support/LSM/GGT/ProMAXSuite/ProMAX/5000/5000_8/Help/promax/agc.pdf

    Parameters
    ----------
    record : obspy Stream
        The record that has to be balanced.
    oper_len : float
        Window length in seconds.
    type_scal : str
        Which type of scaling is used on the data. Can be 'mean', 'median' or
        'RMS'
    basis : str
        Location of the window compared to each sample. Can be 'trailing', 
        'leading', 'centred'.

    Returns
    -------
    new_record : obspy Stream
        The new record after application of AGC.

    """
    # Faster methods have not yet been implemented for the other means, so they
    # use the old function
    if type_scal in ['median','RMS']:
        return AGC_old(record,oper_len,type_scal,basis)
    
    dt = record[0].stats.delta
    # The operator length in amount of data points
    oper_len_items = int(np.round(oper_len/dt+1))
    
    new_record = record.copy()
    
    # The convolution operator for the mean
    operator = np.ones(oper_len_items)
    
    # Convert the data to an array
    data = np.array(record)
    
    # Calculate how many data points are used for each point
    scal_vals = np.convolve(np.ones(data.shape[1]),operator,'full')
    
    # Convolve the data with the operator and divide by the amount of points
    # used to get the mean
    convolve = convolve2d(abs(data),operator[np.newaxis,:],'full')/scal_vals[np.newaxis,:]
    
    convolve = np.where(convolve==0,1,convolve)
    
    # Now snip out the relevant part for each method
    if basis == 'trailing':
        snipped = 1/convolve[:,oper_len_items-1:]
    elif basis == 'leading':
        snipped = 1/convolve[:,:-oper_len_items+1]
    elif basis == 'centred':
        snipped = 1/convolve[:,int(oper_len_items/2):int(-oper_len_items/2)]
    
    # Multiply the data with the scaling values
    data *= snipped
    
    # Now add the new data to the record
    for i,trace in enumerate(new_record):
        trace.data = data[i,:]
        
    return new_record

def TAR_trace(trace, power_constant):
    """
    TAR function that works on a single trace. For more information see the 
    function TAR.

    Parameters
    ----------
    trace : obspy Trace
        Input trace.
    power_constant : float
        Power constant for time raised to a power gain correction.

    Returns
    -------
    new_trace : obspy Trace
        Output trace.

    """
    # Create a copy of the trace
    new_trace = trace.copy()
    
    # Get the gain corrections
    pow_mult = trace.times()**power_constant
    # Apply it to the new trace
    new_trace.data = trace.data*pow_mult
    
    return new_trace

def TAR(record, power_constant):
    """
    Apply a very basic version of True Amplitude Recovery to a section. Based
    on the TAR function in SeisSpace ProMAX. A gain correction is applied to 
    the data as time raised to a power:
        g(t)=t^power_constant
    
    See also:
    https://esd.halliburton.com/support/LSM/GGT/ProMAXSuite/ProMAX/5000/5000_8/Help/promax/tar.pdf

    Parameters
    ----------
    record : obspy Stream
        Stream of input data.
    power_constant : float
        Power constant to which the time is raised for the gain correction.

    Returns
    -------
    new_record : obspy Stream
        Stream with TAR applied.

    """
    # Create a new record
    new_record = record.copy()
    
    # Get the gain correction for each time
    pow_mult = record[0].times()**power_constant
    
    # Multiply this with the data
    data = np.array(record)
    data = data*pow_mult[np.newaxis,:]
    
    # Now attach the information to each trace
    for i,trace in enumerate(new_record):
        trace.data = data[i,:]
    return new_record

def dom_plot(record, reclen=None, recstart = 0):
    if reclen == None:
        reclen = record[0].stats.endtime - record[0].stats.starttime
        
    record.plot(type='section',
                time_down = True,
                fillcolors = ([0.5,0.5,0.5],None),
                grid_color='white',
                recordlength=reclen,
                recordstart=recstart)
    
def ramp_func(len_ramp,idx_ramp,len_data):
    ramp = np.linspace(0,1,len_ramp)
    mult = np.zeros(len_data)
    
    idx_start = int(max(0,idx_ramp-0.5*len_ramp))
    idx_end = int(min(len_data,idx_ramp+0.5*len_ramp))
    
    idx_start_ramp = max(0,int(-(idx_ramp-0.5*len_ramp)))
    idx_end_ramp = min(len(ramp),int(len_data - idx_ramp+0.5*len_ramp))
        
    mult[idx_start:idx_end] = ramp[idx_start_ramp:idx_end_ramp]
    mult[idx_end:] = 1
    
    return mult

def logistic_func(len_ramp,idx_ramp,len_data):
    x_vals = np.linspace(1,len_data,len_data)
    act_func = 1/(1+np.exp(-(x_vals-idx_ramp)/(0.33*len_ramp)))
    
    return act_func

def trace_mute(data,method,idx_cut,len_ramp = None):
    
    if method == 'box':
        # Compute the new trace
        return np.where(np.linspace(1,len(data),len(data))-1 <= idx_cut,0,data)
    elif method == 'ramp':
        mult = ramp_func(len_ramp,idx_cut,len(data))
    elif method == 'sigmoid':
        mult = logistic_func(len_ramp,idx_cut,len(data))
    else:
        raise ValueError("Method is not used correctly, can be 'box', 'ramp' or 'sigmoid'.")
    
    return data*mult
    
def mute_cone(record,method,vel_mute,shift,len_ramp=None):
    
    dt = record[0].stats.delta
    
    methods_ramp = ['ramp','sigmoid']
    if method in methods_ramp:
        if len_ramp == None:
            raise ValueError(f"len_ramp must be defined when using the following methods: {methods_ramp}")
        else:
            len_ramp = int(len_ramp/dt)
    
    # Get the distances from the shot location
    dists = []
    for trace in record:
        dists.append(trace.stats.distance)
    dists = abs(np.array(dists))
    
    # Calculate at what index to start the trace
    cut_off_idcs = np.array(dists/vel_mute/dt,dtype=int) + int(shift/dt)
    
    new_record = record.copy()
    for i,trace in enumerate(new_record):
        # To be certain, enforce a minimum and maximum
        cut_off = max(0,cut_off_idcs[i])
        cut_off = min(len(trace)-1,cut_off)
        
        trace.data = trace_mute(trace.data,method,cut_off,len_ramp=len_ramp)
    return new_record

def levinson_recursion(autocorr, rhs):
    """
    Python implementation of the Matlab CREWES function to solve system Tx=b
    with Levinson recursion.

    Parameters
    ----------
    autocorr : numpy.ndarray
        Input autocorrelation vector. Must be fully causal
    rhs : numpy.ndarray
        Input right-hand-side vector.

    Raises
    ------
    ValueError
        Raised if the autocorrelation does not have it's maximum value at the
        first index.

    Returns
    -------
    x : np.ndarray
        Solution vector.

    """
    autocorr = autocorr.squeeze()
    # Test if autocorr has a single dimension
    if autocorr.ndim != 1:
        raise ValueError("autocorr must be a vector")
    # Make it into a column vector
    autocorr = autocorr[:,np.newaxis]
    
    rhs = rhs.squeeze()
    if rhs.ndim != 1:
        raise ValueError("rhs must be a vector")
        
    rhs = rhs[:,np.newaxis]
    
    # Normalise autocorr
    if autocorr[0] != 1.:
        autocorr = autocorr/autocorr.max()
    
    if autocorr[0] != autocorr.max():
        raise ValueError("Invalid autocorrelation, zero lag not maximum")
    
    # Initialise
    a = autocorr[1:]
    n = len(rhs)
    y = np.zeros(len(a))
    x = np.zeros(len(rhs))
    z = np.zeros(len(a))
    
    y[0] = -a[0]
    x[0] = rhs[0]
    beta = 1
    alpha = -a[0]
    
    # Main recursion loop
    for k in range(1,n):
        beta = (1 - alpha**2)*beta
        beta = beta.squeeze()
        
        mu = (rhs[k] - a[:k].T.dot(x[k-1::-1]))/beta
        mu = mu.squeeze()
        
        nu = x[:k] + mu*y[k-1::-1]

        x[:k] = nu[:k]
        x[k] = mu
        
        if k < n-1:
            # print(a[:k].T.shape, y[k::-1].shape,beta.shape)
            alpha = -(a[k] + a[:k].T.dot(y[k-1::-1]))/beta
            alpha = alpha.squeeze()
                        
            z[:k] = y[:k] + alpha * y[k-1::-1]
            y[:k] = z[:k]
            y[k] = alpha
    
    return x

def wiener_decon(trace,design_trace,n,stab=0.0001):
    """
    Python implementation of Matlab CREWES Wiener deconvolution function. 

    Parameters
    ----------
    trace : obspy.core.trace.Trace
        Input trace.
    design_trace : obspy.core.trace.Trace
        Trace used for operator design.
    n : int
        How many lags of the autocorrelation to use.
    stab : float, optional
        Stabilisation factor as fraction of zero-lag. The default is 0.0001.

    Returns
    -------
    new_trace : obspy.core.trace.Trace
        Deconvolved trace.

    """
    
    # autocorr_raw = th.TauP.cross_corr(design_trace,design_trace)
    autocorr_raw = np.correlate(design_trace,design_trace,mode='full')
    # Take causal part and right lag
    halfway = int(len(autocorr_raw)/2)
    autocorr = autocorr_raw[halfway:halfway+n]
    
    # Stabilise the autocorrelation
    autocorr[0] = autocorr[0]*(1+stab)
    autocorr = autocorr / autocorr[0]
    
    # Generate right-hand-side 
    rhs = np.zeros(len(autocorr))
    rhs[0] = 1.
    
    x = levinson_recursion(autocorr, rhs)
    x /= np.sqrt(x.T.dot(x))
    
    new_trace = obspy.Trace()
    new_trace.data = np.convolve(trace,x)[:int(-n+1)]
    new_trace.stats = trace.stats
    # new_trace = np.convolve(trace,x)
    new_trace = normalise_trace(new_trace)
    return new_trace

def wiener_decon_stream(record,design_id,n):
    new_stream = obspy.Stream()
    
    if design_id == 'all':
        for trace in record:
            new_stream += wiener_decon(trace,trace,n)
        return new_stream
    
    design_trace = record[design_id].copy()
    
    for trace in record:
        new_stream += wiener_decon(trace, design_trace, n)
    
    return new_stream

def NMO_corr(record,vel):
    times = record[0].times()[:,np.newaxis]
        
    if isinstance(vel,np.ndarray) and vel.ndim == 1:
        vel = vel[:,np.newaxis]
    
    dists = []
    for trace in record:
        dists.append(trace.stats.distance)
    dists = np.array(dists)[np.newaxis,:]
    # dists2 = np.sort(dists)
    dt = record[0].stats.delta
    
    # test = np.sqrt(np.square(dists2) / np.square(vel) + np.square(times)) - times
    idx_time = np.round(( np.sqrt(np.square(dists) / np.square(vel) + np.square(times)) ) / dt).astype(int)
    
    mask_zeroes = np.where(idx_time >= len(times), True, False)
    idx_time[mask_zeroes] = len(times)-1
    # shift2 = np.round(( np.sqrt(np.square(dists2) / np.square(vel) + np.square(times)) ) / dt)
    
    idx_space = (np.linspace(0,len(record)-1,len(record))[np.newaxis,:] + np.zeros(len(times))[:,np.newaxis]).astype(int)
    
    data = np.array(record)
    
    shift_data = data[idx_space,idx_time]
    shift_data[mask_zeroes] = 0
    
    return shift_data