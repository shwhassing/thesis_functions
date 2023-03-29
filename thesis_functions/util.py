import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate_template
from scipy.signal import fftconvolve
from os.path import split

def cross_corr(master_trace, trace2):
    """
    Function that crosscorrelates trace2 with the master trace. A new trace
    is returned. 

    Parameters
    ----------
    master_trace : obspy.core.trace.Trace
        This trace will be second in the crosscorrelation.
    trace2 : obspy.core.trace.Trace
        This trace will be first in the crosscorrelation.

    Returns
    -------
    trace_corr : obspy.core.trace.Trace
        The trace resulting from the crosscorrelation.

    """
    # Make a copy of the master trace
    mast_trc = master_trace.copy()
    
    # Crosscorrelate, a few alternatives are commented out
    # corr = np.correlate(trace2.data, mast_trc.data, mode='full')
    corr = correlate_template(trace2, mast_trc, mode='full', normalize = None)
    # corr = correlate_template(trace2, mast_trc, mode='full', normalize='full')
    
    # Initialise a new trace
    trace_corr = obspy.Trace()
    
    # Copy over the relevant information
    trace_corr.data = corr
    trace_corr.stats.starttime = master_trace.stats.starttime - (master_trace.stats.endtime - master_trace.stats.starttime)
    trace_corr.stats.station = trace2.stats.station
    trace_corr.stats.network = master_trace.stats.network
    trace_corr.stats.delta = master_trace.stats.delta
    trace_corr.stats.channel = master_trace.stats.channel
    trace_corr.stats.location = trace2.stats.location
    
    return trace_corr

def time_from_sec(seconds):
    """
    Get the amount of hours, minutes and seconds in a certain number of seconds.

    Parameters
    ----------
    seconds : float
        Amount of seconds that is converted.

    Returns
    -------
    int, int, int
        The amount of hours, minutes and seconds in the input amount of seconds.
        Output seconds is rounded down.

    """
    hours = seconds // 3600
    minutes = (seconds - hours*3600) // 60
    seconds_left = (seconds - hours*3600 - minutes*60)
    return int(hours), int(minutes), int(seconds_left)

def tfs_string(seconds):
    """
    Takes an amount of seconds and returns a string that indicates how many
    hours, minutes and seconds the input time represents.

    Parameters
    ----------
    seconds : float
        Amount of seconds.

    Returns
    -------
    str
        String indicating amount of time past in a more readable manner.

    """
    hours, minutes, seconds_left = time_from_sec(seconds)
    return f'{hours}h. {minutes}m. {seconds_left}s.'

def find_closest(array,value):
    """
    Simple function that finds the entry in an array that is closest to the 
    provided value. Returns an index

    Parameters
    ----------
    array : np.ndarray
        Array with values.
    value : float
        Value where the closest entry needs to be found in array.

    Returns
    -------
    idx : int
        Index of the entry in array that is closest to value.

    """
    # The 'distance' to each entry in the array
    dists = abs(array-value)
    # The index(/ices) of the smallest value
    idx = np.argwhere(dists==dists.min())
    return idx

def fftcorrelate(signal1, signal2):
    """
    Convert fftconvolve to fftcorrelate by time reversing the second signal

    Parameters
    ----------
    signal1 : np.ndarray
        First signal.
    signal2 : np.ndarray
        Second signal.

    Returns
    -------
    np.ndarray
        Crosscorrelated signal.

    """
    return fftconvolve(signal1, signal2[:,::-1], mode="full")

def stream_to_array(stream):
    """
    Simple function that converts a stream to an array. It is the equivalent of
    np.array(stream), but significantly faster. Assumes all traces have the
    same length, time step and start at the same time. 

    Based on timeit, this function is 52+/-1.5 times faster than np.array()

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Input stream to be converted to an array.

    Returns
    -------
    array : np.ndarray
        Output array.

    """
    array = np.zeros([len(stream),stream[0].stats.npts])
    
    for i,trace in enumerate(stream):
        array[i,:] = trace.data
    
    return array

def find_strict_limits(prec):
    """
    Simple function that finds the time where all stations were active. It
    loads a file called Durations.npy, which simply contains the start and end
    times for each station.

    Parameters
    ----------
    prec : float
        The precision to work with. This should be the size of the data blocks
        you work with. For me that is half an hour, so 30*60.

    Returns
    -------
    lower : obspy.core.UTCDateTime
        The lower bound for when all stations were active.
    upper : obspy.core.UTCDateTime
        The upper bound for when all stations were active.

    """
    # Load the durations of all stations
    durations = np.load('Multi-processing/Arrays/Durations.npy')
    # Get the minimum and maximum for the start and end
    dur_edges = np.array([durations[:, 0].max(),
                          durations[:, 1].min()])

    # Round up/down and transform to UTCDateTime objects
    lower = obspy.core.UTCDateTime((dur_edges[0] // prec + 1) * prec)
    upper = obspy.core.UTCDateTime((dur_edges[1] // prec) * prec)

    return lower, upper

def control_filenames(files, lower, upper):
    """
    Function that extracts the date from each noise panel file and selects it 
    based on the limits provided by find_strict_limits. Returns a boolean array 
    used as indexing for the file list.

    Parameters
    ----------
    files : list
        List with filenames that are considered
    lower : obspy.core.UTCDateTime
        Lower limit
    upper : obspy.core.UTCDateTime
        Upper limit

    Returns
    -------
    mask : np.ndarray
        Boolean array used as index for the file list
    """
    # Initialise list
    mask = []

    # Go over each file
    for file in files:
        # Convert the filename to a UTCDateTime object
        date = obspy.core.UTCDateTime(split(file)[-1][:-8].replace('.', ''))

        # Now select the file if it fits within the limits
        if lower <= date < upper:
            mask.append(True)
        else:
            mask.append(False)

    return np.asarray(mask)

def zero_idcs(record):
    """
    For a 2D stream where the traces end with an irregular amount of zeros, 
    find the indices where the first of these trailing zeros can be found.

    Parameters
    ----------
    record : obspy.Stream
        Stream containing the trace data, where the trailing zeros must be 
        identified.

    Returns
    -------
    idcs : np.ndarray
        Array containing the index of the first trailing zero of each trace.

    """
    # Convert the stream to an array
    data = stream_to_array(record)
    
    # An array indicating whether to continue with this row
    cont = np.ones(data.shape[0], dtype=bool)
    # An array containing the last index containing a zero for this row
    idcs = np.zeros(data.shape[0], dtype=int) + data.shape[0]
    
    # Go backwards over the indices for each trace
    for i in np.arange(data.shape[1] - 1, 0, -1):
        # Get the mask of locations where there is a zero and we want to continue
        mask = np.where(np.logical_and(cont,data[:,i] == 0.), True, False)
        
        # Update the last zero index for these locations
        idcs[mask] = i
        # Now set the locations where cont is still True, but there was not zero
        # to False
        cont = np.where(np.logical_and(cont,np.logical_not(data[:,i] == 0)), False, cont)
        
        # If all entries in cont are False, stop the loop
        if not np.any(cont):
            break
    return idcs

def zero_mask(record):
    """
    Gives a mask that identifies the indices of a stream (converted to array)
    where trailing zeros can be found. Trailing zeros are zeros at the end of a
    trace.

    Parameters
    ----------
    record : obspy.Stream
        Stream for which the mask is designed.

    Returns
    -------
    mask : np.ndarray
        Boolean array that has the same size as np.array(record).

    """
    # Get the last index containing a zero for each trace
    idcs = zero_idcs(record)
    
    # Now create a mask for the data array with True at all trailing zeroes
    mask = np.arange(record[0].stats.npts)[np.newaxis,:] >= idcs[:,np.newaxis]
    
    return mask

def apply_trailing_zeros(record, masking_rec):
    """
    Based on a masking record (masking_rec), identify trailing zeros and assert
    these in the other provided stream (record). This can be used to force 
    end values to be zero in a bandpass filtered stream. 

    Parameters
    ----------
    record : obspy.Stream
        Stream for which the trailing zeros are applied. 
    masking_rec : obspy.Stream
        Stream where the trailing zeros are found.

    Returns
    -------
    new_rec : obspy.Stream
        A version of record where trailing zeros are applied.

    """
    # Get the mask for all locations that should be zero
    mask = zero_mask(masking_rec)
    
    # Convert the stream to an array
    data = stream_to_array(record)
    # Set the right locations to zero
    data[mask] = 0
    
    # Initialise a new stream with the same data as the original one
    new_rec = record.copy()
    for i,trace in enumerate(new_rec):
        # Set the data of all traces to then new data
        trace.data = data[i,:]
    
    return new_rec