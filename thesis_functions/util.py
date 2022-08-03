# import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate_template

def cross_corr(master_trace, trace2):
    mast_trc = master_trace.copy()
    # corr = np.correlate(trace2.data, mast_trc.data, mode='full')
    corr = correlate_template(trace2, mast_trc, mode='full', normalize = None)
    # corr = correlate_template(trace2, mast_trc, mode='full', normalize='full')
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