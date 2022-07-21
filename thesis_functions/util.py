import numpy as np
import obspy

def auto_filter(record):
    f_ranges = []
    width = 2
    centres = [21,42,63]
    for centre in centres:
        f_ranges.append([centre-width, centre+width])
        
    for f_range in f_ranges:
        record = record.filter('bandstop', 
                               freqmin=f_range[0],
                               freqmax=f_range[1],
                               corners=4)
    
    return record

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