# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:31:54 2022

@author: Sverre
"""
import os
import obspy
# import numpy as np


def open_seis_file(path,filename):
    """
    Opens a miniseed file with the obspy module given a path and filename. The 
    folder structure should be the same as in my thesis work. The path points to
    the location of all the files. Then there are folders for each day (so 23, 24, etc.).
    These folders contain the .miniseed files with the following filename:
        45300{station name}..0.{recording number}.{year}.{month}.{day}.{hour}.{minute}.{second}.{microseconds}.{component}.miniseed
    The station name is a four letter code.
    Recording number shows how many recordings have been done for this survey at this station
    The timestamp is the time at start of the recording. For values that can be larger or smaller than ten,
    for example the month can be 7 or 12, two numbers are always used, so that 7 is 07. For the microseconds 000.
    Component can be N, Z, E

    Parameters
    ----------
    path : string
        The path to the location of the files. The specific folder for the day does not have to be specified.
    filename : string
        The filename that needs to be opened with the format specified above.

    Returns
    -------
    record : Stream object
        The obspy Stream that is opened from the .miniseed file.

    """
    
    # See if the wildcard can be added at position 13, otherwise giving a warning.
    try:
        filename = filename[:13] + "*" + filename[14:]
    except IndexError:
        print("Filename is not long enough for my thesis work")
    # Creating the path of the fill by getting the day from the filename.
    path_full = os.path.join(path,filename[23:25],filename)
    
    # Opening the file
    return obspy.read(path_full)

def eff_day(bound_day, UTC_time):
    """
    Checks to which day's record the timestamp belongs. For example, my data starts 75 seconds after midnight.
    Then the data from a minute after midnight is recorded in the trace of the last day. To load it in,
    the previous day has to be used. 

    Parameters
    ----------
    bound_day : int or float
        How many seconds after midnight the record for the day starts.
    UTC_time : UTCDateTime object
        The time for which the effective day has to be determined

    Returns
    -------
    int or float
        A number indicating the day of the month.

    """
    # If the time is earlier than the boundary time, the previous days record has to be used, so 
    # the previous day is returned.
    if UTC_time.time < (obspy.core.UTCDateTime('2021-01-01T00:00:00')+bound_day).time:
        return (UTC_time-24*60*60).day
    # Otherwise the current day is returned
    else:
        return UTC_time.day
    
def find_days_between(date1, date2, bound_day):
    """
    Find the amount of days that has passed between two dates. bound_day edits
    when a new day starts compared to 00:00:00. For example, if bound_day is 75
    the new day starts at 00:01:15. Then the day for 2021-07-26 00:00:35 is 
    2021-07-25.

    Parameters
    ----------
    date1 : obspy.core.UTCDateTime
        Start date.
    date2 : obspy.core.UTCDateTime
        End date.
    bound_day : str
        How late after midnight the new day is considered to start.

    Returns
    -------
    days : list
        A list of days starting with the first input one and ending with the
        last input one.

    """
    # Subtract bound_day to move back a bit
    date1_eff = date1 - bound_day
    date2_eff = date2 - bound_day
    
    # Set up the list with the first day
    days = [date1_eff.day]
    new_date = date1_eff
    
    end_reached = date1_eff.date == date2_eff.date
    # end_reached = False
    
    # Iterate until the last day has been reached
    while not end_reached:
        # Add 24 hours
        new_date += 24*3600
        # Add the new day
        days.append(new_date.day)
        
        # See if the end has been reached
        if new_date.date >= date2_eff.date:
            end_reached = True
    return days

def open_cont_record(station,time_start,component,base_path,time_end=None,duration=None,bound_day=75, print_progress = True):
    """
    Merges different records from .miniseed files to get a continuous record between times. The files must have
    the structure of my thesis work as described in the function open_seis_file. The end time can be determined
    with a time stamp or a duration from the start time. 

    Parameters
    ----------
    station : string or int
        The station for which the record must be opened. Is a four letter code.
    time_start : string
        A timestamp for the desired start time of the record. Must fit the criteria for the obspy.core.UTCDateTime module.
    component : string
        Which component of the station to open, can be N, E, Z.
    base_path : string
        The path pointing to the location of the files, see also the function open_seis_file.
    time_end : string, optional
        The desired end time of the record as specified by a time stamp. Either this or the duration must be given. 
        Overwrites duration if both are given. The default is None.
    duration : float or int, optional
        The duration of the record from the start time in seconds. Either this or time_end must be given. 
        Overwritten by duration if both are given. The default is None.
    bound_day : float or int, optional
        At which time the records start each day (not counting the first day of the survey) in seconds after 00:00. The default is 75.

    Raises
    ------
    a
    ValueError
        if neither an end time stamp, nor a duration are given.

    Returns
    -------
    record : Stream object
        A merged record between the start and end time for one component.

    """
    # Convert the start time to a UTCDateTime object
    time_start_UTC = obspy.core.UTCDateTime(time_start)
    
    # Now check if time_end or duration are provided
    if time_end == None and duration == None:
        # If none are provided, raise an error
        raise ValueError("Please specify an end time with the arguments time_end or duration")
    elif time_end == None:
        # If only the duration is provided create the UTCDateTime for the end by adding the duration to the start
        time_end_UTC = time_start_UTC + duration
    else:
        # If the end time is provided (even if the duration is provided too), convert it to a UTCDateTime object
        time_end_UTC = obspy.core.UTCDateTime(time_end)
    
    if time_end_UTC < time_start_UTC:
        raise ValueError("End time specified is before start time")
    elif time_end == time_start:
        raise ValueError("Start and end times are the same")
    
    # Calculate over which days the record stretches
    days = find_days_between(time_start_UTC, time_end_UTC, bound_day)
    days_amt = len(days)
    
    if print_progress:
        # To see how long the process roughly takes, the progress is shown
        print("Loading", days_amt, "file(s):")
        print(f"0/{days_amt}", end="")
    
    # To start, add the first record, first create the right file name
    filename = f'45300{station}..0.*.{time_start_UTC.year}.{str(time_start_UTC.month).zfill(2)}.{str(int(days[0])).zfill(2)}.*.000.{component}.miniseed'
    
    # Open this file
    try:
        record = open_seis_file(base_path, filename)
    except FileNotFoundError:
        # If the file can't be found, for example because there were no recordings on that day, open an empty stream object
        record = obspy.Stream()
        
        # If this is the only files that is loaded, raise an exception, this is probably
        # not meant to be the case
        if days_amt == 1:
            raise FileNotFoundError(f"No file fitting this description could be found: \n{os.path.join(base_path,filename[23:25], filename)}")
    else:
        # Then trim the time to the right interval
        record.trim(time_start_UTC,time_end_UTC)
    
    if print_progress:
        print(f"\r1/{days_amt}",end="")
        counter = 1
    
    # Now add the other days as extra traces
    for day in days[1:]:
        filename = f'45300{station}..0.x.{time_start_UTC.year}.{str(time_start_UTC.month).zfill(2)}.{str(int(day)).zfill(2)}.*.000.{component}.miniseed'
        record += open_seis_file(base_path, filename).trim(time_start_UTC,time_end_UTC)
        
        if print_progress:
            counter += 1
            print(f"\r{counter}/{days_amt}",end="")
    # Finally merge all of  the traces
    record.merge(method=1)
    
    if print_progress:
        print("")
    
    # try:
    #     test = record[0].stats
    # except IndexError:
    #     raise IndexError("Empty stream detected, there were probably no recordings at the specified station and time")
        
    
    return record

def open_all_comp(station, time_start, base_path, time_end=None, duration=None):
    """
    Simple function that loads in the record for a specific time span and function for all three components

    Parameters
    ----------
    station : string
        The record of which station should be used
    time_start : string
        String describing the starting time, should be valid as input for a UTCDateTime object of obspy
    base_path : os path or string
        Path to the data folder
    time_end : string, optional
        String describing the end time, should be valid as input for a UTCDateTime object of obspy. Note that either this, or the duration is necessary. The default is None.
    duration : int or float, optional
        How long the record will continue after the start time in seconds. Note that either this or the end time has to be provided. The default is None.

    Returns
    -------
    record : Stream object from obspy
        The seismic record containing the N, E and Z traces for the specified time period and station.

    """
    # Define all of the components
    components = ["N","Z","E"]
    
    # Then simply open the record for all components and add them to the same Stream object
    record = obspy.Stream()
    for component in components: 
        record += open_cont_record(station, time_start, component, base_path, time_end = time_end, duration = duration)
    return record

def open_diff_stat(stations, time_start, component, base_path, time_end = None, duration = None, print_progress = True):
    """
    Create a stream with records for different stations. 

    Parameters
    ----------
    stations : list
        List containing the four letter station codes for the stations that have to be opened.
    time_start : string
        A timestamp for the desired start time of the record. Must fit the criteria for the obspy.core.UTCDateTime module.
    component : string
        Which component of the stations to open, can be N, E, Z.
    base_path : string
        The path pointing to the location of the files, see also the function open_seis_file.
    time_end : string, optional
        The desired end time of the record as specified by a time stamp. Either this or the duration must be given. 
        Overwrites duration if both are given. The default is None.
    duration : float or int, optional
        The duration of the record from the start time in seconds. Either this or time_end must be given. 
        Overwritten by duration if both are given. The default is None.

    Returns
    -------
    record : Stream object
        The obspy stream containing the records for the specified stations at the specified time interval.

    """
    record = obspy.Stream()
    
    if print_progress:
        print(f"Opening stations:\n0/{len(stations)}", end = '')
        counter = 0
    
    for station in stations:
        record += open_cont_record(station, time_start, component, base_path, time_end = time_end, duration = duration, print_progress = False)
        
        if print_progress:
            counter += 1
            print(f'\r{counter}/{len(stations)}', end = '')
    if print_progress:
        print("")
    return record