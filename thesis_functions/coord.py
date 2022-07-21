# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:33:59 2022

@author: sverr
"""

import obspy
import numpy as np
from sklearn.linear_model import LinearRegression

def read_coords(path_info):
    """
    Open file containing information on the stations and read it. Gives the 
    stake id, coordinates (as lat. - long. - elev.) and corresponding station
    numbers. 

    Parameters
    ----------
    path_info : string/path
        Path to the file containing all of the information. 

    Returns
    -------
    stakes : Numpy array with strings [no. stations]
        The stake numbers of each station location. This also contains the line ID.
    stations : Numpy array with strings [no. stations]
        The station numbers as strings. Sorted alphabetically. Indices correspond
        to the other two arrays.
    coords : Numpy array with floats [no. stations, 3]
        Array containing the coordinates of each station as latitude, longitude,
        elevation. 

    """
    # Open the file and read the information, then close it, because it is no longer needed.
    file = open(path_info, 'r')
    lines = file.readlines()[1:]
    file.close()
    
    # Get the amount of lines and initialise the desired arrays
    amt_lines = len(lines)
    stations = []
    stakes = []
    coords = np.zeros([amt_lines,3])
    
    # Go through each line, extract the information and assign it to the correct array
    for i, line in enumerate(lines):
        stake, coords[i,2], coords[i,0], coords[i,1], station_info, station = line.split(';')
        stations.append(station[:-1])
        stakes.append(stake[:])
    # XXX Change the lists to arrays with strings, lists seem to work a bit better, so maybe it will be changed.
    stations = np.array(stations, dtype = str)
    stakes = np.array(stakes, dtype = str)
    return stakes, stations, coords

def attach_coords(record, path_info):
    """
    Attach the coordinates to a provided record. For finding the coordinates,
    see the function read_coords.

    Parameters
    ----------
    record : obspy Stream object
        Stream, possibly with multiple traces that needs coordinates attached.
        Traces can have different stations and the function will still work.
    path_info : string
        Path to the information on the coordinates. Needed for the read_coords function.

    Returns
    -------
    record : obspy Stream object
        The provided stream with the coordinates attached.

    """
    # Read in the coordinate information
    stakes, stations, coords = read_coords(path_info)
    
    # Now go through the traces
    for trace in record:
        # Find the station number (in the Stream saved with five numbers, but we only need four)
        station = trace.stats.station[1:]
        # Now find on which index this station can be found, so we slice out the correct information
        mask = np.where(stations == station, True, False)
        coord_stat = coords[mask[:,np.newaxis].repeat(3,axis=1)]
        
        # Attach all of the information
        trace.stats.location = stakes[mask][0]
        trace.stats.latitude = coord_stat[0]
        trace.stats.longitude = coord_stat[1]
        trace.stats.elevation = coord_stat[2]
    return record

def WGS84_to_cart(coords):
    """
    Functions to transform coordinates from the WGS84 ellipsoid to Cartesian 
    coordinates with the centre at Earth's centre (the same point as the WGS84
    centre). Provide the coordinates in array.
    
    For the source of the calculations, see:
        https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
        https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84

    Parameters
    ----------
    coords : array [amt points, 3]
        The original WGS84 coordinates that are converted to Cartesian 
        coordinates. Should be an array with three entries in the order of 
        [latitude, longitude, elevation].

    Returns
    -------
    array [amt points, 3]
        Returns a similar array as the input in the order [X, Y, Z].

    """
    if coords.ndim == 1:
        coords = coords[np.newaxis,:]
    cart_coords = np.zeros([len(coords),3])
    
    for i in range(len(coords)):
        # Unpack numbers and transform to radians
        lat, long, elev = coords[i,:]
        lat = lat/180*np.pi
        long = long/180*np.pi
        
        # Constants for WGS84 model
        a = 6378137                                         # semi-major axis of Earth [m]
        b = 6356752.314245                                  # semi-minor axis of Earth [m]
        e2 = 1 - b**2 / a**2                                # square of first numerical eccentricity
        N = a / np.sqrt(1 - e2 * (np.sin(lat))**2)          # prime vertical radius of curvature
        
        # Convert
        cart_coords[i,0] = (N + elev) * np.cos(lat) * np.cos(long)
        cart_coords[i,1] = (N + elev) * np.cos(lat) * np.sin(long)
        cart_coords[i,2] = (b**2 / a**2 * N + elev) * np.sin(lat)
    
    # Return the Cartesian coordinates (and remove extra dimensions)
    return cart_coords.squeeze()

def WGS84_to_ENU(coords, ref_point = None):
    """
    
    See: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

    Parameters
    ----------
    coords : TYPE
        DESCRIPTION.

    Returns
    -------
    local_coords : TYPE
        DESCRIPTION.

    """
    if coords.ndim == 1:
        coords = coords[np.newaxis,:]
    local_coords = np.zeros([len(coords),3])
    
    if isinstance(ref_point, type(None)):
        ref_point = np.array([np.average(coords[:,0]), np.average(coords[:,1]), 0])
        
    lat = ref_point[0]/180*np.pi
    long = ref_point[1]/180*np.pi
    T = np.array([
        [-np.sin(long),             np.cos(long),               0           ],
        [-np.cos(long)*np.sin(lat), -np.sin(long)*np.sin(lat),  np.cos(lat) ],
        [np.cos(long)*np.cos(lat),  np.sin(long)*np.cos(lat),   np.sin(lat)]
        ])
    ref_point = WGS84_to_cart(ref_point)
    
    for i in range(len(coords)):
        
        # lat, long, elev = coords[i,:]
        # lat = lat/180*np.pi
        # long = long/180*np.pi

        coords_cart = WGS84_to_cart(coords[i,:])
        local_coords[i,:] = T.dot(coords_cart - ref_point)
    return local_coords
        
def calc_dist(coords1, coords2):
    """
    Calculates the distance between two sets of WGS84 coordinates based on
    a Cartesian straight line. Works well at short distances. 

    Parameters
    ----------
    coords1 : array
        Coordinates of the first point, should be an array containing the
        [latitude, longitude, elevation].
    coords2 : array
        Coordinates of the second point, should have the same format as the other
        point.

    Returns
    -------
    float
        Distance between the two sets of coordinates as a Cartesian straight line.

    """
    return np.linalg.norm(WGS84_to_cart(coords2) - WGS84_to_cart(coords1))

def calcCoord(path_info):
    """
    Calculates the distances between all stations with the coordinates provided.
    Information is found in a .csv file that can be read by the function read_coords.
    Returns a matrix giving the distance between two stations determined by the 
    index. 
    In the resulting matrix the point [0,1] gives the distance between the 
    coordinates on the first and second positions in the array from read_coords.
    This means that the diagonal will only contain zeroes and the matrix is 
    symmetric.

    Parameters
    ----------
    path_info : string / path
        Path to the information file.

    Returns
    -------
    dx_mat : numpy array
        Square, symmetric matrix containing the distance between stations. Index
        follows the read_coords output.
    """
    # Get the information from the file
    __, __, coords = read_coords(path_info)
    
    # if pos_mstr_trc == None:
    #     pos_mstr_trc = coords[0,:]
    
    # dist_mstr_to_stat = np.zeros(len(coords))
    
    # Initialise array
    dx_mat = np.empty([len(coords), len(coords)])
    
    # Now go over each pair of stations and calculate the distance between them
    for i,pos_stat1 in enumerate(coords):
        # dist_mstr_to_stat[i] = calc_dist(pos_mstr_trc,pos_stat1)
        for j, pos_stat2 in enumerate(coords):
            dx_mat[i,j] = calc_dist(pos_stat1,pos_stat2)
    return dx_mat

def fit_line(distances, elevations):
    """
    Fit a line through a set of points with linear regression. Used to estimate
    the elevation based on distance to some station. Can also be used to fit
    elevation to local coordinates.

    Parameters
    ----------
    distances : np.ndarray
        Array containing the distances from different stations to some station.
    elevations : np.ndarray
        Array containing the elevations of these stations.

    Returns
    -------
    coef : list or float
        The coefficient(s) of the linear regression fitting the elevations.
    intercept : float
        Intercept of the linear regression fitting the elevations.
    """
    # If there is only a single input value add a dimension so the regression
    # behaves
    if distances.ndim == 1:
        distances = distances[:,np.newaxis]
    # Set up a linear regression model
    model = LinearRegression()
    # Fit the model
    model.fit(distances, elevations)
    # Get the coefficients and intercept out
    coef = model.coef_
    intercept = model.intercept_
    
    return coef, intercept

def adapt_distances(path_info, line):
    """
    Function that finds the ordering of a line from elevation information. It
    assumes that there is an elevation in the line, which can be fitted to get
    rough indication of the geometry. This results in a distance matrix, similar
    to calcCoord that also gives negative distances for stations lower in the 
    line. 
    This can be used for the offset or for elevation corrections.

    Parameters
    ----------
    path_info : string / path
        Path to the coordinate information csv.
    line : int
        Indicator for which line to use.

    Returns
    -------
    dx_mat : array [amt stations x amt stations]
        Matrix showing the distances between stations. Index i,j gives the 
        distance from station i to station j and is negative if station j lies
        lower in the line than station i.

    """
    line = int(line)
    
    # Open the relevant information
    __, stations, coords = read_coords(path_info)
    dx_mat_full = calcCoord(path_info)
    
    # Find out to which line each station belongs
    line_id = open_line_id(path_info)
    
    # Then filter the information to use the right line
    stations = stations[line_id == line]
    coords = coords[line_id == line,:]
    dx_mat = dx_mat_full[line_id == line,:]
    dx_mat = dx_mat[:,line_id == line]
    
    # Convert to a local coordinate system for proper geometry (otherwise you
    # are working on a sphere and things are weird)    
    coords = WGS84_to_ENU(coords)

    # Find out which station is at the bottom of the hill and so at the start
    # of the line. This means the method does not work if there is no slope
    for i in range(2):
        # The stations at the edges are the ones the furthest apart
        stations_edges = np.argwhere(dx_mat == dx_mat.max())[i]
        # Now take the distance from one of the edges to all other stations
        distances = dx_mat[stations_edges[i],:]
        # Find if the slope is negative or positive
        coef, intercept = fit_line(distances, coords[:,2])
        
        # If the slope is positive, the right station is selected, otherwise
        # use the other one
        if coef >= 0:
            break
    
    # Predict the elevations for each station based on the distance to the 
    # starting station
    pred_elevs = coef*distances+intercept
    
    # Now if the elevation of another station is lower than the selected station,
    # it lies lower on the hill and thus is earlier in the line. These stations
    # get a negative offset.
    for i in range(len(stations)):
        stat_elev = pred_elevs[i]
        dx_mat[i, pred_elevs < stat_elev] *= -1
    
    # Return the corrected distance matrix
    return dx_mat

def find_outer_stats(line, path_info):
    line = int(line)
    
    __, stations, coords = read_coords(path_info)
    dx_mat_full = calcCoord(path_info)
    
    # Find out to which line each station belongs
    line_id = open_line_id(path_info)
    
    # Then filter the information to use the right line
    stations = stations[line_id == line]
    coords = coords[line_id == line,:]
    dx_mat = dx_mat_full[line_id == line,:]
    dx_mat = dx_mat[:,line_id == line]
    
    # Convert to a local coordinate system for proper geometry (otherwise you
    # are working on a sphere and things are weird)    
    coords = WGS84_to_ENU(coords)
    
    for i in range(2):
        # The stations at the edges are the ones the furthest apart
        stations_edges = np.argwhere(dx_mat == dx_mat.max())[i]
        # Now take the distance from one of the edges to all other stations
        distances = dx_mat[stations_edges[i],:]
        # Find if the slope is negative or positive
        coef, intercept = fit_line(distances, coords[:,2])
        
        # If the slope is positive, the right station is selected, otherwise
        # use the other one
        if coef >= 0:
            break
        
    return stations_edges


def open_line_id(path_info = None):
    """
    Get the array containing line identifiers either saved on disk or remake
    it by reading the coordinate information if the file cannot be found.

    Parameters
    ----------
    path_info : str or path, optional
        Path to the coordinate information. Can be left out, but if the array
        is not saved in 'basepath/Arrays/line_id.npy', the function will not 
        work. The default is None.

    Returns
    -------
    line_id : np.ndarray
        Array containing an id for each station, indicating to which line it
        belongs.

    """
    # Try if the file can be loaded in
    try:
        line_id = np.load('./Arrays/line_id.npy')
    except FileNotFoundError:
        # Otherwise read the coordinate information and get the information
        # from the stakes.
        stakes, __, __ = read_coords(path_info)
        line_id = np.zeros(len(stakes))
        for i,stake in enumerate(stakes):
            line_id[i] = int(stake[1])
    return line_id

def attach_distances(record, mtr_idx, line, path_info):
    """
    Attach distance information to each trace in a record. This is the inline 
    distance between the stations and one selected station. This only works
    for a single line (crossline distance does not translate well).

    Parameters
    ----------
    record : obspy.core.Stream
        Input stream.
    mtr_idx : int
        Index of the station that is used as the zero point.
    line : str
        String identifying which line is used.
    path_info : str or path
        Path to coordinate information.

    Returns
    -------
    record : obspy.core.Stream
        Stream with distance information attached at location 
        trace.stats.distance.

    """
    # Get the inline distance matrix (can be negative) for the line
    dx_mat = adapt_distances(path_info, line)
    # Select the row with the right station as zero.
    dist_sel = dx_mat[mtr_idx,:]
    
    # Go over each trace and attach the information
    for i, trace in enumerate(record):
        trace.stats.distance = dist_sel[i]
        
    return record

def attach_line(record, path_info):
    """
    Attaches line information to each trace in a record as the location
    (trace.stats.location). 

    Parameters
    ----------
    record : obspy.core.Stream
        Input stream.
    path_info : str or path
        Path to the coordinate information.

    Returns
    -------
    record : obspy.core.Stream
        .Stream with location information based on the line attached.

    """
    # Open the line identifiers for the traces
    line_id = open_line_id(path_info)
    # Attach the line to each trace
    for trace, line_no in zip(record, line_id):
        trace.stats.location = str(int(line_no))
    return record

def select_line(record, line, path_info):
    """
    Selects only the stations that belong to a certain line. 

    Parameters
    ----------
    record : obspy.core.Stream
        Input data with all stations.
    line : str
        String indicating which line should be selectd.
    path_info : str or path
        Path to coordinate information.

    Returns
    -------
    obspy.core.Stream
        Stream containing only the stations belonging to the specified line.

    """
    record = attach_line(record, path_info)
    return record.select(location = line)

def get_unique_lines(path_info):
    """
    Find all of the lines from the array line_id.

    Parameters
    ----------
    path_info : str or path
        Path to coordinate information.

    Returns
    -------
    lines : np.ndarray
        Array containing the name of each line as a string.

    """
    line_id = open_line_id(path_info)
    # Find the options for the lines (the conversion to int and then string is to get rid of decimal signs)
    lines = np.unique(line_id).astype('int32').astype('U')
    return lines