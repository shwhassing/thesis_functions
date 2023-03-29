# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:33:59 2022

@author: sverr
"""

# import obspy
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy

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
    
    # Constants for WGS84 model
    a = 6378137                                         # semi-major axis of Earth [m]
    b = 6356752.314245                                  # semi-minor axis of Earth [m]
    e2 = 1 - b**2 / a**2                                # square of first numerical eccentricity
    
    for i in range(len(coords)):
        # Unpack numbers and transform to radians
        lat, long, elev = coords[i,:]
        lat = lat/180*np.pi
        long = long/180*np.pi
        
        # Extra conversion value
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
    
    if ref_point == None:
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
    """
    Function that finds the index of the stations on the edges of a line. The
    index corresponds to the array stations[line_id == line]. 

    Parameters
    ----------
    line : int
        Integer indicating which line is used for the function.
    path_info : str
        Path to coordinate information.

    Returns
    -------
    stations_edges : np.ndarray
        Array containing the index of two stations that lie on the outside of 
        the line.

    """
    # Enforce line being an int
    line = int(line)
    
    # Open coordinate information
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
    
    # Try each station on an edge to see which one is on the bottom
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

def distances_mtr(mtr_idx_full, path_info):
    """
    Finds the distance of each station to the master trace. The function does
    provide crossline distances, but is not really meant for it, so these could
    be incorrect. 

    Parameters
    ----------
    mtr_idx_full : int
        Index of the master trace in the coordinate list.
    path_info : str
        Path to the coordinate information.

    Returns
    -------
    distances : np.ndarray
        Array containing the distance of each station to the master trace. 
        One direction along the line is taken as negative

    """
    # Open the relevant information
    __, stations_full, coords_full = read_coords(path_info)
    dx_mat_full = calcCoord(path_info)
    
    # Convert to a local coordinate system for proper geometry (otherwise you
    # are working on a sphere and things are weird)    
    coords_full = WGS84_to_ENU(coords_full)
    
    # mtr_idx_full = np.argwhere(stations_full == mtr_station).squeeze()
    
    # Find out which lines exist
    lines = get_unique_lines(path_info)
    
    # Find out to which line each station belongs
    line_id = open_line_id(path_info)
    
    # Initialise distance array
    distances = np.zeros(len(stations_full))
    
    # Finds the distance to the master station
    dists_raw = dx_mat_full[mtr_idx_full].squeeze()
    
    # Goes over each line
    for line in lines:
        # Enforce line being an integer
        line = int(line)
        
        # Get the indices of all stations on this line
        idcs_line = np.arange(len(stations_full), dtype=int)[line_id == line]
        
        # Then filter the information to use the right line
        stations = stations_full[line_id == line]
        coords = coords_full[line_id == line,:]
        dx_mat = dx_mat_full[line_id == line,:]
        dx_mat = dx_mat[:,line_id == line]
        dists_line = dists_raw[line_id == line]
        
        # Find out which station is at the bottom of the hill and so at the start
        # of the line. This means the method does not work if there is no slope
        for i in range(2):
            # The stations at the edges are the ones the furthest apart
            stations_edges = np.argwhere(dx_mat == dx_mat.max())[0]
            # Find which station is located at the bottom
            stat_bottom = stations[stations_edges[i]]
            # Now take the distance from one of the edges to all other stations
            dists_along = dx_mat[stations_edges[i],:]
            # Find if the slope is negative or positive
            coef, intercept = fit_line(dists_along, coords[:,2])
            
            # If the slope is positive, the right station is selected, otherwise
            # use the other one
            if coef >= 0:
                break

        # Now get the index of the station at the bottom
        stat_bot_idx = np.argwhere(stations_full == stat_bottom)
        # And find the distance from the bottom of the line to the master station
        # XXX Does not work with crossline distance
        dist_mtr_bot = dx_mat_full[stat_bot_idx,mtr_idx_full]
        
        # Predict the elevations for each station based on the distance to the 
        # starting station
        pred_elevs = coef*dists_along+intercept
        
        # Now if the elevation of another station is lower than the selected station,
        # it lies lower on the hill and thus is earlier in the line. These stations
        # get a negative offset.
        stat_elev = np.squeeze(coef*dist_mtr_bot+intercept)
        dists_line[pred_elevs < stat_elev] *= -1

        # Add the distance of this line to the array
        distances[idcs_line] = dists_line
    
    return distances

def calc_axisangle(path_info):
    """
    Calculate the angle between the axes

    Parameters
    ----------
    path_info : str
        Path to coordinate information.

    Returns
    -------
    phi : float
        Returns the angle between the axes in radians.

    """
    # Get the coordinates to local format
    __, __, glob_coords = read_coords(path_info)
    coords = WGS84_to_ENU(glob_coords)

    # Find to which line the points belong
    line_id = open_line_id(path_info)

    # Split up into the two lines
    coords0 = coords[line_id==0]
    coords1 = coords[line_id==1]

    # Fit a line through the points
    fit0 = fit_line(coords0[:,0],coords0[:,1])
    fit1 = fit_line(coords1[:,0],coords1[:,1])
    
    # Get the angle of these lines with the x-axis
    alpha0 = np.arctan(fit0[0])
    alpha1 = np.arctan(fit1[0])

    # Angle between axes
    phi = np.squeeze(abs(alpha1-alpha0))
    
    return phi

def force_cycle(array, cycle_len = 2*np.pi, cycle_start = 0):
    """
    Given the characteristics of a cycle (length and starting position), reduces
    all the values in the array to the first cycle. Meant to reduce angle to
    smallest equivalent. For example, 2.5*pi --> 0.5*pi or -0.5*pi --> 1.5*pi

    Parameters
    ----------
    array : np.ndarray
        Array with values in the cycle.
    cycle_len : float, optional
        Length of the cycle. The default is 2*np.pi.
    cycle_start : float, optional
        Starting point of the cycle. The default is 0.

    Returns
    -------
    new_cycle : np.ndarray
        Array with the corrected values

    """
    # Remove every full cycle from the value and then add the starting position
    new_cycle = array % cycle_len + cycle_start
    
    return new_cycle

def calc_x_angle(path_info):
    """
    Calculate the angle between the crossline and E-W axis

    Parameters
    ----------
    path_info : str
        Path to coordinate information.

    Returns
    -------
    phi : float
        Returns the angle between the axes in radians.

    """
    # Get the coordinates to local format
    __, __, glob_coords = read_coords(path_info)
    coords = WGS84_to_ENU(glob_coords)

    # Find to which line the points belong
    line_id = open_line_id(path_info)

    # Split up into the two lines
    coords1 = coords[line_id==1]

    # Fit a line through the points
    fit1 = fit_line(coords1[:,0],coords1[:,1])
    
    angle_xaxis = np.arctan(fit1[0][0])
    
    return angle_xaxis

def calc_backazim(dom_slow):
    """
    Calculates the back azimuth and vector length from slowness values. It 
    interprets the second column as the y-axis and the first as the x-axis. 
    Then it adds pi to get back azimuth and reduces the angle the the lowest
    option. 

    Parameters
    ----------
    dom_slow : np.ndarray
        Array containing slowness values along columns. First column contains
        values from line 0, used as y-axis and second column values from line 1

    Returns
    -------
    back_azim : np.ndarray
        Back azimuth for each slowness row in radians. Angle is taken from east
        axis counterclockwise
    veclen : np.ndarray
        Length to the centre of the coordinate system.

    """
    # Convert slowness to azimuth
    azim = np.arctan2(dom_slow[:,0],dom_slow[:,1])
    # Get the back azimuth by adding 180 degrees
    back_azim = azim + np.pi
    # Get the angles between 0 and 2*pi radians
    back_azim = force_cycle(back_azim)
    
    # And the length of the vector
    veclen = np.linalg.norm(dom_slow,axis=1)
    
    return back_azim, veclen

def axis_correction(x_coords, y_coords, angle):
    """
    Correct for the angle between the axes to make them perpendicular. Based
    on:
    https://math.stackexchange.com/questions/62581/convert-coordinates-from-cartesian-system-to-non-orthogonal-axes

    Parameters
    ----------
    x_coords : np.ndarray
        Array with x-coordinates.
    y_coords : np.ndarray
        Array with y-coordinates.
    angle : float
        Angle between the axes.

    Returns
    -------
    np.ndarray
        Array containing new x- and y-coordinates, in indices 0 and 1 respectively

    """
    return np.array([x_coords+y_coords*np.cos(angle),y_coords*np.sin(angle)])

def correct_slowness(dom_slow0, dom_slow1, path_info):
    """
    Correct and rotate slowness values to fit with coordinate system

    Parameters
    ----------
    dom_slow0 : np.ndarray
        Slowness values along the main line.
    dom_slow1 : np.ndarray
        Slowness values along the crossline.
    path_info : str
        Path to the coordinate information.

    Returns
    -------
    np.ndarray
        Corrected slowness values N-S axis on first index, E-W on second index.

    """
    # Calculate the angle between the axes
    phi = calc_axisangle(path_info)
    
    # Correct the angle to be perpendicular
    dom_slow_axiscorr = axis_correction(dom_slow1, dom_slow0, phi)

    # Get the angle of the crossline with the E-W line
    angle_xaxis = calc_x_angle(path_info)
    # Get the rotation matrix for this angle
    rot_mat = np.array([[np.cos(angle_xaxis),-np.sin(angle_xaxis)],
                        [np.sin(angle_xaxis),np.cos(angle_xaxis)]])

    # Rotate the slowness results
    dom_slow_rot = np.matmul(rot_mat[np.newaxis,:,:],dom_slow_axiscorr.T[:,:,np.newaxis]).squeeze()
    
    return dom_slow_rot[:,::-1]

class Coords:
    
    def __init__(self, path_info):
        """
        Open file containing information on the stations and read it. Provides
        the station numbers, line identifiers, latitude, longitude, elevation

        Parameters
        ----------
        path_info : str
            Path to the coordinate information in a csv

        Returns
        -------
        None.

        """
        
        # Open the file and read the information, then close it, because it is no longer needed.
        file = open(path_info, 'r')
        lines = file.readlines()[1:]
        file.close()
        
        # Get the amount of lines and initialise the desired arrays
        amt_lines = len(lines)
        stations = []
        stakes = []
        line_id = []
        coords = np.zeros([amt_lines,3])
        
        # Go through each line, extract the information and assign it to the correct array
        for i, line in enumerate(lines):
            stake, coords[i,2], coords[i,0], coords[i,1], station_info, station = line.split(';')
            stations.append(station[:-1])
            stakes.append(stake[:])
            line_id.append(int(stake[1]))
            
        self.stations = np.array(stations, dtype = str)
        self.stakes = np.array(stakes)
        self.amt_stats = len(stations)
        self.line_id = np.array(line_id, dtype = int)
        self.lat = coords[:,0]
        self.long = coords[:,1]
        self.elev = coords[:,2]
        self.lines = np.unique(self.line_id)

    def __call__(self):
        print(f"Coordinate list with:\nStations:\t\t{self.stations.size}\nLines: \t\t\t{self.lines}")
        for line in self.lines: 
            print(f'Stats. line {line}\t{np.sum(np.where(self.line_id == line,1,0))}')
    
    def __getitem__(self, key):
        new = deepcopy(self)
        
        new.stations = np.array(new.stations[key])
        new.stakes = np.array(new.stakes[key])
        new.amt_stats = new.stations.size
        new.line_id = np.array(new.line_id[key])
        new.lat = np.array(new.lat[key])
        new.long = np.array(new.long[key])
        new.elev = np.array(new.elev[key])
        new.lines = np.array(np.unique(new.line_id))
        
        return new

    def coords(self):
        # Get the coordinates in a single array
        return np.stack([self.lat,self.long,self.elev]).T
    
    def line_mask(self, line):
        # Get a mask for the coordinates to select a single line
        return np.where(self.line_id == line, True, False)
    
    def select_line(self, line):
        # Create Coord object with only coordinates from specified line
        return self[self.line_mask(line)]
    
    def line_stream(self, record, line):
        # Select only traces from selected line in provided stream
        record = self.attach_distances(record, line)
        return record.select(location = str(line))
    
    def cart(self):
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
                
        # Convert angles to radians
        lat = self.lat/180*np.pi
        long = self.long/180*np.pi
        
        # Constants for WGS84 model
        a = 6378137                                         # semi-major axis of Earth [m]
        b = 6356752.314245                                  # semi-minor axis of Earth [m]
        e2 = 1 - b**2 / a**2                                # square of first numerical eccentricity
        N = a / np.sqrt(1 - e2 * (np.sin(lat))**2)          # prime vertical radius of curvature
        
        # Convert coordinates
        cart_coords = np.array([
            (N + self.elev) * np.cos(lat) * np.cos(long),
            (N + self.elev) * np.cos(lat) * np.sin(long),
            (b**2 / a**2 * N + self.elev) * np.sin(lat)
            ])
        
        return cart_coords.T
    
    def ENU(self, ref_point = None):
        """
        Converts coordinates to East-North-Up coordinate system centred on 
        ref_point. If none is provided, the average of all coordinates is taken.
        
        See: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

        Parameters
        ----------
        coords : th.coords.coords
            The coordinate object.

        Returns
        -------
        local_coords : np.ndarray
            Array containing all the coordinates in the object transformed
            to a local ENU system.

        """
        # If no point is defined, take the average
        if ref_point == None:
            ref_point = np.array([np.average(self.lat), np.average(self.long), 0])
        
        # Convert to radians
        lat = ref_point[0]/180*np.pi
        long = ref_point[1]/180*np.pi
        # Get transformation matrix
        T = np.array([
            [-np.sin(long),             np.cos(long),               0           ],
            [-np.cos(long)*np.sin(lat), -np.sin(long)*np.sin(lat),  np.cos(lat) ],
            [np.cos(long)*np.cos(lat),  np.sin(long)*np.cos(lat),   np.sin(lat)]
            ])
        
        # XXX Hacky solution to more easily convert coordinates to Cartesian
        # coordinates
        ref_coord = self[0]
        ref_coord.lat = ref_point[0]
        ref_coord.long = ref_point[1]
        ref_coord.elev = ref_point[2]
        ref_point = ref_coord.cart()
        
        coords_cart = self.cart()
        
        # Convert the coordinates to the local system by matrix multiplication
        # of T with the coordinates minus the reference point
        local_coords = T.dot(coords_cart.T - ref_point[:,np.newaxis])
        
        # Transpose the result to get the original format back
        return local_coords.T

    def attach_coords(self, record):
        """
        Attach coordinate information to the traces in the provided stream

        Parameters
        ----------
        record : obspy.Stream
            Stream containing traces for which the coordinate information must
            be set. The station number must be found in trace.stats.station[1:]
            for the function to work.

        Returns
        -------
        record : obspy.Stream
            Stream with the coordinate information included.

        """
        # Go through the traces
        for trace in record:
            # Find the station number (in the Stream saved with five numbers, but we only need four)
            station = trace.stats.station[1:]
            # Now find on which index this station can be found, so we slice out the correct information
            mask = np.where(self.stations == station, True, False)
            # coord_stat = coords[mask[:,np.newaxis].repeat(3,axis=1)]
            
            # Attach all of the information
            trace.stats.location    = self.stakes[mask][0]
            trace.stats.latitude    = self.lat[mask].squeeze()
            trace.stats.longitude   = self.long[mask].squeeze()
            trace.stats.elevation   = self.elev[mask].squeeze()
            trace.stats.location    = str(self.line_id[mask].squeeze())
        
        return record
    
    def attach_distances(self, record, line, mtr_idx = None):
        """
        Attach distance information to traces. The function will result in a 
        stream with only traces from the selected line included. The distance
        is defined as the distance to a station found at index mtr_idx

        Parameters
        ----------
        record : obspy.Stream
            Input stream.
        line : str
            For which line the distance information must be attached.
        mtr_idx : int, optional
            The distance that is attached is relative to the station at this 
            index. By default, the station at the bottom of each line is taken.
            The default is None.

        Returns
        -------
        record : obspy.Stream
            The stream with distance information attached.

        """
        # If mtr_idx is not defined set it to the bottom station of the line
        if mtr_idx == None:
            mtr_idx = self.find_bottom(line)
        
        # Get the distance matrix for the selected line
        dx_mat = self.select_line(line).dx()
        
        # Select the row with the right station as zero.
        dist_sel = dx_mat[mtr_idx,:]
        
        # Go over each trace and attach the information
        for i, trace in enumerate(record):
            trace.stats.distance = dist_sel[i]
            
        return record
    
    def dx(self):
        """
        Finds the distance matrix between each station by computing the straight 
        line distance between the Cartesian coordinates


        Returns
        -------
        np.ndarray
            [amt_stations x amt_stations] Matrix with the distance between station
            i and station j on element ij. This means that the diagonal is only
            zero.

        """

        return np.linalg.norm(self.cart()[np.newaxis,:,:] - self.cart()[:,np.newaxis,:], axis=2)
    
    def find_outer(self, line):
        """
        Finds the indices of the stations at the outer ends of a line. Note 
        that these indices may (somewhat unintuitively) not fit with the object
        it was used on. It provides the right indices when using 
        self.select_line(line)
        
        The first of the two indices lies at the bottom of the line, while
        the second is found at the top
        
        Parameters
        ----------
        line : int
            Indicate for which line you want the outer indices.

        Returns
        -------
        stations_edges : np.ndarray
            Array containing the two indices.

        """
        # Finds the indices of the stations on the outer ends of a line
        
        # Select only the right line
        coord_line = self.select_line(line)
        
        # Get the distance matrix
        dx_mat = self.select_line(line).dx()
        
        # Find out which station is at the bottom of the hill and so at the start
        # of the line. This means the method does not work if there is no slope
        for i in range(2):
            # The stations at the edges are the ones the furthest apart
            stations_edges = np.argwhere(dx_mat == dx_mat.max())[i]
            # Now take the distance from one of the edges to all other stations
            distances = dx_mat[stations_edges[i],:]
            # Find if the slope is negative or positive
            coef, intercept = fit_line(distances, coord_line.ENU()[:,2])
            
            # If the slope is positive, the right station is selected, otherwise
            # use the other one
            if coef >= 0:
                break
        
        return stations_edges
    
    def find_bottom(self, line):
        # Find the station at the bottom of a line
        stations_edges = self.find_outer(line)
        
        return stations_edges[0]
    
    def fit_line(self, line):
        """
        Fit a line through the coordinates on a single line in ENU coordinates.
        Returns intercept and coefficient

        Parameters
        ----------
        line : int
            Over the coordinates of which geophone line the fit is performed.

        Returns
        -------
        coef : np.ndarray
            Coefficient for the line.
        intercept : float
            Intercept for the line.

        """
        # Select only a line
        coord_line = self.select_line(line)
        
        # Get the distance from the bottom station
        dx_mat = self.select_line(line).dx()
        distances = dx_mat[self.find_bottom(line),:]
        
        # Now fit a line throug this
        coef, intercept = fit_line(distances, coord_line.ENU()[:,2])
            
        return coef, intercept
    
    def adapt_dx(self, line):
        """
        Adapt the distance matrix to include negative distances. Distance 
        towards the bottom of the line is defined as negative. This only works
        when selecting stations that fall on the same geophone line

        Parameters
        ----------
        line : int
            For which line the adapted distance matrix is calculated.

        Returns
        -------
        dx_mat : np.ndarray
            Matrix that contains the distance between station i and station j
            at index i,j.

        """
        # Adapts the distance matrix for all stations on a single line to get
        # negative distances
        coord_line = self.select_line(line)
        
        # Get the distance to the bottom of the line
        dx_mat = self.select_line(line).dx()
        distances = dx_mat[self.find_bottom(line),:]

        coef, intercept = self.fit_line(line)
        
        # Predict the elevations for each station based on the distance to the 
        # starting station
        pred_elevs = coef*distances+intercept
        
        # Now if the elevation of another station is lower than the selected station,
        # it lies lower on the hill and thus is earlier in the line. These stations
        # get a negative offset.
        for i in range(len(coord_line.stations)):
            stat_elev = pred_elevs[i]
            dx_mat[i, pred_elevs < stat_elev] *= -1
            
        return dx_mat
    
    def adapt_full_dists(self, mtr_idx):
        """
        A version of adapt_dx where the full distance matrix is used. This
        only works at the crossing of the two geophone lines. Obviously, this
        does not fully work to map 2D distances to 1D values. 

        Parameters
        ----------
        mtr_idx : int
            Index of the station for which the distance must be determined. 
            Negative distances are included towards the bottom of the lines. 

        Returns
        -------
        distances : TYPE
            DESCRIPTION.

        """
        # Get distance matrix
        dx_mat_full = self.dx()
        
        dists_raw = dx_mat_full[mtr_idx,:]
        
        distances = np.zeros(self.amt_stats)
        
        # Go over both lines
        for line in self.lines:
            # Get the idcs of the stations on this line
            idcs_line = np.arange(self.amt_stats)[self.line_mask(line)]
            
            # Get the distances for this line
            dists_line = dists_raw[self.line_mask(line)]
            
            # Fit a line
            coef, intercept = self.fit_line(line)
            
            # Get a distance matrix for this line
            dx_mat = self.select_line(line).dx()
            
            # Get the distances along the line
            dists_along = dx_mat[self.find_bottom(line),:]
            
            # Predict the elevations for each station based on the distance to the 
            # starting station
            pred_elevs = coef*dists_along+intercept
            
            # Determine which station is the bottom of the line
            bot_station = self.select_line(line).stations[self.find_bottom(line)]
            # Get the index of this bottom station on the original distance array
            bot_idx_full = np.argwhere(bot_station == self.stations)
            # Get all distances between the master trace and the bottom of the line
            dist_mtr_bot = dx_mat_full[mtr_idx,bot_idx_full]
            
            # Now if the elevation of another station is lower than the selected station,
            # it lies lower on the hill and thus is earlier in the line. These stations
            # get a negative offset.
            stat_elev = np.squeeze(coef*dist_mtr_bot+intercept)
            dists_line[pred_elevs < stat_elev] *= -1

            # Add the distance of this line to the array
            distances[idcs_line] = dists_line
        
        return distances
            