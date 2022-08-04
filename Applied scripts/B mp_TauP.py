from multiprocessing import Pool
import obspy
import time
import os
import numpy as np
import glob
from functools import partial
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
import csv

# File used for the illumination analysis. Input parameters are found inside
# if statement

# Note that the multiprocessing module does not work with interactive 
# interpreters like iPython, for example, Spyder. I ran it in PyCharm where it
# did work

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

# Required for multiprocessing to not make infinite subprocesses
if __name__ == '__main__':
    # See at which time the program starts
    start = time.time()

    #%% Input parameters

    window_length = 10.
    # Path to coordinate information
    path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')
    # Path to the raw data
    base_path = os.path.normpath('E:\\Thesis\\clip_data')
    # Path where the output of the script is put
    path_out = os.path.normpath("E:\\Thesis\\Arrays")

    component = 'E'
    mtr_idx = 86

    amt_p_vals = 2400  # amount of slowness values tested, see it as the resolution
    min_vel = 200  # minimum velocity [m/s]
    p_range = np.linspace(-1 / min_vel, 1 / min_vel, amt_p_vals)

    #%% The code

    stations = np.load('./Arrays/Stations.npy')
    # Get the bounds for when all stations were active
    low_lim, upp_lim = find_strict_limits(30 * 60)

    # Set up the edge frequencies for the notch filters
    f_ranges = [[0.0001,1]]
    width = 2
    centres = [21, 42, 63]
    for centre in centres:
        f_ranges.append([centre - width, centre + width])

    # Set up the header of the output file
    header = ['Start', 'End', 'Master', 'Line', 'Dom. vel', 'Min. val', 'Max val.']
    # Set the date format for in the file
    date_format = '%Y-%m-%d - %H-%M-%S'

    # Attach a line identifier to the traces
    line_id = th.coord.open_line_id(path_info)
    lines = np.unique(line_id).astype('int32').astype('U')  # Get all lines
    
    # Get the distance of each station to the master station
    dx_mat = th.coord.calcCoord(path_info)
    distances = dx_mat[mtr_idx, :]

    print("Preparation finished...")

    # Get a list with all of the folders in the base path
    folder_list = glob.glob(os.path.join(base_path, '*'))
    print(f"Progress:\n0/{len(folder_list)}", end="")
    for i, folder in enumerate(folder_list):
        day = os.path.split(folder)[-1]
        # Now go over each file in this folder
        file_list = glob.glob(os.path.join(folder, f'*.{component}.mseed'))
        
        # Set up the output file so that it closes correctly on a crash
        with open(os.path.join(path_out, f'Log day {day} - master {stations[mtr_idx]}{component}.txt'),
                  'w',
                  newline='') as out_file:
            # Start up the csv writer in the file
            csv_writer = csv.writer(out_file)
            csv_writer.writerow(header)
            
            for j, file in enumerate(file_list):
                print(f'\r{i}/{len(folder_list)}\t[{j}/{len(file_list)}]\tOpening file...              ', end='')

                # Get the start time of this data file, if it is outside of the
                # specified bounds, skip it
                start_time = obspy.UTCDateTime(os.path.split(file)[-1][:-8].replace('.', ''))
                if start_time < low_lim or start_time > upp_lim:
                    continue
                
                # Open the file
                record = obspy.read(file)

                # Attach line information to the record
                for trace, line_no in zip(record, line_id):
                    trace.stats.location = str(int(line_no))
                
                # Filter the record
                for f_range in f_ranges:
                    record = record.filter('bandstop',
                                           freqmin=f_range[0],
                                           freqmax=f_range[1],
                                           corners=4)
                # record = record.filter('bandpass',
                #                        freqmin=5,
                #                        freqmax=40,
                #                        corners=5)

                print(f'\r{i}/{len(folder_list)}\t[{j}/{len(file_list)}]\tCalculating dom. slowness...', end='')

                # Initiate the parallel processing with 14 cores
                with Pool(14) as p:
                    
                    # Although it looks confusing, this is a loop over 
                    # record.slide(...) and in the loop the function 
                    # th.TauP.process_window is called with a lot of fixed
                    # arguments.
                    results = p.map(partial(th.TauP.process_window,
                                        amt_stations=len(record),
                                        lines=lines,
                                        mtr_idx=mtr_idx,
                                        window_length=window_length,
                                        p_range=p_range,
                                        distances=distances,
                                        line_id=line_id
                                        ),
                                record.slide(window_length=window_length,
                                             step=window_length
                                             )
                                )
                
                # The result of the loop is written into the file
                th.TauP.write_results(results, f'{stations[mtr_idx]}{component}', csv_writer, date_format)

    print(f'\r{i+1}/{len(folder_list)}\t[{j+1}/{len(file_list)}]\tCalculating dom. slowness...')
    print(f'Program took {time.time()-start} s')