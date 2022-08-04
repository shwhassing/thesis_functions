import obspy
import os
import numpy as np
working_dir = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\Scripts')
os.chdir(working_dir)
import thesis_functions as th
from multiprocessing import Pool
import glob
from functools import partial

def cross_corr_all(mtr_idx,panel, path_info, slows):
    """
    Function that handles the crosscorrelations of a single virtual shot location
    for a single panel. 

    Parameters
    ----------
    mtr_idx : int
        Index of the master trace in the panel.
    panel : obspy.core.stream.Stream
        Noise panel that will be crosscorrelated.
    path_info : str
        Path to the coordinate information.
    slows : np.ndarray
        Array containing the dominant slowness found in the illumination 
        analysis for each line.

    Returns
    -------
    result : dict
        A dictionary containing the results of the crosscorrelation. Has the
        following entries:
            Line - line id for this virtual shot gather
            Shot_loc - Index of the virtual shot location
            Data - Array containing the data for the virtual shot gather

    """
    # Get the master trace at the virtual shot location
    master_trace = panel[mtr_idx]
    # Determine which line it belongs to
    line = master_trace.stats.location
    # And the station number of the master trace
    mtr_stat = master_trace.stats.station
    
    # Select only the traces of the right line
    panel_sel = th.coord.select_line(panel, line, path_info)

    # Find at which index the master trace is found in the new record
    for idx, trace in enumerate(panel_sel):
        if trace.stats.station == mtr_stat:
            new_j = idx
            break
        
    # Set up a new stream
    record_corr = obspy.Stream()
    # Crosscorrelate all traces and add to the new stream
    for trace in panel_sel:
        trace_corr = th.TauP.cross_corr(trace, master_trace)
        record_corr += trace_corr

    # Attach the distance to the shot location to each trace
    record_corr = th.coord.attach_distances(record_corr, new_j, line, path_info)

    # Apply TRBI with the slowness of the panel along this line
    add_line = th.proc.flip_shot(record_corr, slows[int(line)])

    # Add the result to a dictionary
    result = {'Line': line,
              'Shot_loc': new_j,
              'Data': np.array(add_line)}

    return result

# File that creates the virtual common shot gathers. They are saved at a 
# specified location

if __name__ == '__main__':
    
    # Master trace used for illumination analysis
    mtr_station = "7149"
    # Path to raw data
    path_base = os.path.normpath('E:/Thesis/clip_data/')
    # Path to coordinate information
    path_info = os.path.normpath('H:\\Onderwijs\\TU Delft\\2-3 Master\'s thesis\\topografia.csv')
    # Path where the results are saved, a new folder will be created called
    # 'Crosscorr {min_vel}'
    path_saved = os.path.normpath("E:/Thesis/Arrays/")
    
    # Load extra information
    stations = np.load('./Arrays/Stations.npy')
    line_id = np.load('./Arrays/line_id.npy')
    
    # Which streams to return and then save
    return_stream = 'all'
    
    # Minimum apparent velocity used to select noise panels
    min_vel = 5000
    # Length of the noise panels
    window_length = 10.
    # Component used for the illumination analysis
    component = 'Z'
    # Which illumination analysis to use, can be '' for the first one and
    # ' - filtered' for the second
    added_string = ' - filtered'
    # Whether to print the progress the script is making
    print_progress = True

    #%%
    
    # Extract the results of the illumination analysis
    start_time, __, dom_slow0, dom_slow1 = th.proc.extract_results(path_saved, mtr_station, component, added_string)

    # Select the right noise panels
    mask = th.proc.select_panels(dom_slow0, dom_slow1, min_vel)
    
    # Convert the dates to another format
    times_sel = th.proc.convert_date(start_time[mask],'obspy')
    # And select the right slowness for the panels
    dom_slow = np.stack([dom_slow0, dom_slow1]).swapaxes(0,1)
    dom_slow_sel = dom_slow[mask,:]

    # Read one file to get some information
    record = obspy.read(glob.glob(os.path.join(path_base,'*','*.mseed'))[0])
    
    # Initialise all virtual shot gathers, for each line and each receiver 
    # location
    virt_shots = [np.zeros([np.sum(line_id==0),np.sum(line_id==0),int(record[0].stats.sampling_rate*window_length+1)]),
                  np.zeros([np.sum(line_id==1),np.sum(line_id==1),int(record[0].stats.sampling_rate*window_length+1)])]

    # Go through every folder of data
    folder_list = glob.glob(os.path.join(path_base,'*'))

    if print_progress:
        counter = 0
        print(f"Progress:\n0/{len(times_sel)}", end='')

    for folder in folder_list:
        
        # Find every file in the folder
        file_list = glob.glob(os.path.join(folder,f'*.{component}.mseed'))

        for file in file_list:
            # Get all panels that fall in this data file
            mask_chunk = th.proc.times_mask(times_sel,os.path.split(file)[-1])
            slows = dom_slow_sel[mask_chunk,:]
            times_chunk = times_sel[mask_chunk]

            # If there are no times selected, skip this file
            if len(times_chunk) == 0:
                continue
            
            # Read the file
            record = obspy.read(file)
            # Attach line information
            record = th.coord.attach_line(record,path_info)
            # Apply the right filters
            record = th.filt.apply_filters(record)
            
            # Now go through the selected noise panels in this file
            for i,panel in enumerate(th.proc.get_panel(record,times_chunk,window_length)):
                
                # Normalise the panel
                panel = th.proc.normalise_section(panel)
                
                # Parallel processing to crosscorrelate each panel
                with Pool(14) as p:
                    results = p.map(partial(cross_corr_all,
                                            panel=panel,
                                            path_info=path_info,
                                            slows=slows[i,:]),
                                    range(len(panel)))
                
                # Extract all of the results and stack on the right location
                for result in results:
                    line = result['Line']
                    loc = result['Shot_loc']
                    data = result['Data']

                    virt_shots[int(line)][loc,:,:] += data

                if print_progress:
                    counter += 1
                    print(f"\r{counter}/{len(times_sel)}     ",end='')
    if print_progress:
        print(f"\r{counter}/{len(times_sel)}     ")

        print("Saving...")
    
    # Now go over each line and convert the stacked arrays to .mseed files
    lines = th.coord.get_unique_lines(path_info)
    for line in lines:
        # First convert to a stream
        streams = th.proc.convert_shotdata(virt_shots[int(line)], record, line, path_info)
        # Then save as .mseed
        th.proc.save_shotdata(path_saved,streams,line,min_vel)
        print(f'\rSaved line {line}', end='')
