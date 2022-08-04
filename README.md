# thesis_functions
Functions used for my Masters thesis. The spectrogram_alt file is ripped from the obspy function so that I could create my own plotting routine with the same function.

The code makes some assumptions of how the data looks, as it was only meant for the data in my thesis:
- The raw noise data is provided as .mseed files with a file for every station and component. Every file is 24 hours long and starts at 00:01:15 every day. This starting time can be changed with bound_day in open_cont_record. I converted the data into chunks of half an hour with every station included. The script for this is included in the example files. 
- Information on the lines and coordinates of every station is provided as a .csv (separated with ; for some reason). It contains the following rows:
  * stake - Stake number, the second number is the line identifier
  * latitude - As decimal WGS84 latitude
  * longitude - '             ' longitude
  * elevation - In metre above sea level, assume some WGS height datum
  * station_info - Information about the station, geophone name, station number and line name are included
  * station - The station number

All processing done on the data is included as example files, so the results should be reproducable if the original data was provided. The files that created the plots used in the thesis text are also included.
