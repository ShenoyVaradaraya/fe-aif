#!/bin/bash

# Open xterm windows and run each python script in them
xterm -hold -e "python3 scripts/fisheye2pano.py -v /fe8k-data/sensornet/Tue_2017-03-14_073002.avi --start_nv 0 --end_nv 1" &
xterm -hold -e "python3 scripts/fisheye2pano.py -v /fe8k-data/sensornet/Tue_2017-03-14_073002.avi --start_nv 1 --end_nv 2" &
xterm -hold -e "python3 scripts/fisheye2pano.py -v /fe8k-data/sensornet/Tue_2017-03-14_073002.avi --start_nv 2 --end_nv 3" &
xterm -hold -e "python3 scripts/fisheye2pano.py -v /fe8k-data/sensornet/Tue_2017-03-14_073002.avi --start_nv 3 --end_nv 4" &
