#!/bin/bash

basename=$(dirname $(readlink -f $0))

chmod -R 700 ${basename}/..


chmod -R 700 ${basename}

python ${basename}/j_c.py --gpu_ids 2,3 -d wpreid --num_picked 60 --iters 100 --logs-dir logs/wp60_nogps_nns 
sleep 20

#python ${basename}/j_c.py --gpu_ids 2,3 -d campus4k --num_picked 60 --iters 100 --logs-dir logs/cams60_nogps_nns
#sleep 20
