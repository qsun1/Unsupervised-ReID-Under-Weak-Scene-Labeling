#!/bin/bash

basename=$(dirname $(readlink -f $0))

chmod -R 700 ${basename}/..


chmod -R 700 ${basename}

python ${basename}/j_c.py --gpu_ids 1,6 -d wpreid --num_picked 60  --logs-dir logs/j_c/ami_wp 
sleep 20

python ${basename}/j_c.py --gpu_ids 1,6 -d campus4k --num_picked 60  --logs-dir logs/j_c/ami_c
sleep 20

#python ${basename}/j_c.py --gpu_ids 4,5 -d campus4k --num_picked 60 --iters 100 --logs-dir logs/cams60_nogps_nns
#sleep 20
