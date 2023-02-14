#!/bin/bash

basename=$(dirname $(readlink -f $0))

chmod -R 700 ${basename}/..

chmod -R 700 ${basename}

#python ${basename}/j_d.py -d wpreid --gpu_ids 6,7 --logs-dir logs/kapa/wp_9 -k 0.9
#sleep 20

python ${basename}/j_o.py -d wpreid --gpu_ids 0,5 --logs-dir logs/kapa/wp_7 -k 0.7
sleep 20

python ${basename}/j_o.py -d wpreid --gpu_ids 0,5 --logs-dir logs/kapa/wp_5 -k 0.5
sleep 20

python ${basename}/j_o.py -d wpreid --gpu_ids 0,5 --logs-dir logs/kapa/wp_3 -k 0.3
sleep 20

python ${basename}/j_o.py -d wpreid --gpu_ids 0,5 --logs-dir logs/kapa/wp_1 -k 0.1
sleep 20

# campus4k
python ${basename}/j_o.py -d campus4k --gpu_ids 0,5 --logs-dir logs/kapa/c_1 -k 0.1
sleep 20

python ${basename}/j_o.py -d campus4k --gpu_ids 0,5 --logs-dir logs/kapa/c_3 -k 0.3
sleep 20

python ${basename}/j_o.py -d campus4k --gpu_ids 0,5 --logs-dir logs/kapa/c_5 -k 0.5
sleep 20

python ${basename}/j_o.py -d campus4k --gpu_ids 0,5 --logs-dir logs/kapa/c_7 -k 0.7
sleep 20

#python ${basename}/j_d.py -d campus4k --gpu_ids 6,7 --logs-dir logs/kapa/c_9 -k 0.9
#sleep 20

# python ${basename}/j_a.py -d wpreid --gpu_ids 6,7 --gps_iter 5 --epochs 60  --logs-dir logs/wp_gps_5
# sleep 20





