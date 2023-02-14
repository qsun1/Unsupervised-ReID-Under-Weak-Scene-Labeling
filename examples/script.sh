#!/bin/bash

basename=$(dirname $(readlink -f $0))

chmod -R 700 ${basename}/..


chmod -R 700 ${basename}
python ${basename}/j_c.py --eps 0.6 --k1 20 --k2 6 --logs-dir logs/a
sleep 20
