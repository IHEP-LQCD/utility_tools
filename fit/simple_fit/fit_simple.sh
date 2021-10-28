#!/bin/bash
###
 # @Description: 
 # @version: 1.0
 # @Author: Wei Sun
 # @Date: 2021-06-15 20:15:21
 # @LastEditors: Wei Sun
 # @LastEditTime: 2021-10-28 16:46:30
### 
set -e
set -u

nconf=6084
nt=128

./fit_simple.py -i ../data/corrGG.npy -o output/GG -dim ${nconf} ${nt} \
    -xl -0.4 -xh 17.4 \
    -yl  0.0 -yh 0.5 \
    -rx 0 25 -ry -0.04 0.01 -rmin 1 -rmax 15 \
    -choose 2 -tkxy 5 1 0.2 0.1 -x "\$t/a_t\$"


