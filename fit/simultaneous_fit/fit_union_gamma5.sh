#!/bin/bash
###
 # @Description: 
 # @version: 1.0
 # @Author: Wei Sun
 # @Date: 2021-06-15 20:15:21
 # @LastEditors: Wei Sun
 # @LastEditTime: 2021-09-08 16:06:51
### 
set -e
set -u

nconf=6084
nt=128

./fit_union_gamma5.py -i ../data/corr{CC,GG,GC}.npy -o output/gamma5phys -dim ${nconf} ${nt} \
    -xl 9.6 -0.4 -0.4 -xh 32.4 17.4 22.4 \
    -yl 0.3 0.0 -0.04 -yh 0.35 0.5 0.005 \
    -rx 0 25 -ry -0.04 0.01 -rmin 10 1 1 -rmax 30 15 20 \
    -choose 13 3 2 -tkxy 5 1 0.2 0.1 -x "\$t/a_t\$"


