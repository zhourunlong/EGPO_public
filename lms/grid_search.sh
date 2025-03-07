#!/bin/bash

# for y_yp_mixture_coef in 0.5 0.75 1.0; do
for y_yp_mixture_coef in 0; do
    for y_yp_temperature in 2.0; do
        for y_yp_top_k in 5 10; do
            deepspeed --master_port=23456 --include="localhost:4,5,6,7" online_ipo_1.py  --lr=0.000001 --y_yp_mixture_coef=$y_yp_mixture_coef --y_yp_temperature=$y_yp_temperature --y_yp_top_k=$y_yp_top_k --y_yp_min_p=0.0
        done
    done
done
