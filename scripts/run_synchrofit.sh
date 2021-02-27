#! /usr/bin/bash

synchrofit \
    --work_dir "/home/sputnik/Documents/software/synchrofit/exampes/"
    --data "test_spectra.dat" \
    --fit_type "CI" \
    --remnant_range 0 \
    --mc_length 50 \
    --plot \
    --write_model \
    --age \
    --bfield 0.174 \
    --z 0.2133