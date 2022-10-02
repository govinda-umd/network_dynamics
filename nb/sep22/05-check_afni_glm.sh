#!/bin/csh

set sync = $1

set out = '.'

set play_basis = "CSPLIN(-10, 5, 13)"

if ("$sync" == 'no') then
    set stim_time = "1D: 19.13429" # not synchronized with the TR (1.25 seconds)
else if ("$sync" == 'yes') then 
    set stim_time = "1D: 20" # synchronized with the TR (1.25 seconds)
endif

3dDeconvolve -overwrite \
    -nodata 100 1.25 \
    -noFDR \
	-polort A \
	-local_times \
    -num_stimts 1 \
    -stim_times 1 "$stim_time" "$play_basis"    -stim_label 1 Trial \
    -x1D "$out"/Simul-ABA_full.x1D \
    -x1D_stop