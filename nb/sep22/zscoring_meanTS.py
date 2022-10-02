#!/usr/bin/env python

import sys
from scipy.stats import zscore
import numpy as np

class ARGS(): pass
args = ARGS()

print('inside python')

subj = sys.argv[1]
print(subj)

ts = np.loadtxt(
    f"/home/govindas/network_dynamics/data/max/neutral_runs_trial_level_FNSandFNT/mashid/{subj}/{subj}_meanTS_old.1D"
)

ts.shape

args.RUN_LEN = 336

run_tss = np.split(
    ts,
    ts.shape[0] // args.RUN_LEN
)
run_tss = [zscore(run_ts, axis=0) for run_ts in run_tss]
ts = np.concatenate(run_tss, axis=0)
ts.shape
np.savetxt(
    f"/home/govindas/network_dynamics/data/max/neutral_runs_trial_level_FNSandFNT/mashid/{subj}/{subj}_meanTS.1D", 
    ts
)

print('leaving python!!')