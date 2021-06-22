#!/bin/bash

# Compute goodness metric for the hawkes example.
#
# This will create one file per process file (adding a .lambda_hat suffix).
# These files will contain one line per point in the process file, containing the corresponding lambda_hat value.
# That is, for point x_m in process_m.bed, lambda_hat_m(x_m).
# 
# In addition it will also create for each process file "f" a f.lambda_hat.tmax file,
# containing lambda_hat values for tmax for each region.
# These tmax default to max{x in union_m N_{m,r}}, and can be overriden with command line option '-tmax'.
# This override applies to ALL regions uniformly, for simplicity ; as the expected use is with one region this should not be a problem.
../goodness -f p1.bed -f p2.bed estimated_a -histogram 3 20

