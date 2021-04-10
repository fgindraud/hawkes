#!/bin/bash

# Compute goodness metric for the hawakes example.
# This will create one file per process file (adding a .lambda_hat suffix).
# These files will contain one line per point in the process file, containing the corresponding lambda_hat value.
# That is, for point x_m in process_m.bed, lambda_hat_m(x_m).
../goodness -f p1.bed -f p2.bed estimated_a -histogram 3 20
