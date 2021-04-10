#!/bin/bash

# print commands
set -x

# Run inferrence on (p1,p2) with histograms(k=3, delta=20)
# verbose adds a description in the output : inferrence arguments, dimensions of a
../hawkes -f p1.bed -f p2.bed -histogram 3 20 -verbose > estimated_a
cat estimated_a

# With kernels
#../hawkes -f p1.bed -f p2.bed -histogram 5 20 -kernel homogeneous_interval > estimated_a

# Haar wavelets base instead of histograms
# ../hawkes -f p1.bed -f p2.bed -haar 2 30 > estimated_a

# Help with list of options
# There are some debug options :
# - print intermediate values
# - print stats for bed files, to check proper parsing of data
# - list of supported modes (kernel x base)
# ../hawkes -h
