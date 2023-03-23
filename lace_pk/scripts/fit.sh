#!/bin/bash

source /data/desi/scratch/jchavesm/mambaforge/bin/activate lace

echo "args: " $@

python /data/desi/scratch/jchavesm/LaCE_pk/lace_pk/scripts/fit_pflux.py $@

echo "JDONE!"

