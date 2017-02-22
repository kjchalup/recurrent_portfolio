#!/bin/bash

# Give the script an offset indicating which run it should start on,
# such that the different ec2 instances have different seeds.
offset=$1

for i in $(seq 1 4)
do
    echo "*** This is run $i ****"
    python -m linear_hf.supply_hyper_noncausal $(($i+$offset))
    python -m linear_hf.supply_hyper_causal $(($i+$offset))
done
