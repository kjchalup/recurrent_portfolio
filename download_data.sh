#!/bin/bash
# Script to download hyperparamter test data from the ec2 instances.

# List of ec2 instances.
ecs="
ubuntu@ec2-52-37-223-20.us-west-2.compute.amazonaws.com
ubuntu@ec2-52-26-75-82.us-west-2.compute.amazonaws.com
ubuntu@ec2-35-165-35-72.us-west-2.compute.amazonaws.com
ubuntu@ec2-52-40-226-151.us-west-2.compute.amazonaws.com
ubuntu@ec2-52-40-234-245.us-west-2.compute.amazonaws.com
ubuntu@ec2-52-40-133-49.us-west-2.compute.amazonaws.com
ubuntu@ec2-52-36-63-29.us-west-2.compute.amazonaws.com
ubuntu@ec2-52-38-176-13.us-west-2.compute.amazonaws.com"

for ec in $ecs
do
    ssh -i "causeai.pem" $ec 'cd data/causehf/saved_data; cp hyper_1000_results.pkl hyper_download.pkl'
    scp -i "causeai.pem" "$ec:/home/ubuntu/data/causehf/saved_data/hyper_download.pkl" "$ec.pkl"
    ssh -i "causeai.pem" $ec 'cd data/causehf/saved_data; rm hyper_download.pkl'
done

# The noncausal instance
# ubuntu@ec2-52-36-251-66.us-west-2.compute.amazonaws.com
# The causal instance
# ubuntu@ec2-35-164-100-106.us-west-2.compute.amazonaws.com
