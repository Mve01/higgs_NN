#!/bin/bash

# Remove the existing condor log file
rm -f /project/atlas/users/mveldijk/drellyan_maf/run/run_train.condorlog

source ~/.bashrc
source /etc/profile
shopt -s expand_aliases
pyenv activate myenv
cd /project/atlas/users/mveldijk/drellyan_maf 
python train.py


