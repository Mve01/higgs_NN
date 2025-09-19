#!/bin/bash
condorsub -J run_train -q short -n 1 -m 16000 \
"cd /project/atlas/users/mveldijk/drellyan_maf && \
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && \
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh && \
. /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh && \
python train.py"
