#!/bin/bash

export SCRIPT=/project/atlas/users/mveldijk/higgs_NN_git/run/condor/mveldijk/condorsub/run_train_3512553.sh
export SCRIPTsub=/project/atlas/users/mveldijk/higgs_NN_git/run/condor/mveldijk/condorsub/run_train_3512553.sub
export enviromentvariablescript=/project/atlas/users/mveldijk/higgs_NN_git/run/condor/mveldijk/condorsub/enviromentvariables_3512553.sh
export PATH=/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/scripts:/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/bin:/cvmfs/sft.cern.ch/lcg/releases/gcc/13.1.0-b3d18/x86_64-el9/bin:/cvmfs/sft.cern.ch/lcg/releases/binutils/2.40-acaab/x86_64-el9/bin:/user/mveldijk/.vscode-server/cli/servers/Stable-f220831ea2d946c0dcb0f3eaa480eb435a2c1260/server/bin/remote-cli:/user/mveldijk/bin:/user/mveldijk/.local/bin:/usr/share/Modules/bin:/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/user/mveldijk/bin/:/user/mveldijk/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/user/mveldijk/bin/
cd /project/atlas/users/mveldijk/higgs_NN_git/run
source /project/atlas/users/mveldijk/higgs_NN_git/run/condor/mveldijk/condorsub/enviromentvariables_3512553.sh

cd /project/atlas/users/mveldijk/higgs_NN_git && export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh && . /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh && python train.py

