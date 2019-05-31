#!/usr/bin/env bash

# change below
source /tools/config.sh
source activate "/scratch/scratch1/sidnayak/gym-mujoco"

# Change below
export HOME=/storage/home/sidnayak
export PATH="/usr/local/nvidia/bin:/tools/local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/nvidia-384:/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export C_LIBRARY_PATH="/usr/local/nvidia/lib:$C_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/storage/home/sidnayak/.mujoco/mujoco200/bin"

# virtualenv path
# change below (to your conda env path)
export C_INCLUDE_PATH="/scratch/scratch1/sidnayak/gym-mujoco/include:$C_INCLUDE_PATH"
export DISABLE_MUJOCO_RENDERING=1

cd /storage/home/sidnayak/mujoco
# python3 test.py 
python3 test.py  >& out
# python3 -u qlearning/main.py --use_gpu True --checkpointFile ckptSuc1 --gamma 0.99 &> logs/successor-fetch1.log


