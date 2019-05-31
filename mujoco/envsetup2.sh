#!/usr/bin/env bash

export HOME=/storage/home/rahulr

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin:/sbin
export PATH=/usr/local/nvidia/bin:/tools/local/bin$PATH
export LD_LIBRARY_PATH=/usr/lib/nvidia-384:/usr/local/nvidia/lib$LD_LIBRARY_PATH
export C_LIBRARY_PATH=/usr/local/nvidia/lib:$C_LIBRARY_PATH
export C_INCLUDE_PATH=/scratch/scratch1/rahul/mujoco-gym/include:$C_INCLUDE_PATH

source /tools/config.sh
export MUJOCO_PY_MJPRO_PATH=/tools/opt/mujoco/.mujoco/mjpro150
export MUJOCO_PY_MJKEY_PATH=/tools/opt/mujoco/.mujoco/mjkey.txt

export LD_LIBRARY_PATH=/tools/opt/mujoco/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
export DISABLE_MUJOCO_RENDERING=1

cd /storage/home/rahulr/RL/fetch
source activate /scratch/scratch1/rahul/mujoco-gym
python3 test.py  &> test.log
# python3 -u qlearning/main.py --use_gpu True --checkpointFile ckptSuc1 --gamma 0.99 &> logs/successor-fetch1.log


