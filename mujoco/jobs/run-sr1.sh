# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export HOME=/storage/home/rahulr

export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/nvidia-384/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="/scratch/scratch1/rahul/mujoco/lib:$LD_LIBRARY_PATH"


export PATH="/usr/local/nvidia/bin:$PATH"
# export PATH="/scratch/scratch1/rahul/mujoco/bin:$PATH"

export PATH=/tools/anaconda3/envs/py27/bin:$PATH
export PATH=/tools/anaconda3/envs/py35/bin:$PATH
export PATH=/tools/anaconda3/bin:$PATH

export PATH=/tools/local/bin:/tools/cuda/bin:$PATH
export LD_LIBRARY_PATH=/tools/local/lib:/tools/cuda/lib64:/tools/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/tools/cuda

# export C_INCLUDE_PATH="/scratch/scratch1/rahul/mujoco/include:$C_INCLUDE_PATH"
# export C_LIBRARY_PATH="/usr/local/nvidia/lib:$C_LIBRARY_PATH"

export DISABLE_MUJOCO_RENDERING=1

export HOME=/storage/home/rahulr
export PATH="/usr/local/nvidia/bin:/tools/local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/nvidia-384:/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export C_LIBRARY_PATH="/usr/local/nvidia/lib:$C_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin"
export C_INCLUDE_PATH="$basepath/rlenv/include:$C_INCLUDE_PATH"

# export CXXFLAGS="$CXXFLAGS -fPIC"

 #Tools config for CUDA, Anaconda installed in the common /tools directory
cd /storage/home/rahulr/RL/fetch/
source activate /scratch/scratch1/rahul/mujoco

# python3 -u qlearning/main.py --use_gpu True --checkpointFile ckptSuc1 --gamma 0.99 &> logs/successor-fetch1.log
python3 test.py
