# --- Apt get libcupti
# apt-get update && apt-get install -y cuda-command-line-tools-12-2

# --- List sorted space usage
# du -h | sort -h

# --- Copy libdevice from tensor rt to readable dir
# cp /opt/conda/envs/mimic3/lib/python3.9/site-packages/triton/third_party/cuda/lib/libdevice.10.bc /usr/local/cuda/nvvm/libdevice/

# --- Find a file
# find / -name "libcupti.so.12" 2>/dev/null

# --- TF version from metadata
# /opt/conda/envs/mimic3/lib/python3.9/site-packages/tensorflow-2.14.0.dist-info/METADATA

# --- Export CUDA paths to point torch a existing torch
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

# --- Delete preinstalled cuda
# rm -rf /opt/conda/envs/mimic3/lib/python3.9/site-packages/nvidia

# --- Delete container cuda
# rm -rf /usr/local/cuda-11.8

# --- Simlink to libcupti version agnostic
# ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcupti.so /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcupti.so.11.7

# --- Quick cuda test
# import torch
# torch.cuda.is_available()

# --- Quick tensorflow test
# import tensorflow as tf
# tf.config.list_physical_devices()
