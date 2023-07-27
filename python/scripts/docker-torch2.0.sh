#!/bin/bash

LD_LIBRARY_PATH='/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.8/dist-packages/torch/lib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/cuda-12/lib64'

sudo docker run --shm-size=10g \
  --mount type=bind,source=/abacus,target=/abacus \
  --mount type=bind,source=/data,target=/data \
  --mount type=bind,source=$HOME,target=/root \
  --env CUDA_HOME=/usr/local/cuda-12.1 \
  --env LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
  --net=host --gpus all -it --rm ${1:-nvcr.io/nvidia/pytorch:23.03-py3}
