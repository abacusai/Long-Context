apt-get -y update
apt-get install -y libaio-dev

# export CUDA_HOME=/usr/local/cuda-12.1
# LD_LIBRARY_PATH
#    /usr/local/cuda/compat/lib.real
#    /usr/local/lib/python3.8/dist-packages/torch/lib
#    /usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib
#    /usr/local/cuda/compat/lib
#    /usr/local/cuda-12/lib64

cd /tmp
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
CUDA_HOME=/usr/local/cuda-12.1 CUDA_VERSION=121 make cuda12x
python setup.py install
python -m bitsandbytes  # This should work if setup correctly
pip install -r $HOME/llm-training/config/requirements.txt