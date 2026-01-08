# conda create -n Grounded-SAM2 python=3.10

# export PATH=/opt/common/cuda/cuda-12.1.1/bin:$PATH
# export LD_LIBRARY_PATH=/opt/common/cuda/cuda-12.1.1/lib64:$LD_LIBRARY_PATH

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python transformers supervision pycocotools addict yapf timm
pip install -e .

# /usr/bin/g++ --version
# module avail gcc
# module load gcc/11.2.0   # 또는 10/12 등 가능한 최신
# export CC=gcc
# export CXX=g++

pip install --no-build-isolation -e grounding_dino