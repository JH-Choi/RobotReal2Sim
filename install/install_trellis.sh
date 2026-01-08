# conda create -n trellis python=3.10

# export PATH=/opt/common/cuda/cuda-11.8.0/bin:$PATH
# export LD_LIBRARY_PATH=/opt/common/cuda/cuda-11.8.0/lib64:$LD_LIBRARY_PATH

# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 
pip install --no-build-isolation flash-attn==2.5.8 -v
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html --no-build-isolation -v

mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast --no-build-isolation -v

mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation -v

mkdir -p /tmp/extensions
cp -r extensions/vox2seq /tmp/extensions/vox2seq
pip install /tmp/extensions/vox2seq --no-build-isolation -v

 pip install spconv-cu118 --no-build-isolation -v
 pip install gradio==4.44.1 gradio_litmodel3d==0.0.1 --no-build-isolation -v