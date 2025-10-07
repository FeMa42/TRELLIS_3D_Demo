export PATH=/mnt/damian/miniconda3/bin:$PATH 
eval "$(conda shell.bash hook)"
conda activate trellis_new

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Multi-GPU is not compatible with TRELLIS CPU offloading
# Use optimized single GPU mode with CPU offloading for low vram (>10GB)
# With Multi-GPU cpu offloading for TRELLIS is disabled. Needs more than 16GB of VRAM
export USE_MULTI_GPU=false

# To enable quality ranking:
export ENABLE_REWARD_MODEL=false

# TRELLIS 
export TRELLIS_MODEL_ID="JeffreyXiang/TRELLIS-image-large"
# cpu offload if we have less than 24GB of VRAM
export ENABLE_TRELLIS_CPU_OFFLOAD=true
# Gaussian Rendering has issues on the A30 Servers and tries to allocate 66TB of VRAM
# Deactivate Gaussian Rendering on A30 Servers, on L40s you can activate it
# It also has issues with multi GPU setup right now. So if you use multiple gpus deactivate it 
export USE_GAUSSIAN_RENDERING=true

# Image Model 
export IMAGE_MODEL=flux # flux or gemini
# FLUX MODEL 
# cpu offload if we have less than 24GB of VRAM 
export ENABLE_FLUX_CPU_OFFLOAD=true 
# Compilation 
# Needs more time at start but faster inference 
export COMPILE_FLUX_OPTIMIZATION=false 
export USE_FLUX_DEV=true

streamlit run streamlit-app.py --server.fileWatcherType none

