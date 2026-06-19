export PATH=/home/damian/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate trellis_blackwell

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
export ENABLE_TRELLIS_CPU_OFFLOAD=false

# Image Model
export IMAGE_MODEL=qwen # flux, gemini, qwen
# FLUX MODEL 
# cpu offload if we have less than 24GB of VRAM 
export USE_FLUX_DEV=false
export ENABLE_IMAGE_CPU_OFFLOAD=false
export FAST_IMAGE_SAMPLING=true
export QWEN_LOAD_IN_8BIT=false  # Default, reduces VRAM

# Printability (Stage-1 sparse structure)
# Fill enclosed voids in the Stage-1 occupancy grid before Stage-2.
# ~15% less 3D-print support material, detail-safe. Free (no GPU/training cost).
export TRELLIS_STAGE1_FILL_HOLES=true

# Optional: load a printability DPO LoRA onto the Stage-1 flow model.
# Stacks with fill_holes for the full effect (see printability_optimization_3d.md).
# Validated at sparse_structure_cfg ~ 5.0. Leave empty to disable.
# export TRELLIS_STAGE1_LORA=""
# To enable, point at the vendored r=16 checkpoint:
export TRELLIS_STAGE1_LORA=checkpoints/printability_lora_r16

# Printable STL export: voxel-remesh the mesh into a single watertight, manifold solid
# on STL export (TRELLIS meshes are fragmented non-watertight shells). Validated in
# investigations/mesh_repair: 100% watertight, DINOv2 quality impact ~0.033 (noise floor).
# Costs ~9s on export and softens the very finest detail; set to 'off' to disable.
export TRELLIS_PRINT_REMESH=voxel288_smooth
# Decimate the print-ready mesh to this target face count (the raw remesh is ~340k tiny
# faces, which strains viewers/slicers). Validated in investigations/mesh_repair: ~25k
# keeps quality at the perceptual noise floor while being 14x lighter. 0 disables.
export TRELLIS_PRINT_TARGET_FACES=25000


streamlit run streamlit-app.py --server.fileWatcherType none

# [1] 2690470