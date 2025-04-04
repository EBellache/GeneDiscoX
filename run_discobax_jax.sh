#!/usr/bin/env bash
# Usage:  ./run_discobax_jax.sh
#
# Bash script to launch GeneDisco with:
#  - A custom JAX-based model (e.g. jax_sparse_gp)
#  - The two-step DiscoBAX acquisition function in JAX
#  - GPU acceleration enabled via environment variables
#
# Make sure to "chmod +x run_discobax_jax.sh" before running.

set -e  # Exit on error

########################################
# 1) Adjust environment & pipeline config
########################################

# Name of your conda environment containing GeneDisco + JAX (CUDA-enabled).
ENV_NAME="my_genedisco_env"

# Name of the dataset/feature set to run on. E.g. "single_cell_cycle" + "achilles" features
DATASET_NAME="single_cell_cycle"
FEATURE_SET_NAME="achilles"

# Your custom JAX model name (defined in the GeneDisco model registry or factory code).
# e.g., "jax_sparse_gp"
MODEL_NAME="jax_sparse_gp"

# Paths for your custom acquisition function
ACQ_FUNC_NAME="custom"
ACQ_FUNC_PATH="genedisco/active_learning_methods/acquisition_functions/disco_bax_two_stage_jax.py"

# Active learning parameters
ACQ_BATCH_SIZE=64
NUM_ACTIVE_LEARNING_CYCLES=8

########################################
# 2) Activate conda environment
########################################

# Adjust path to conda.sh if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

########################################
# 3) Set environment variables for JAX + GPU
########################################

# Make sure JAX sees GPU device 0 (the A6000).
export CUDA_VISIBLE_DEVICES=0

# Force JAX to use the GPU backend (if you have jax>=0.4.14, set JAX_PLATFORMS="gpu" instead).
export JAX_PLATFORM_NAME="gpu"

# Prevent JAX from preallocating nearly all GPU memory; let it grow as needed.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Optionally limit JAX to a fraction (e.g. 30%) of total GPU memory, to avoid conflicts with PyTorch, etc.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

########################################
# (Optional) Ensure JAX w/ CUDA is installed
# (Uncomment if you need to install automatically here)
########################################
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

########################################
# 4) Sanity check: does JAX see the GPU?
########################################

echo "==== JAX Devices ===="
python -c 'import jax; print(jax.devices())'

########################################
# 5) Run GeneDisco pipeline
########################################

python run_pipeline.py \
    --dataset_name="${DATASET_NAME}" \
    --feature_set_name="${FEATURE_SET_NAME}" \
    --model_name="${MODEL_NAME}" \
    --acquisition_function_name="${ACQ_FUNC_NAME}" \
    --acquisition_function_path="${ACQ_FUNC_PATH}" \
    --acquisition_batch_size="${ACQ_BATCH_SIZE}" \
    --num_active_learning_cycles="${NUM_ACTIVE_LEARNING_CYCLES}" \
    --debug  # (Optional) If you want more verbose logs

echo "==== Run Completed ===="
