#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp15_L2_multi

# Experiment 15: Fully Paired - L2 Intermediate Multi-Step
BASE_DIR="/gpfs/scratch/rl5285/test/unsbmri"
cd "$BASE_DIR" || exit 1

export DATAROOT="/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"
export PYTHON_BIN="/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"
export EXPERIMENT_NAME="ablation_exp15_fully_pair_L2_inter_multi"
export N_EPOCHS=400
export N_EPOCHS_DECAY=200
export BATCH_SIZE=1
export PAIRED_STAGE="--paired_stage"
export PAIRED_SUBSET_RATIO=1.0
export PAIRED_SUBSET_SEED=42
export COMPUTE_METRICS="--compute_paired_metrics"
export USE_L2_INTER_MULTI="--use_l2_intermediate_multi"
export DISABLE_GAN="--disable_gan"
export DISABLE_NCE="--disable_nce"
export CONTINUE_TRAIN=""
export PRETRAINED_NAME=""
export LOAD_EPOCH=""
export EPOCH_COUNT=1

bash run_train.sh
