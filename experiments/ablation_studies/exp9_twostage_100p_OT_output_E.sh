#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp9_2s100_out_E

# Experiment 9: Two-Stage 100% - OT Output + Entropy
BASE_DIR="/gpfs/scratch/rl5285/test/unsbmri"
cd "$BASE_DIR" || exit 1

export DATAROOT="/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"
export PYTHON_BIN="/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"
export EXPERIMENT_NAME="ablation_exp9_twostage_100p_OT_output_E"
export N_EPOCHS=500
export N_EPOCHS_DECAY=100
export BATCH_SIZE=1
export PAIRED_STAGE="--paired_stage"
export PAIRED_SUBSET_RATIO=1.0
export PAIRED_SUBSET_SEED=42
export COMPUTE_METRICS="--compute_paired_metrics"
export USE_OT_OUTPUT="--use_ot_output"
export USE_ENTROPY="--use_entropy_loss"
export DISABLE_GAN="--disable_gan"
export DISABLE_NCE="--disable_nce"
export CONTINUE_TRAIN="--continue_train"
export PRETRAINED_NAME="unpaired"
export LOAD_EPOCH="latest"
export EPOCH_COUNT=401

bash run_train.sh
