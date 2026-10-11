#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --job-name=ux_sim
#SBATCH --output=/home/x_yushe/cb_sim/ux_table/ux_%j.out
#SBATCH --error=/home/x_yushe/cb_sim/ux_table/ux_%j.err
set -euo pipefail
cd /home/x_yushe/cb_sim/ux_table
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/proj/test_extender/ICLR/EditBoundaryDetection/gpu_env312/bin/python embed_ux.py
