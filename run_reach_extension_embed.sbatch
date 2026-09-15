#!/bin/bash
#SBATCH --job-name=reach-extension-embed
#SBATCH --account=group-jasonclark
#SBATCH --partition=gpufairshare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=12:00:00
#SBATCH --output=/home/g91p721/logs/reach_extension_embed_%j.out
#SBATCH --error=/home/g91p721/logs/reach_extension_embed_%j.err
set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
export TORCH_DISABLE_NATIVE_JIT=1
export HF_HOME=/home/group/jasonclark/g91p721/sci_diff/huggingface
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
torchrun --standalone --nproc_per_node=4 reach_extension_embed.py
