#!/bin/bash
#SBATCH --job-name=sci-diff-qss-embed
#SBATCH --account=group-jasonclark
#SBATCH --partition=gpupriority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_qss_embed_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_qss_embed_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
torchrun --standalone --nproc_per_node=4 qss_embed.py
