#!/bin/bash
#SBATCH --job-name=sci-diff-qss-model-smoke
#SBATCH --account=group-jasonclark
#SBATCH --partition=gpufairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_qss_model_smoke_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_qss_model_smoke_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export HF_HOME=/home/group/jasonclark/g91p721/sci_diff/huggingface
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python qss_model_smoke.py
