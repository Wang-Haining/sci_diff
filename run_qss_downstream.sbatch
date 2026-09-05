#!/bin/bash
#SBATCH --job-name=sci-diff-downstream
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_downstream_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_downstream_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python qss_downstream.py
