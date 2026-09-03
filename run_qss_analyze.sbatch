#!/bin/bash
#SBATCH --job-name=sci-diff-qss-analyze
#SBATCH --account=group-jasonclark
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=750G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_qss_analyze_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_qss_analyze_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python qss_analyze.py
