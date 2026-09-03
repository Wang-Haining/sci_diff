#!/bin/bash
#SBATCH --job-name=sci-diff-qss-exposure
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_qss_exposure_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_qss_exposure_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python qss_exposure.py
