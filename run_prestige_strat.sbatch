#!/bin/bash
#SBATCH --job-name=sci-diff-prestige
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_prestige_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_prestige_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python prestige_strat.py
