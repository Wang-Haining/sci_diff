#!/bin/bash
#SBATCH --job-name=sci-diff-qss-finalize
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_qss_finalize_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_qss_finalize_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python qss_summary.py
python qss_figures.py
git add artifacts/qss_v1/run_*.json results/qss_v1
git commit -m "Add QSS specialization results"
git push origin main
