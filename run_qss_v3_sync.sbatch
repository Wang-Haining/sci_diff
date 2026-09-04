#!/bin/bash
#SBATCH --job-name=sci-diff-v3-sync
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_v3_sync_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_v3_sync_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
git add artifacts/qss_v3/run_prepare.json artifacts/qss_v3/run_embed.json \
  artifacts/qss_v3/run_analyze.json results/qss_v3
git commit -m "Add QSS v3 dirty pilot results"
git push origin HEAD
