#!/bin/bash
#SBATCH --job-name=sci-diff-dirty-sync
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_dirty_sync_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_dirty_sync_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
git add artifacts/qss_v2/qwen3_taxonomy.npz \
  artifacts/qss_v2/run_outcome_embed.json artifacts/qss_v2/run_dirty_analyze.json \
  results/qss_v2/dirty_report.md results/qss_v2/dirty_estimates.csv \
  results/qss_v2/dirty_gates.csv
git commit -m "Add QSS v2 dirty Qwen3 results"
git push origin HEAD
