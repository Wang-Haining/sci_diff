#!/bin/bash
#SBATCH --job-name=sci-diff-downstream-sync
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_downstream_sync_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_downstream_sync_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
git add artifacts/qss_v3/run_downstream.json \
  results/qss_v3/subgroup_estimates.csv results/qss_v3/subgroup_tests.csv \
  results/qss_v3/downstream_balance.csv results/qss_v3/downstream_outcome_diagnostics.csv \
  results/qss_v3/downstream_propensity.csv results/qss_v3/macro_labels.csv \
  results/qss_v3/figures
git commit -m "Add QSS downstream results"
git push origin HEAD
