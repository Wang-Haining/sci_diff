#!/bin/bash
#SBATCH --job-name=sci-diff-network-sync
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_network_sync_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_network_sync_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
git add artifacts/qss_v3/run_network.json \
  results/qss_v3/network_nodes.csv results/qss_v3/network_edges.csv \
  results/qss_v3/network_metrics.csv results/qss_v3/network_leave_one_domain_out.csv \
  results/qss_v3/figures
git commit -m "Add network validation and NHB figures"
git push origin HEAD
