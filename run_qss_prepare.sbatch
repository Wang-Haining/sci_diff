#!/bin/bash
#SBATCH --job-name=sci-diff-qss-prepare
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_qss_prepare_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_qss_prepare_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
/usr/bin/python3 qss_prepare.py
