#!/bin/bash
#SBATCH --job-name=sci-diff-pilot
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
/usr/bin/python3 pilot.py
