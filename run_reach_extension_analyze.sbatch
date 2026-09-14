#!/bin/bash
#SBATCH --job-name=reach-extension-analyze
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/g91p721/logs/reach_extension_analyze_%j.out
#SBATCH --error=/home/g91p721/logs/reach_extension_analyze_%j.err
set -euo pipefail
cd /home/g91p721/sci_diff
export PYTHONUNBUFFERED=1
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
case "${1:?expected 60 or tenyear}" in
  60) python reach_extension_analyze.py 60 ;;
  tenyear)
    python reach_extension_tenyear.py outcomes
    python reach_extension_analyze.py tenyear ;;
  *) exit 2 ;;
esac
