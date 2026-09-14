#!/bin/bash
#SBATCH --job-name=reach-extension-sync
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/g91p721/logs/reach_extension_sync_%j.out
#SBATCH --error=/home/g91p721/logs/reach_extension_sync_%j.err
set -euo pipefail
cd /home/g91p721/sci_diff
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python -c 'import json; from pathlib import Path; p=Path("artifacts/reach_extension_v1"); runs=[json.loads((p/f"run_analyze_{m}.json").read_text()) for m in ("60","tenyear")]; assert all(r["status"]=="complete" for r in runs),runs; print("Both extension analyses complete")'
git add results/reach_extension_v1 artifacts/reach_extension_v1
git diff --cached --check
git commit -m "Add audience breadth and follow-up extension results"
git push origin HEAD
