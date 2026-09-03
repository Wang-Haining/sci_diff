#!/bin/bash
#SBATCH --job-name=sci-diff-qss-setup
#SBATCH --account=group-jasonclark
#SBATCH --partition=fairshare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/g91p721/logs/sci_diff_qss_setup_%j.out
#SBATCH --error=/home/g91p721/logs/sci_diff_qss_setup_%j.err

set -euo pipefail
cd /home/g91p721/sci_diff
export PIP_CACHE_DIR=/home/group/jasonclark/g91p721/sci_diff/pip_cache
python3 -m venv --system-site-packages /home/group/jasonclark/g91p721/sci_diff/qss_venv
source /home/group/jasonclark/g91p721/sci_diff/qss_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-qss.txt
python - <<'PY'
import adapters, lightgbm, torch, transformers
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(),
       "transformers": transformers.__version__, "adapters": adapters.__version__,
       "lightgbm": lightgbm.__version__})
PY
