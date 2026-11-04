#!/bin/bash
# Download one manifest batch from PhysioNet's GCS bucket (requester pays, billed to icpo-510602),
# then re-verify everything on disk (also catches partial files left by the earlier wget job).
#
#   sbatch --export=ALL,BATCH=1,MAX_USD=4.5 scripts/data/gcs_download.sbatch
#
#SBATCH --job-name=icpo-gcs
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out

set -euo pipefail
BATCH=${BATCH:?set BATCH}
MAX_USD=${MAX_USD:?set MAX_USD (spending cap for this batch)}
WORKERS=${WORKERS:-32}

REPO=/fs/ess/PAS2836/mugezhang/code/ChestICPO
PY=$REPO/.venv/bin/python
cd "$REPO"

$PY scripts/data/gcs_download.py get --batch "$BATCH" --max-usd "$MAX_USD" --workers "$WORKERS" --report-every 2000
$PY scripts/data/verify_originals.py --batch "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
du -sh "$($PY -c "from icpo import config; print(config.path('originals_root'))")"
