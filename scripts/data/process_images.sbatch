#!/bin/bash
# Resize -> QA gate -> pack shards for one manifest batch. Stops at the first failing step.
#
#   sbatch --export=ALL,BATCH=1 scripts/data/process_images.sbatch
#
#SBATCH --job-name=icpo-img
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out

set -euo pipefail
BATCH=${BATCH:?set BATCH}
REPO=/fs/ess/PAS2836/mugezhang/code/ChestICPO
PY=$REPO/.venv/bin/python
cd "$REPO"
export OMP_NUM_THREADS=1

echo "== resize"; $PY scripts/data/resize_images.py --batch "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
echo "== QA";     $PY scripts/data/check_resized.py --batch "$BATCH" --sample 300
echo "== pack";   $PY scripts/data/shards.py pack --batch "$BATCH"
du -sh "$($PY -c "from icpo import config; print(config.path('shards_root'))")"
