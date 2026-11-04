#!/bin/bash
# Label all MIMIC-CXR reports with CheXbert on CPU (the project torch build is CPU-only).
#
#   sbatch scripts/labelers/run_chexbert.sbatch
#
#SBATCH --job-name=icpo-chexbert
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out

set -euo pipefail
REPO=/fs/ess/PAS2836/mugezhang/code/ChestICPO
cd "$REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python scripts/labelers/run_chexbert.py --batch-size 64
