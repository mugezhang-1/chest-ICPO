#!/bin/bash
#SBATCH --job-name=icpo-hash
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out
set -euo pipefail
cd /fs/ess/PAS2836/mugezhang/code/ChestICPO
.venv/bin/python scripts/data/image_hashes.py --workers 28
