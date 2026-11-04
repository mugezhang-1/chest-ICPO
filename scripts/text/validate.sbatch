#!/bin/bash
#SBATCH --job-name=icpo-textval
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out
set -euo pipefail
cd /fs/ess/PAS2836/mugezhang/code/ChestICPO
export HF_HOME=/fs/ess/PAS2836/mugezhang/hf_cache
.venv/bin/python scripts/text/validate.py --model qwen3.5-9b-nothink --round "${ROUND:-}"
