#!/bin/bash
# Rebuild every derived table from the raw PhysioNet files, in dependency order, from committed code.
# Optional: REFERENCE=<hash json> to verify the rebuild matches earlier outputs exactly.
#
#   sbatch --export=ALL,REFERENCE=/fs/ess/PAS2836/mugezhang/data/icpo/work/repro/baseline_hashes_2026-10-07.json \
#          scripts/pipeline/run_all.sbatch
#
#SBATCH --job-name=icpo-pipeline
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out

set -euo pipefail
REPO=/fs/ess/PAS2836/mugezhang/code/ChestICPO
PY=$REPO/.venv/bin/python
cd "$REPO"
export HF_HOME=/fs/ess/PAS2836/mugezhang/hf_cache
echo "commit $(git rev-parse HEAD)"

echo "== labelers";  $PY scripts/labelers/run_chexbert.py --batch-size 64
                     $PY scripts/labelers/qba_labels.py
echo "== step 1";    $PY scripts/pipeline/timeline.py
echo "== step 2";    $PY scripts/pipeline/labels.py
                     $PY scripts/pipeline/labels.py --labeler cxnb
echo "== step 3";    $PY scripts/pipeline/progression.py
echo "== step 4";    $PY scripts/pipeline/text_context.py
echo "== step 5";    $PY scripts/pipeline/windows.py
echo "== step 6";    $PY scripts/pipeline/mint_questions.py
echo "== step 8";    $PY scripts/pipeline/splits.py
echo "== hashes";    $PY scripts/data/image_hashes.py --workers ${SLURM_CPUS_PER_TASK:-28}
echo "== step 7";    $PY scripts/pipeline/build_bundles.py
