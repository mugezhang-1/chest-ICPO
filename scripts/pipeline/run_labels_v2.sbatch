#!/bin/bash
# Labels v2 partial rebuild (docs/label_fix_design.md): steps 2 (main + ablations), 6, 8, then the independent
# re-derivation. Steps 1, 3-5 and the labelers do not depend on the label rule and are not rerun.
#SBATCH --job-name=icpo-labels-v2
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out

set -euo pipefail
REPO=/fs/ess/PAS2836/mugezhang/code/ChestICPO
PY=$REPO/.venv/bin/python
cd "$REPO"
echo "commit $(git rev-parse HEAD)"
echo "== step 2";   $PY scripts/pipeline/labels.py
                    $PY scripts/pipeline/labels.py --labeler chexbert_v1
                    $PY scripts/pipeline/labels.py --labeler cxnb
echo "== step 6";   $PY scripts/pipeline/mint_questions.py
echo "== step 8";   $PY scripts/pipeline/splits.py
