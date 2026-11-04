#!/bin/bash
# Download one manifest batch of MIMIC-CXR-JPG originals to scratch, then verify checksums.
# Credentials come from ~/.netrc (machine physionet.org). Safe to re-submit: only missing files are fetched.
#
#   sbatch --export=ALL,BATCH=1 scripts/data/download_images.sbatch
#
#SBATCH --job-name=icpo-dl
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out

set -euo pipefail
BATCH=${BATCH:?set BATCH=1 or BATCH=2}
MANIFEST=${MANIFEST:-images_v1}
PARALLEL=${PARALLEL:-8}

REPO=/fs/ess/PAS2836/mugezhang/code/ChestICPO
PY=$REPO/.venv/bin/python
cd "$REPO"

ROOT=$($PY -c "from icpo import config; print(config.path('originals_root'))")
WORK=$($PY -c "from icpo import config; print(config.path('work_root'))")
mkdir -p "$ROOT"
LIST=$(mktemp -d)

# URLs for manifest entries not already on disk
$PY - "$WORK/manifests/$MANIFEST.csv" "$BATCH" "$ROOT" > "$LIST/urls.txt" <<'EOF'
import sys, os, pandas as pd
from icpo.data.images import JPG_BASE_URL
man, batch, root = pd.read_csv(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for rel in man.loc[man.batch == batch, "relpath"]:
    if not os.path.exists(os.path.join(root, rel)):
        print(JPG_BASE_URL + rel)
EOF
echo "to download: $(wc -l < "$LIST/urls.txt") files (batch $BATCH)"

split -n l/"$PARALLEL" -d "$LIST/urls.txt" "$LIST/part."
for part in "$LIST"/part.*; do
    # -x -nH --cut-dirs=3 maps .../mimic-cxr-jpg/2.1.0/files/pXX/... to $ROOT/files/pXX/...
    wget -nv -c -x -nH --cut-dirs=3 -P "$ROOT" --tries=10 --waitretry=30 --timeout=60 -i "$part" \
        > "$part.log" 2>&1 &
done
wait || true
grep -h -iE "error|failed" "$LIST"/part.*.log | head -20 || true

$PY scripts/data/verify_originals.py --manifest "$MANIFEST" --batch "$BATCH" --workers "$SLURM_CPUS_PER_TASK"
du -sh "$ROOT"
rm -rf "$LIST"
