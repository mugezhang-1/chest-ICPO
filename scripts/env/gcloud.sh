# source scripts/env/gcloud.sh  — Google Cloud CLI on OSC without touching the (full) home directory.
# The CLI lives on scratch (reinstallable); login state lives on project storage.
export CLOUDSDK_CONFIG=/fs/ess/PAS2836/mugezhang/.gcloud
export PATH=/fs/scratch/PAS2836/mugezhang/tools/google-cloud-sdk/bin:$PATH
