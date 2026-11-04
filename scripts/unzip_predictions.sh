#!/usr/bin/env bash
# Unpack the archived out-of-group predictions (downloaded from the Zenodo record) into
# results/preds/ and results_logo/preds/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/preds
for z in results/preds_zip/*.zip; do unzip -oq "$z" -d results/preds; done
# the first, leave-one-group-out run: its results are committed in results_logo/; its predictions are optional
if [ -f results_logo/preds.zip ]; then mkdir -p results_logo/preds && unzip -oq results_logo/preds.zip -d results_logo/preds; fi
echo "predictions: $(ls results/preds | wc -l) files in results/preds"
