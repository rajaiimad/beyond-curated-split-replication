#!/usr/bin/env bash
# Fetch the second public archive (Yalcin et al., 2024, Front. Virtual Real. 5:1364207; MIT licence) at the
# exact commit used in the paper, into data/raw/yalcin, and check the 40 files the analysis reads
# (eye-tracking and button-press files) against data/SHA256SUMS.yalcin. Needed only to re-run
# src/fa_yalcin.py; its results are already in results/rev3_yalcin.json.
# The archive is not redistributed here: it holds third-party physiological recordings of human participants.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/m1237/automatic-cybersickness-detection
COMMIT=14a242a451c15a7954f4bdd6fe085543676db437
D=data/raw/yalcin
if [ ! -d "$D/.git" ]; then git clone --quiet "$URL" "$D"; fi
git -C "$D" fetch --quiet origin
git -C "$D" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "second archive at commit $(git -C "$D" rev-parse --short HEAD)"
sha256sum --quiet -c data/SHA256SUMS.yalcin && echo "second archive: the 40 files used match data/SHA256SUMS.yalcin"
