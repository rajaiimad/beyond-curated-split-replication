#!/usr/bin/env bash
# Unpack the two UFFCSData archives shipped in data/raw/UFFCSData/ (unmodified upstream files, commit b5128e6)
# into the folder layout the code expects, check every file against data/SHA256SUMS*, and create .env from
# .env.example if it does not exist. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
U=data/raw/UFFCSData
command -v unzip >/dev/null || { echo "unzip is required (e.g. apt install unzip)"; exit 1; }
mkdir -p "$U/RAW_DATABASE" "$U/DATABASE_SELECTED_DATA"
unzip -oq "$U/RAW_DATABASE.zip" -d "$U/RAW_DATABASE"                     # -> RAW_DATABASE/RAW_DATABASE/<74 sessions>
unzip -oq "$U/DATABASE_SELECTED_DATA.zip" -d "$U/DATABASE_SELECTED_DATA" # -> DATABASE_SELECTED_DATA/DATABASE/<35 sessions>
echo "raw sessions:     $(find "$U/RAW_DATABASE/RAW_DATABASE" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "curated sessions: $(find "$U/DATABASE_SELECTED_DATA/DATABASE" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }
bash verification/verify_data.sh
