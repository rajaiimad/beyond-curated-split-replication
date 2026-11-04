#!/usr/bin/env bash
# Check the data files against their SHA-256 checksums.
#   data/SHA256SUMS           the shipped UFFCSData files (unmodified, upstream commit b5128e6) and two derived tables
#   data/SHA256SUMS.decompressed  the content of data/features_w30.csv.gz
#   data/SHA256SUMS.unpacked  the 179 session files unpacked by scripts/prepare_data.sh (checked if present)
#   data/SHA256SUMS.yalcin    the 40 files of the second archive (checked if scripts/fetch_second_archive.sh was run)
set -euo pipefail
cd "$(dirname "$0")/.."
sha256sum --quiet -c data/SHA256SUMS && echo "data: $(wc -l < data/SHA256SUMS) shipped files OK"
# the feature table is compared by its decompressed content: a gzip header records the time of writing, so a
# re-run of src/fa_data.py gives the same table in a file with different bytes
want=$(cut -d' ' -f1 data/SHA256SUMS.decompressed); got=$(gunzip -c data/features_w30.csv.gz | sha256sum | cut -d' ' -f1)
[ "$want" = "$got" ] && echo "data: features_w30.csv.gz content OK" || { echo "data: features_w30.csv.gz content DIFFERS"; exit 1; }
if [ -d data/raw/UFFCSData/RAW_DATABASE/RAW_DATABASE ]; then
  sha256sum --quiet -c data/SHA256SUMS.unpacked && echo "data: $(wc -l < data/SHA256SUMS.unpacked) unpacked session files OK"
else
  echo "data: archives not unpacked yet (run scripts/prepare_data.sh)"
fi
if [ -d data/raw/yalcin ]; then
  sha256sum --quiet -c data/SHA256SUMS.yalcin && echo "data: $(wc -l < data/SHA256SUMS.yalcin) second-archive files OK"
else
  echo "data: second archive not fetched (optional; scripts/fetch_second_archive.sh)"
fi
