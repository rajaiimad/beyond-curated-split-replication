# Data

## What is here

| File | What it is | Provenance | Licence |
|---|---|---|---|
| `raw/UFFCSData/RAW_DATABASE.zip` | The raw archive: 74 sessions, each a folder with `FILE.xml` (one record per logging tick) and, for 70 sessions, `VRSQ_FILE.xml` (pre- and post-exposure questionnaire) | UFFCSData, https://github.com/tmp1986/UFFCSData, master branch, commit `b5128e681786b81501d269a73e860d57c495d0aa` (1 May 2020); byte-identical to the file at that commit | GNU AGPL-3.0 (`raw/UFFCSData/LICENSE`) |
| `raw/UFFCSData/DATABASE_SELECTED_DATA.zip` | The curated release: 35 seated sessions | same commit, byte-identical | AGPL-3.0 |
| `raw/UFFCSData/dataset.csv` | The curated release as one table; used by the VR-LENS re-implementation (`src/fa_vrlens.py`). It has Windows (CRLF) line endings, kept as published. | same commit, byte-identical | AGPL-3.0 |
| `raw/UFFCSData/LICENSE`, `raw/UFFCSData/README.md` | The archive's licence and its own README | same commit, byte-identical | — |
| `sessions.csv` | One row per session: audit fields, VRSQ scores, inferred participant group, quality flags and `sha16` (the first 16 hex digits of the SHA-256 of each raw `FILE.xml`) | written by `src/fa_data.py` | AGPL-3.0 (derived from UFFCSData) |
| `features_w30.csv.gz` | One row per raw record (18,384 rows): labels, context, demographic answers and the past-only features | written by `src/fa_data.py` | AGPL-3.0 (derived) |
| `../results/session_level.csv` | One row per session: `sessions.csv` plus in-game rating summaries | written by `src/fa_run.py` (stage `vrsq`) | AGPL-3.0 (derived) |
| `SHA256SUMS` | Checksums of the files above, except `features_w30.csv.gz` | — | — |
| `SHA256SUMS.decompressed` | Checksum of the *content* of `features_w30.csv.gz`. A gzip file records the time it was written, so a re-run of `src/fa_data.py` gives the same table in a file with different bytes. | — | — |
| `SHA256SUMS.unpacked` | Checksums of the 179 session files the two zips unpack to (74 raw `FILE.xml`, 70 `VRSQ_FILE.xml`, 35 curated `FILE.xml`) | — | — |
| `SHA256SUMS.yalcin` | Checksums of the 40 files of the second archive that the analysis reads | — | — |

The archive's other files are not included, because no script uses them: the game build (`GAME collector BUILD 29112019.7z`, 46 MB), the WEKA/ARFF files and the pre-processed 7z archive. They are available at the same upstream commit.

## Preparing the data

```bash
bash scripts/prepare_data.sh
```

The script does three things:
1. Unpacks the two zips into `raw/UFFCSData/RAW_DATABASE/RAW_DATABASE/<74 sessions>` and `raw/UFFCSData/DATABASE_SELECTED_DATA/DATABASE/<35 sessions>`, the paths in `.env.example`.
2. Checks every file against `SHA256SUMS` and `SHA256SUMS.unpacked`.
3. Creates `.env`.

Unpacking reproduces byte for byte the session files that the analysis read.

The derived tables are shipped, so you can skip the parsing step. To rebuild them and confirm they are identical:

```bash
python src/fa_data.py data/raw/UFFCSData/RAW_DATABASE/RAW_DATABASE data data/raw/UFFCSData/DATABASE_SELECTED_DATA/DATABASE
bash verification/verify_data.sh      # sessions.csv byte for byte, features_w30.csv.gz by its content
```

## The second archive (not redistributed)

Three evaluation-design findings are also tested on a second public archive:
- **Source:** Yalcin et al. (2024), *Front. Virtual Real.* 5:1364207, https://github.com/m1237/automatic-cybersickness-detection, MIT licence;
- **Contents:** eye tracking and a button pressed while feeling sick, for 20 participants on a VR roller coaster.

To re-run that analysis:

```bash
bash scripts/fetch_second_archive.sh   # clones it into data/raw/yalcin at commit 14a242a and checks the 40 files used
python src/fa_yalcin.py data/raw/yalcin results
```

These recordings belong to other researchers and their participants, so they are not copied here. The fetched files are checked against `SHA256SUMS.yalcin`. The analysis's results are in `results/rev3_yalcin.json`.

## Participants and personal data

- **What each session records:** seven self-reported demographic answers, in Portuguese: gender, age band, previous VR experience, flicker sensitivity, glasses use, vision problems and eye dominance. Each session also records posture, which marks the recording campaign.
- **No identifiers:** the archive has no names or participant identifiers. Sessions with identical answers form the paper's 38 *inferred groups*; these are answer profiles, not confirmed identities.
- **Already published:** the archive's authors publish these data openly. Nothing beyond the published archive is included here.

## Credit

If you use these data, cite UFFCSData:
- T. Porcino et al., *Entertainment Computing* 41 (2022) 100473;
- T. Porcino, D. Trevisan and E. Clua, XR in Games Workshop (SBGames 2021);
- the repository above.
