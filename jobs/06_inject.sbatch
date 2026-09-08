#!/bin/bash
#SBATCH -J damsa_inject
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH -o logs/inject_%A_%a.out
#SBATCH -e logs/inject_%A_%a.err
#
# Geant4 re-injection of the gamma-gamma pairs, one array task per ALP mass.
#   sbatch --dependency=afterok:<signal_id> jobs/06_inject.sbatch
#
# damsa_alp_inject still reads CSV (alp_generator.h keeps its CSV parser), so
# each task converts its own mass back to CSV first. Teaching that generator to
# read TTree directly is the remaining piece of the migration.
#
# It fires one Geant4 event per CSV row: 4M rows is far too many for a 12 h job,
# so subsample first. SUBSAMPLE preserves total events/day in expectation via
# Horvitz-Thompson reweighting -- do not just `head` the file.

set -euo pipefail
source jobs/env.sh

MASSES=(1 5 10 20 50 100 200 500)
MA="${MASSES[$SLURM_ARRAY_TASK_ID]}"
SUBSAMPLE="${SUBSAMPLE:-200000}"

SRC="output/alp_decay_photons_ma${MA}MeV.root"
CSV="output/alp_decay_photons_ma${MA}MeV.csv"
SUB="output/alp_decay_photons_ma${MA}MeV_sub.csv"

[[ -s "$SRC" ]] || { echo "missing $SRC (run 04_signal first)"; exit 1; }

./build/damsa_convert "$SRC" "$CSV"
./build/subsample_alp_csv "$CSV" "$SUB" "$SUBSAMPLE" $((1234 + SLURM_ARRAY_TASK_ID))
rm -f "$CSV"                      # the full CSV is ~1 GB; the subsample is what runs

echo "=== injecting ma=${MA} MeV, $SUBSAMPLE pairs ==="
./build/damsa_alp_inject "$SUB"
