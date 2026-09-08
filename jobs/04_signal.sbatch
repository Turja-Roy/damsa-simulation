#!/bin/bash
#SBATCH -J damsa_signal
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH --mem=16G
#SBATCH -o logs/signal_%j.out
#SBATCH -e logs/signal_%j.err
#
#   sbatch --dependency=afterok:<merge_id> jobs/04_signal.sbatch
#
# 8 masses x 4M decay pairs. Measured locally: ~10 s and 243 MB per mass, so
# ~2 min and ~2 GB total (the CSVs this replaces were 7.9 GB). The sensitivity
# scan (60 couplings x 8 masses) adds a few minutes. Single-threaded; the wall
# limit is generous because cluster I/O is slower than a local SSD.

set -euo pipefail
source jobs/env.sh

./build/damsa_alp_signal \
    --flux output/alplib_brems_flux.csv \
    --alplib alplib \
    --auto-coupling \
    --outdir output \
    --decay-samples 500

echo "=== decay 4-vector files ==="
ls -lh output/alp_decay_photons_ma*MeV.root
du -sh output/
