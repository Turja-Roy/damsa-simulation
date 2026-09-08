#!/usr/bin/env bash
# DAMSA C++ migration - cluster runbook.
#
# Heavy work runs as SBATCH JOBS (jobs/*.sbatch), not on the login node.
# Only the rsync in section 0 and the submissions in section 2 are typed at a
# prompt; everything else is submitted.
#
# Not meant to be executed top to bottom: copy blocks out of it.

set -u
LOCAL=/My-Drive/Research/damsa-simulation
CLUSTER=txr7972@master2.tier3-atlas.uta.edu
REMOTE=damsa-cpp                         # new folder, does not touch the old one

# ─────────────────────────────────────────────────────────────────────────────
# 0. Upload  (run on the LAPTOP)
# ─────────────────────────────────────────────────────────────────────────────
# Source + the alplib reference oracle + jobs/. Excludes build products and the
# 7.6 GB of old output, which the cluster regenerates.
rsync -avz --progress \
  --exclude 'build*/' --exclude 'output/' --exclude 'output_old/' \
  --exclude 'plots/'  --exclude 'plots_old/' --exclude '.git/' \
  --exclude '.venv/'  --exclude '__pycache__/' --exclude 'Ref/' --exclude 'Reading/' \
  "$LOCAL"/ "$CLUSTER:$REMOTE"/

# Inputs the verification job needs (~5 MB total).
rsync -avz --progress \
  "$LOCAL"/output/chunk*_alplib_brems_flux.csv \
  "$LOCAL"/output/pi0_decays.csv \
  "$LOCAL"/output/alplib_brems_flux.csv \
  "$CLUSTER:$REMOTE/output/"

ssh "$CLUSTER"
cd damsa-cpp
mkdir -p logs

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check the environment BEFORE submitting anything
# ─────────────────────────────────────────────────────────────────────────────
module avail geant4 2>&1 | head
module avail root   2>&1 | head
sinfo -s                      # partition names and limits; jobs/*.sbatch assume -p normal

# Then edit jobs/env.sh so its module lines match what actually exists.
# ROOT version does NOT matter: storage is TTree, which reads and writes the
# same on 6.26 and 6.40. (RNTuple would have needed >= 6.36 -- that is why the
# backend is TTree.)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Submit. Each job depends on the previous one, so you can queue them all.
# ─────────────────────────────────────────────────────────────────────────────
BUILD=$(sbatch --parsable jobs/00_build.sbatch)                                  ; echo "build   $BUILD"
VERIFY=$(sbatch --parsable --dependency=afterok:$BUILD  jobs/01_verify.sbatch)   ; echo "verify  $VERIFY"

# STOP HERE and read logs/verify_$VERIFY.out. It ends in
#   ================ VERIFY PASSED ================
# Do not run production on a FAILED verify.

LIB=$(sbatch    --parsable --dependency=afterok:$BUILD  jobs/02_library.sbatch)  ; echo "library $LIB"
MERGE=$(sbatch  --parsable --dependency=afterok:$LIB    jobs/03_merge.sbatch)    ; echo "merge   $MERGE"
SIG=$(sbatch    --parsable --dependency=afterok:$MERGE  jobs/04_signal.sbatch)   ; echo "signal  $SIG"
PILE=$(sbatch   --parsable --dependency=afterok:$MERGE  jobs/05_pileup.sbatch)   ; echo "pileup  $PILE"
INJ=$(sbatch    --parsable --dependency=afterok:$SIG    jobs/06_inject.sbatch)   ; echo "inject  $INJ"

squeue -u "$USER"
# sacct -j $LIB --format=JobID,JobName%20,State,Elapsed,MaxRSS,ExitCode

# ─────────────────────────────────────────────────────────────────────────────
# 3. What each job is
# ─────────────────────────────────────────────────────────────────────────────
#  00_build    16 cpu   40 min   builds all 15 targets, fails loudly if any is missing
#  01_verify    8 cpu    2 h     every cross-check vs the Python; ends PASSED/FAILED
#  02_library   8 cpu    4 h     ARRAY 0-19: 20 x 50k electrons = 1e6, one chunk per task
#  03_merge     4 cpu    1 h     64G: the calo-face merge renumbers eventIDs in memory
#  04_signal    4 cpu    2 h     8 masses x 4M decay pairs, ~2 GB of TTree
#  05_pileup    2 cpu    8 h     ARRAY 0-3: one beam mode each; xleap/interleaved are slow
#  06_inject    8 cpu   12 h     ARRAY 0-7: one ALP mass each, Geant4 re-injection
#
# Sizing came from a local 2000-event run (16 s on ~7 cores) and a full-scale
# single-mass export (3.95M pairs in 10 s). Cluster I/O is slower, hence the
# headroom. Tune -t and --mem once you have one sacct line to look at.

# ─────────────────────────────────────────────────────────────────────────────
# 4. Knobs
# ─────────────────────────────────────────────────────────────────────────────
# If a 50k chunk still OOMs, shrink the chunk and add tasks (product stays 1e6):
#   EVENTS=25000 sbatch --array=0-39 jobs/02_library.sbatch
#   NCHUNKS=40   sbatch jobs/03_merge.sbatch
#
# Keep the full per-chunk output (~7 GB/chunk, normally pruned):
#   KEEP_ALL=1 sbatch jobs/02_library.sbatch
#
# Cheaper pileup while testing:
#   N_TRIALS=10000 sbatch jobs/05_pileup.sbatch
#
# Fewer injected pairs per mass:
#   SUBSAMPLE=50000 sbatch jobs/06_inject.sbatch

# ─────────────────────────────────────────────────────────────────────────────
# 5. Two results that look like failures but are not
# ─────────────────────────────────────────────────────────────────────────────
# * The merged flux differs from merge_library.py's on exactly one line: the
#   "MERGED from 20 chunks by ..." provenance line. Any difference in a DATA row
#   is a real failure. 01_verify already skips that line.
#
# * Pileup on the pi0 library WITHOUT --use-geom-accept is zero on both sides.
#   In the 1e6-electron library no pi0 gamma survives the material to the calo
#   face (AtCalo = 0 for all 5532 decays; 89/87 pass pure geometric acceptance).
#   That is physics. 05_pileup runs both variants.

# ─────────────────────────────────────────────────────────────────────────────
# 6. Pull results back  (run on the LAPTOP)
# ─────────────────────────────────────────────────────────────────────────────
# Everything except the multi-hundred-MB decay files:
# rsync -avz --progress --exclude 'alp_decay_photons_*' \
#   "$CLUSTER:$REMOTE/output/" "$LOCAL/output_cluster/"
# rsync -avz "$CLUSTER:$REMOTE/logs/" "$LOCAL/logs_cluster/"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Migrating the old 1 GB CSVs (optional, only if you want them kept)
# ─────────────────────────────────────────────────────────────────────────────
# On the cluster, as a job -- not on the login node:
#   sbatch --wrap='for f in output/alp_decay_photons_ma*MeV.csv; do
#                    ./build/damsa_convert "$f" "${f%.csv}.root"; done' \
#          -p normal -c 2 -t 4:00:00 --mem=8G -o logs/convert_%j.out
# ~7.9 GB -> ~2.1 GB.
