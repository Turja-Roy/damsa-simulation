#!/bin/bash
#SBATCH -J damsa_merge
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH --mem=64G
#SBATCH -o logs/merge_%j.out
#SBATCH -e logs/merge_%j.err
#
#   sbatch --dependency=afterok:<lib_array_id> jobs/03_merge.sbatch
#
# Memory: the calo-face merge holds every row in memory to renumber eventIDs.
# 1e6 electrons is a few GB; 64G is deliberate headroom.

set -euo pipefail
source jobs/env.sh
NCHUNKS="${NCHUNKS:-20}"

./build/damsa_merge --nchunks "$NCHUNKS"

echo "=== canonical outputs ==="
ls -lh output/alplib_brems_flux.csv output/pi0_decays.root output/calo_face_particles.root
# The flux merge is the one that cannot be a concatenation: it reconstructs the
# integer count per bin from each chunk, sums, then renormalizes by the TOTAL
# electrons. A naive concat here is silently wrong physics.
