#!/bin/bash
#SBATCH -J damsa_pileup
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 08:00:00
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH -o logs/pileup_%A_%a.out
#SBATCH -e logs/pileup_%A_%a.err
#
# Accidental gamma-gamma rate, one array task per beam mode.
#   sbatch --dependency=afterok:<merge_id> jobs/05_pileup.sbatch
#
# xleap and interleaved are the slow ones: occupancy per gate is enormous, so
# each gate has many photons and the pair loop is O(K^2). --max-pairs and
# --max-photons-per-gate bound it (the tool warns when it starts subsampling and
# says the result is then a lower bound). 8 h is sized for those two; dark and
# lesa finish far sooner.

set -euo pipefail
source jobs/env.sh

MODES=(dark lesa xleap interleaved)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

echo "=== $MODE : full calo-face pool ==="
./build/damsa_pileup \
    --library output/calo_face_particles.root --calo-face \
    --n-library-electrons "${N_ELECTRONS:-1000000}" \
    --beam-mode "$MODE" \
    --n-trials "${N_TRIALS:-100000}" \
    --out-csv "output/pileup_calo_${MODE}.csv"

echo "=== $MODE : pi0 library (geometric acceptance) ==="
# Without --use-geom-accept this is legitimately zero: no pi0 gamma in the
# library survives the material to the calo face.
./build/damsa_pileup \
    --library output/pi0_decays.root --use-geom-accept \
    --n-library-electrons "${N_ELECTRONS:-1000000}" \
    --beam-mode "$MODE" \
    --n-trials "${N_TRIALS:-100000}" \
    --out-csv "output/pileup_pi0_${MODE}.csv"
