#!/bin/bash
#SBATCH -J damsa_reco
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o logs/reco_%A_%a.out
#SBATCH -e logs/reco_%A_%a.err
#
# SM calorimeter reconstruction vs truth over the plan §3 grid, one task per
# beam mode (Random-2-photons plan §5.4-§7).
#   sbatch --dependency=afterok:<merge_id> jobs/07_reco.sbatch
#
# Needs the merged library WITH calo_hits.root (02_library + 03_merge after
# the crystal restore). One CSV row per (f, window, sigma_t, mass, E bin, z cut).

set -euo pipefail
source jobs/env.sh

MODES=(dark lesa xleap)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

./build/damsa_calo_reco \
    --hits output/calo_hits.root --truth output/calo_face_particles.root \
    --n-library-electrons "${N_ELECTRONS:-1000000}" \
    --beam-mode "$MODE" \
    --bunch-frac 0.1,0.2,0.4,0.5,0.6,0.8,1 \
    --window-ns 5,10,20,30,40,50,60,70,80,90,100 \
    --sigma-t-ns 1,5,10,20,30,40,50,60,70,80,90,100 \
    --mass-MeV 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 \
    --e-bins-GeV 1,2,3,4 \
    --n-trials "${N_TRIALS:-2000}" \
    --out-csv "output/reco_grid_${MODE}.csv"
