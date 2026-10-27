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
#
# Saturation-gap scan (mu between dark current and LESA f=0.1), LESA only:
#   FRACS=0.001,0.002,0.005,0.01,0.02,0.05 TAG=_gap sbatch --array=1 jobs/07_reco.sbatch
#
# Reconstruction (Calo-Reco-Explained §2.13): depth-aware is the default.
#   RECO=summed TAG=_summed     sbatch jobs/07_reco.sbatch   # old: layers summed, energy-rank x/y
#
# Fiducial margin (Edge-HadronID-ALP plan A4): photons and truth pairs need the shower core
# >= FID mm from every calo side (default 10). Margin scan, dark + LESA:
#   for m in 0 5 15; do FID=$m TAG=_fid$m sbatch --array=0,1 jobs/07_reco.sbatch; done

set -euo pipefail
source jobs/env.sh

[[ -s output/calo_hits.root ]] || {
    echo "missing output/calo_hits.root: rebuild the library with the crystal calo"
    echo "(02_library, then 03_merge) before 07_reco"; exit 1; }

# Si veto is opt-in (§2.14): with no Si timing yet, every charged track in the
# window masks 2 R_M of calo, so at LESA (mu ~ 400) it kills ~85 % of photons.
#   SIVETO=1 TAG=_siveto sbatch jobs/07_reco.sbatch
TRACKER=""
[[ "${SIVETO:-0}" == 1 && -s output/tracker_hits.root ]] && TRACKER=output/tracker_hits.root

MODES=(dark lesa xleap)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

./build/damsa_calo_reco \
    --hits output/calo_hits.root --truth output/calo_face_particles.root \
    ${TRACKER:+--tracker "$TRACKER"} \
    --n-library-electrons "${N_ELECTRONS:-1000000}" \
    --beam-mode "$MODE" \
    --bunch-frac "${FRACS:-0.1,0.2,0.4,0.5,0.6,0.8,1}" \
    --window-ns 5,10,20,30,40,50,60,70,80,90,100 \
    --sigma-t-ns 1,5,10,20,30,40,50,60,70,80,90,100 \
    --mass-MeV 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 \
    --e-bins-GeV 1,2,3,4 \
    --n-trials "${N_TRIALS:-2000}" \
    --reco "${RECO:-depth}" --fid-margin-mm "${FID:-10}" \
    --out-csv "output/reco_grid_${MODE}${TAG:-}.csv"
