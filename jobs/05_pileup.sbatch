#!/bin/bash
#SBATCH -J damsa_pileup
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 08:00:00
#SBATCH --array=0-4
#SBATCH -o logs/pileup_%A_%a.out
#SBATCH -e logs/pileup_%A_%a.err
#
# Truth-level accidental gamma-gamma grid (Random-2-photons plan §3, §4.4).
#   sbatch --dependency=afterok:<merge_id> jobs/05_pileup.sbatch
#
# One CSV row per (beam mode, bunch fraction f, window, mass). Tasks:
#   0  all calo photons, --expected   masses whose window does not reach 0
#   1  case-3 photons,  --expected    (scattered outside / born downstream of target)
#   2  all calo photons, MC            low masses (window reaches 0: no pruning)
#   3  case-3 photons,  MC
#   4  pi0 library (geometric acceptance), MC, one row per mode at the old 1 us gate
#
# --expected is exact and fast: the pair sums are library-only and every cell
# reuses them. The MC tasks are the slow ones -- LESA windows hold thousands of
# photons, so the pair loop is subsampled (--max-pairs) and the tool warns in
# the saturated regime.

set -euo pipefail
source jobs/env.sh

N="${N_ELECTRONS:-1000000}"
LIB=output/calo_face_particles.root
MODES=dark,lesa,xleap                     # interleaved (100 Hz) is outside the ns-window regime
FRACS=0.1,0.2,0.4,0.5,0.6,0.8,1
WINDOWS=5,10,20,30,40,50,60,70,80,90,100
MASS_EXP=30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200
MASS_MC=10,20
COMMON=(--calo-face --n-library-electrons "$N" --beam-mode "$MODES"
        --bunch-frac "$FRACS" --window-ns "$WINDOWS" --mass-window-MeV "${MASS_WINDOW:-20}")

case "$SLURM_ARRAY_TASK_ID" in
  0) ./build/damsa_pileup --library "$LIB" "${COMMON[@]}" --mass-MeV "$MASS_EXP" \
        --expected --out-csv output/pileup_grid_exp.csv ;;
  1) ./build/damsa_pileup --library "$LIB" "${COMMON[@]}" --mass-MeV "$MASS_EXP" \
        --expected --case3 --out-csv output/pileup_grid_exp_case3.csv ;;
  2) ./build/damsa_pileup --library "$LIB" "${COMMON[@]}" --mass-MeV "$MASS_MC" \
        --n-trials "${N_TRIALS:-5000}" --out-csv output/pileup_grid_mc.csv ;;
  3) ./build/damsa_pileup --library "$LIB" "${COMMON[@]}" --mass-MeV "$MASS_MC" \
        --n-trials "${N_TRIALS:-5000}" --case3 --out-csv output/pileup_grid_mc_case3.csv ;;
  4) # Without --use-geom-accept this is legitimately zero: no pi0 gamma in the
     # 1e6 library survives the material to the calo face (plan §0).
     ./build/damsa_pileup --library output/pi0_decays.root --use-geom-accept \
        --n-library-electrons "$N" --beam-mode dark,lesa,xleap \
        --n-trials "${N_TRIALS:-100000}" --out-csv output/pileup_pi0.csv ;;
esac

# Correlated pairs (case 1, non-pi0 included) -- cheap, rides along with task 0.
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    ./build/damsa_pairs --library "$LIB" --n-library-electrons "$N" \
        --out-csv output/pairs_correlated.csv
fi
