#!/bin/bash
#SBATCH -J damsa_tracks
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o logs/tracks_%j.out
#SBATCH -e logs/tracks_%j.err
#
# Depth-aware calo reconstruction + photon ID, event by event (Calo-Reco-Explained §2.13-2.14):
#   1. SM two-photon gun (pi0, eta) at several energies, vertices across the VDC,
#      fired through Geant4 with damsa_alp_inject (it fires any gamma-gamma pair);
#   2. damsa_calo_tracks on each sample (Si veto + truth trajectories);
#   3. the same on the beam library (single electrons, no overlay);
#   4. the ALP samples of 06_inject, one per mass, if their calo hits exist
#      (Edge-HadronID-ALP plan C1; 06_inject with SUBSAMPLE=20000 is enough);
#   5. damsa_track_plots per sample, all objects vs photon ID (+ the grid
#      comparison if the 07_reco variants exist: reco_grid_lesa{,_summed}.csv).
# With output/pid_table.csv (09_templates) every object gets species probabilities.
#   sbatch --dependency=afterok:<build_id> jobs/08_tracks.sbatch

set -euo pipefail
source jobs/env.sh

N="${N_GUN:-5000}"
PID=""
[[ -s output/pid_table.csv ]] && PID=output/pid_table.csv
SAMPLES=("pi0 134.977 500" "pi0 134.977 1000" "pi0 134.977 2000" "eta 547.862 1000" "eta 547.862 2000")

for s in "${SAMPLES[@]}"; do
    read -r name mass E <<< "$s"
    tag="${name}_E${E}"
    ./build/damsa_gun --mass-MeV "$mass" --energy-MeV "$E" --n "$N" --out "output/gun_${tag}.root"
    ./build/damsa_alp_inject "output/gun_${tag}.root" macros/run_alp.mac > "logs/tracks_inject_${tag}.log" 2>&1
    g="output/gun_${tag}"
    ./build/damsa_calo_tracks --hits "${g}_calo_hits.root" --truth "${g}_calo_face_particles.root" \
        --tracker "${g}_tracker_hits.root" --traj "${g}_calo_traj.root" --tag "$tag" ${PID:+--pid "$PID"}
done

# Beam library, single electrons (truth photons >= 10 MeV for the efficiency).
./build/damsa_calo_tracks --hits output/calo_hits.root --truth output/calo_face_particles.root \
    --tracker output/tracker_hits.root --traj output/calo_traj.root \
    --tag library --truth-min-MeV 10 --n-display 40 ${PID:+--pid "$PID"}

# ALP signal samples (06_inject outputs).
ALPS=()
for m in 1 5 10 20 50 100 200 500; do
    a="output/alp_decay_photons_ma${m}MeV_sub"
    [[ -s "${a}_calo_hits.root" ]] || { echo "skip ALP ma=${m} MeV: no ${a}_calo_hits.root (rerun 06_inject)"; continue; }
    ./build/damsa_calo_tracks --hits "${a}_calo_hits.root" --truth "${a}_calo_face_particles.root" \
        --tracker "${a}_tracker_hits.root" --traj "${a}_calo_traj.root" --tag "alp_ma${m}" ${PID:+--pid "$PID"}
    ALPS+=("alp_ma${m}")
done

GRID=""
if [[ -s output/reco_grid_lesa_summed.csv ]]; then
    GRID="old summed=output/reco_grid_lesa_summed.csv,depth + photon ID=output/reco_grid_lesa.csv"
fi
TAGS=(library ${ALPS[@]+"${ALPS[@]}"})   # empty-array safe under set -u on old bash
for s in "${SAMPLES[@]}"; do read -r name _ E <<< "$s"; TAGS+=("${name}_E${E}"); done
for tag in "${TAGS[@]}"; do
    ./build/damsa_track_plots --dir output --tag "$tag" ${GRID:+--grid "$GRID"} --out plots/tracks
done
