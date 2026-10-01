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
# Depth-aware calo reconstruction, event by event (Calo-Reco-Explained §2.13):
#   1. SM two-photon gun (pi0, eta) at several energies, vertices across the VDC,
#      fired through Geant4 with damsa_alp_inject (it fires any gamma-gamma pair);
#   2. damsa_calo_tracks on each sample, energy-integrated and boolean readout;
#   3. the same on the beam library (single electrons, no overlay);
#   4. damsa_track_plots per sample (+ the grid comparison if the 07_reco
#      variants exist: reco_grid_lesa{,_summed,_noE}.csv).
#   sbatch --dependency=afterok:<build_id> jobs/08_tracks.sbatch

set -euo pipefail
source jobs/env.sh

N="${N_GUN:-5000}"
SAMPLES=("pi0 134.977 500" "pi0 134.977 1000" "pi0 134.977 2000" "eta 547.862 1000" "eta 547.862 2000")

for s in "${SAMPLES[@]}"; do
    read -r name mass E <<< "$s"
    tag="${name}_E${E}"
    ./build/damsa_gun --mass-MeV "$mass" --energy-MeV "$E" --n "$N" --out "output/gun_${tag}.root"
    ./build/damsa_alp_inject "output/gun_${tag}.root" macros/run_alp.mac > "logs/tracks_inject_${tag}.log" 2>&1
    for mode in "" "--no-energy"; do
        ./build/damsa_calo_tracks --hits "output/gun_${tag}_calo_hits.root" \
            --truth "output/gun_${tag}_calo_face_particles.root" --tag "$tag" $mode
    done
done

# Beam library, single electrons (truth photons >= 10 MeV for the efficiency).
for mode in "" "--no-energy"; do
    ./build/damsa_calo_tracks --hits output/calo_hits.root --truth output/calo_face_particles.root \
        --tag library --truth-min-MeV 10 --n-display 40 $mode
done

GRID=""
if [[ -s output/reco_grid_lesa_summed.csv && -s output/reco_grid_lesa_noE.csv ]]; then
    GRID="old summed=output/reco_grid_lesa_summed.csv,depth+energy=output/reco_grid_lesa.csv,depth boolean=output/reco_grid_lesa_noE.csv"
fi
for s in "${SAMPLES[@]}" "x x library"; do
    read -r name _ E <<< "$s"
    tag=$([[ "$name" == x ]] && echo library || echo "${name}_E${E}")
    ./build/damsa_track_plots --dir output --tag "$tag" ${GRID:+--grid "$GRID"} --out plots/tracks
done
