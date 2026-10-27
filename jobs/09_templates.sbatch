#!/bin/bash
#SBATCH -J damsa_tmpl
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 16:00:00
#SBATCH -o logs/templates_%j.out
#SBATCH -e logs/templates_%j.err
#
# Single-particle templates for the photon ID (Calo-Reco-Explained §2.14):
# gamma, e-, pi+, p, n at several energies, fired along the beam axis from just
# upstream of the Si trackers (z = -99 mm, inside the magnet region), so the
# Si veto and the calo see them as a beam particle would. Uses the standard
# Geant4 /gun commands of the beam generator (one particle per event).
#   sbatch --dependency=afterok:<build_id> jobs/09_templates.sbatch
#
# Ends by writing output/pid_table.csv (species probabilities, Edge-HadronID-ALP
# plan B2; priors from output/tracks_library.csv of an earlier 08_tracks if present)
# and the held-out evaluation plots (B4). 08_tracks then reads the table (--pid).

set -euo pipefail
source jobs/env.sh
mkdir -p output/_tmpl_macros

N="${N_TMPL:-2000}"
PARTICLES=(gamma e- pi+ proton neutron)
ENERGIES=(20 50 100 200 300 500 1000 2000 5000)   # 20-300: the soft region ALP photons live in

for p in "${PARTICLES[@]}"; do
    for E in "${ENERGIES[@]}"; do
        tag="tmpl_$(echo "$p" | tr -d '+-')_E${E}"
        mac="output/_tmpl_macros/${tag}.mac"
        {
            echo "/run/initialize"
            echo "/random/setSeeds $((31 + E)) $((97 + ${#p}))"
            echo "/damsa/setOutputPrefix ${tag}_"
            echo "/damsa/setTrajEventLimit ${TRAJ_EVENTS:-50}"
            echo "/gun/particle $p"
            echo "/gun/energy $E MeV"
            echo "/gun/position 0 0 -99 mm"
            echo "/gun/direction 0 0 1"
            echo "/run/beamOn $N"
        } > "$mac"
        ./build/damsa "$mac" > "logs/templates_${tag}.log" 2>&1
        ./build/damsa_calo_tracks --hits "output/${tag}_calo_hits.root" \
            --truth "output/${tag}_calo_face_particles.root" \
            --tracker "output/${tag}_tracker_hits.root" --traj "output/${tag}_calo_traj.root" \
            --tag "$tag"
    done
done

# Per-species feature distributions and ID efficiency vs rejection; PID table + evaluation.
LIB=""
[[ -s output/tracks_library.csv ]] && LIB=output/tracks_library.csv
./build/damsa_track_plots --dir output --templates "$(IFS=,; echo "${PARTICLES[*]}")" \
    --energies "$(IFS=,; echo "${ENERGIES[*]}")" --n-tmpl "$N" --out plots/templates \
    --pid-table output/pid_table.csv ${LIB:+--library "$LIB"} --pid-eval
