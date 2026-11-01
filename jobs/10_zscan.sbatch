#!/bin/bash
#SBATCH -J damsa_zscan
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o logs/zscan_%j.out
#SBATCH -e logs/zscan_%j.err
#
# Template origin z-scan: how the calo object of a single particle depends on
# where along the axis it starts, i.e. which material it crosses (front cap,
# back cap, Si planes), its incidence angles and its flight time. One fixed z
# per run (a one-row /damsa/setGunOriginTable), so every run has a known z0.
# A uniform spread would blend them without a per-event z0 in the output.
#   z0 = -400 target exit (both caps, 6 Si)   -250 mid VDC (back cap, 6 Si)
#        -106 just before the back cap        -99 just after it (old templates)
#         -57 after 2 Si planes                12 after all Si, 8 mm from the face
#   sbatch --dependency=afterok:<build_id> jobs/10_zscan.sbatch
# Output: output/tracks_zs<z0>_<p>_E<E>.csv, plus one summary line per run
# (objects, EM/MIP/hadronic, Si-vetoed) from damsa_calo_tracks in the .out log.

set -euo pipefail
source jobs/env.sh
mkdir -p output/_zscan_macros

N="${N_ZSCAN:-2000}"
PARTICLES=(${ZSCAN_PARTICLES:-gamma neutron})
ENERGIES=(${ZSCAN_ENERGIES:-20 100 1000})
ZS=(${ZSCAN_Z:--400 -250 -106 -99 -57 12})

for z in "${ZS[@]}"; do
    table="output/_zscan_macros/z${z}.csv"
    printf 'z_lo_mm,z_hi_mm,weight\n%s,%s,1\n' "$z" "$z" > "$table"
    for p in "${PARTICLES[@]}"; do
        for E in "${ENERGIES[@]}"; do
            tag="zs${z}_$(echo "$p" | tr -d '+-')_E${E}"
            mac="output/_zscan_macros/${tag}.mac"
            {
                echo "/run/initialize"
                echo "/random/setSeeds $((31 + E)) $((97 + ${#p}))"
                echo "/damsa/setOutputPrefix ${tag}_"
                echo "/gun/particle $p"
                echo "/gun/energy $E MeV"
                echo "/damsa/setGunOriginTable $table"
                echo "/damsa/setGunSpread 60 mm"
                echo "/damsa/setGunMaxSlope 0.3"
                echo "/run/beamOn $N"
            } > "$mac"
            ./build/damsa "$mac" > "logs/zscan_${tag}.log" 2>&1
            ./build/damsa_calo_tracks --hits "output/${tag}_calo_hits.root" \
                --truth "output/${tag}_calo_face_particles.root" \
                --tracker "output/${tag}_tracker_hits.root" --n-display 0 --tag "$tag"
        done
    done
done
