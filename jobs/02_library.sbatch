#!/bin/bash
#SBATCH -J damsa_lib
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH --mem=24G
#SBATCH --array=0-19
#SBATCH -o logs/lib_%A_%a.out
#SBATCH -e logs/lib_%A_%a.err
#
# The 1e6-electron library, as a Slurm ARRAY: 20 chunks x 50k electrons, one
# task each. This is the shape run_library_local.sh wanted -- it ran the same
# 20 chunks sequentially because a laptop had no other option.
#
#   sbatch --dependency=afterok:<build_id> jobs/02_library.sbatch
#   sbatch --dependency=afterok:<lib_id>   jobs/03_merge.sbatch
#
# Chunking (not parallelism) is what avoids the OOM: a monolithic 1e6 beamOn
# holds every scored particle in memory at once. Each task is its own process.
#
# EVENTS=25000 sbatch --array=0-39 jobs/02_library.sbatch   # if 50k still OOMs

set -euo pipefail
source jobs/env.sh
mkdir -p logs output/_chunk_macros

EVENTS="${EVENTS:-50000}"
i="${SLURM_ARRAY_TASK_ID}"
tag=$(printf "chunk%02d" "$i")

# Distinct, reproducible seeds per chunk. Without this every chunk is identical:
# the default Geant4 seed is fixed.
s1=$(( 10007 + 2*i ))
s2=$(( 20011 + 2*i ))

# Resume: skip a chunk whose keepers already exist.
if [[ -s "output/${tag}_alplib_brems_flux.csv" \
   && -s "output/${tag}_pi0_decays.root" \
   && -s "output/${tag}_calo_face_particles.root" ]]; then
    echo "[$tag] already done, skipping"
    exit 0
fi

mac="output/_chunk_macros/${tag}.mac"
{
    echo "/run/initialize"
    echo "/random/setSeeds $s1 $s2"
    echo "/damsa/setOutputPrefix ${tag}_"   # slash-free: a '/' double-nests, see CLAUDE.md
    echo "/run/beamOn $EVENTS"
} > "$mac"

echo "[$tag] $EVENTS events, seeds=($s1,$s2), $(date)"
./build/damsa "$mac"

# Keep only what the merge and steps 2-6 consume. The rest are per-electron exit
# dumps (~7 GB/chunk) that only the separate geometry-optimization workflow
# reads, and it makes its own via damsa_opt. KEEP_ALL=1 to retain everything.
#
# Both formats are kept while the migration is being verified (01_verify diffs
# them). Once the CSVs are retired, drop them here and from run.h.
if [[ "${KEEP_ALL:-0}" != "1" ]]; then
    for g in output/${tag}_*; do
        case "$g" in
            output/${tag}_alplib_brems_flux.csv|\
            output/${tag}_pi0_decays.root|\
            output/${tag}_calo_face_particles.root|\
            output/${tag}_pi0_decays.csv|\
            output/${tag}_calo_face_particles.csv) : ;;      # keep
            *) rm -f "$g" ;;
        esac
    done
fi
echo "[$tag] done $(date); kept $(ls output/${tag}_* 2>/dev/null | wc -l) files"
