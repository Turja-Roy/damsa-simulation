#!/bin/bash
#SBATCH -J damsa_build
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH --mem=16G
#SBATCH -o logs/build_%j.out
#SBATCH -e logs/build_%j.err
#
# Build everything. Submit first; every other job depends on this.
#   sbatch jobs/00_build.sbatch

set -euo pipefail
source jobs/env.sh

# env.sh exports Geant4_DIR when geant4-config is not on PATH.
CMAKE_ARGS=(-DGEANT4_BUILD_MULTITHREADED=ON)
[[ -n "${Geant4_DIR:-}" ]] && CMAKE_ARGS+=("-DGeant4_DIR=$Geant4_DIR")

cmake -S . -B build "${CMAKE_ARGS[@]}"
cmake --build build -j"${SLURM_CPUS_PER_TASK:-8}"

echo "=== built targets ==="
for t in damsa damsa_opt damsa_alp_inject damsa_convert damsa_alp_signal \
         damsa_merge damsa_pileup damsa_io_test alp_test alp_xcheck; do
    [[ -x "build/$t" ]] && echo "  OK   $t" || { echo "  MISSING $t"; exit 1; }
done
echo "Build complete: $(date)"
