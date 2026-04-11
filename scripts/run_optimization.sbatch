#!/bin/bash
#SBATCH -J damsa_opt
#SBATCH -p normal
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=256GB

# ============================================================
# DAMSA Optimization Work - Phase B + Phase C
# Runs Geant4 target scan followed by Pareto analysis
# ============================================================

# Load required modules
module load geant4/10.7
module load root/6.26
module load python/3.10

# Set up Geant4 environment
source /opt/geant4/10.7/share/Geant4/geant4make/geant4env.sh

# Set OpenMP threads (10 parallel jobs × 12 threads = 120 total, fits in 128-core node)
export OMP_NUM_THREADS=12

# Working directory
WORKDIR=$WORKDIR

echo "=== DAMSA Optimization Pipeline ==="
echo "Job started: $(date)"
echo "Working directory: ${WORKDIR}"
echo "OMP threads: ${OMP_NUM_THREADS}"
echo ""

# ------------------------------------------------------------
# Build the project
# ------------------------------------------------------------
echo "=== Building project ==="
cd ${WORKDIR}/build
cmake ..
make -j$(nproc)

if [[ ! -x "${WORKDIR}/build/damsa" ]]; then
    echo "ERROR: Build failed - damsa executable not found"
    exit 1
fi
echo "Build complete."
echo ""

# ------------------------------------------------------------
# Phase B: Target length scan (10 parallel jobs)
# ------------------------------------------------------------
echo "=== Phase B: Target Length Scan ==="
cd ${WORKDIR}
bash run_target_scan.sh 10

echo "Phase B complete."
echo ""

# ------------------------------------------------------------
# Phase C: Pareto optimization analysis
# ------------------------------------------------------------
echo "=== Phase C: Pareto Analysis ==="
bash run_phase_c.sh

echo "Phase C complete."
echo ""
echo "=== DAMSA Optimization Pipeline Finished ==="
echo "Job finished: $(date)"
