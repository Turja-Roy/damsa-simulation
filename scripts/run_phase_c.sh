#!/usr/bin/env bash
# run_phase_c.sh
# Phase C: For each target length output, generate ALP decay CSVs and run
# the joint VDC × calo analytic scan. Then combine into global Pareto front.
#
# Requires Phase B to have completed (output/Tz<N>/ directories exist with
# alplib_brems_flux.csv and all_particles_target_exit.csv inside each).
#
# Usage:
#   cd /My-Drive/Research/damsa-simulation
#   bash run_phase_c.sh

set -euo pipefail

TARGET_LENGTHS=(10 11 12 13 14 15 16 17 18 19 20)
MA_LIST="10 20 50 100 200"
N_SAMPLES=20000
# n_primaries for Tz10 (100k original run); Phase B runs used 5000
N_PRIMARIES_TZ10=100000
N_PRIMARIES_PHASE_B=5000
PARETO_FILES=()

echo "=== DAMSA Phase C: Joint VDC × Calo scan per target length ==="
echo ""

for N in "${TARGET_LENGTHS[@]}"; do
    OUTDIR="output/Tz${N}"
    FLUX="${OUTDIR}/alplib_brems_flux.csv"
    PARTICLES="${OUTDIR}/all_particles_target_exit.csv"
    LABEL="Tz${N}"

    # Tz10 uses the original output directory
    if [[ "${N}" == "10" ]]; then
        FLUX="output/alplib_brems_flux.csv"
        PARTICLES="output/all_particles_target_exit.csv"
    fi

    if [[ ! -f "${FLUX}" ]]; then
        echo "WARNING: ${FLUX} not found — skipping Tz${N}"
        continue
    fi
    if [[ ! -f "${PARTICLES}" ]]; then
        echo "WARNING: ${PARTICLES} not found — skipping Tz${N}"
        continue
    fi

    echo "--- Processing Tz${N} ---"

    # Use correct n_primaries for normalization
    if [[ "${N}" == "10" ]]; then
        N_PRIMARIES=${N_PRIMARIES_TZ10}
    else
        N_PRIMARIES=${N_PRIMARIES_PHASE_B}
    fi

    # Step 2: Run joint VDC × calo scan
    echo "  Running joint VDC × calo scan (n_primaries=${N_PRIMARIES})..."
    python scripts/joint_pareto_scan.py \
        --particles   "${PARTICLES}" \
        --flux        "${FLUX}" \
        --n-primaries "${N_PRIMARIES}" \
        --vdc-min 30 --vdc-max 40 --vdc-step 2 \
        --calo-min 12 --calo-max 20 --calo-step 2 \
        --ma-list ${MA_LIST} \
        --n-samples ${N_SAMPLES} \
        --label "${LABEL}" \
        --output-dir output/joint_pareto

    PARETO_FILE="output/joint_pareto/${LABEL}_pareto.csv"
    if [[ -f "${PARETO_FILE}" ]]; then
        PARETO_FILES+=("${PARETO_FILE}")
        echo "  ✓ Pareto saved: ${PARETO_FILE}"
    fi
    echo ""
done

# Step 3: Combine into global Pareto front
if [[ ${#PARETO_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No Pareto files generated."
    exit 1
fi

echo "=== Combining ${#PARETO_FILES[@]} Pareto fronts into global Pareto ==="
python scripts/combine_pareto.py \
    "${PARETO_FILES[@]}" \
    --output output/global_pareto.csv \
    --plot-dir plots/global_pareto

echo ""
echo "=== Phase C complete ==="
echo "Global Pareto: output/global_pareto.csv"
echo "Global plots:  plots/global_pareto/"
