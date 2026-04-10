#!/usr/bin/env bash
# run_target_scan.sh
# Phase B: Scan target lengths 11–20 cm for DAMSA geometry optimisation.
#
# Runs Geant4 with 5000 events per target length (sufficient: 296 brem photons
# per primary → 1.5M photon spectrum entries, well above what alplib needs).
#
# Parallelism is limited (default: 4 concurrent jobs) to avoid OOM.
# Running 10 × 1-2 GB Geant4 processes simultaneously causes OOM on most machines.
#
# Usage:
#   cd /My-Drive/Research/damsa-simulation
#   bash run_target_scan.sh            # 4 parallel jobs (default)
#   bash run_target_scan.sh 2          # 2 parallel jobs (safer on low-RAM systems)
#
# Monitor progress:
#   tail -f logs/Tz*.log
#
# After completion, run Phase C:
#   bash run_phase_c.sh

set -euo pipefail

BUILD_DIR="$(pwd)/build"
EXEC="${BUILD_DIR}/damsa"
LOG_DIR="$(pwd)/logs"
MAX_PARALLEL="${1:-4}"   # first argument or default 4

if [[ ! -x "${EXEC}" ]]; then
    echo "ERROR: ${EXEC} not found or not executable. Run: cd build && make -j\$(nproc)"
    exit 1
fi

mkdir -p "${LOG_DIR}"

TARGET_LENGTHS=(11 12 13 14 15 16 17 18 19 20)

echo "=== DAMSA Target Length Scan — Phase B ==="
echo "Executable:  ${EXEC}"
echo "Targets:     ${TARGET_LENGTHS[*]} cm"
echo "Events/run:  5000"
echo "Max parallel: ${MAX_PARALLEL}"
echo ""

running=0
declare -a PIDS=()
declare -a PLABELS=()

wait_one() {
    # Wait for any single job to finish, report it, decrement counter
    local idx pid label exit_code
    while true; do
        for idx in "${!PIDS[@]}"; do
            pid="${PIDS[$idx]}"
            label="${PLABELS[$idx]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" && exit_code=0 || exit_code=$?
                if [[ $exit_code -eq 0 ]]; then
                    echo "  ✓ ${label} (PID ${pid}) done"
                else
                    echo "  ✗ ${label} (PID ${pid}) FAILED (exit ${exit_code}) — check ${LOG_DIR}/${label}.log"
                fi
                unset "PIDS[$idx]"
                unset "PLABELS[$idx]"
                PIDS=("${PIDS[@]+"${PIDS[@]}"}")
                PLABELS=("${PLABELS[@]+"${PLABELS[@]}"}")
                running=$((running - 1))
                return
            fi
        done
        sleep 5
    done
}

FAILED=0

for N in "${TARGET_LENGTHS[@]}"; do
    OUTDIR="output/Tz${N}"
    MACRO="macros/target_scan_Tz${N}.mac"
    LOG="${LOG_DIR}/Tz${N}.log"
    LABEL="Tz${N}"

    mkdir -p "${OUTDIR}"

    # Throttle: wait if at capacity
    while [[ $running -ge $MAX_PARALLEL ]]; do
        wait_one || FAILED=$((FAILED + 1))
    done

    echo "Launching: target=${N} cm  (${running}/${MAX_PARALLEL} slots used)"
    "${EXEC}" "${MACRO}" > "${LOG}" 2>&1 &
    PIDS+=($!)
    PLABELS+=("${LABEL}")
    running=$((running + 1))
done

# Drain remaining jobs
while [[ $running -gt 0 ]]; do
    wait_one || FAILED=$((FAILED + 1))
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All 10 runs completed successfully."
    echo ""
    echo "Verify outputs:"
    for N in "${TARGET_LENGTHS[@]}"; do
        has_flux=$(test -f "output/Tz${N}/alplib_brems_flux.csv" && echo "✓" || echo "✗")
        has_bkg=$(test -f "output/Tz${N}/all_particles_target_exit.csv" && echo "✓" || echo "✗")
        echo "  Tz${N}: flux=${has_flux}  particles=${has_bkg}"
    done
    echo ""
    echo "Next step: bash run_phase_c.sh"
else
    echo "${FAILED} run(s) failed. Check logs in ${LOG_DIR}/."
    exit 1
fi
