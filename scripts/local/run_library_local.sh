#!/bin/bash
# ============================================================
# Local chunked replacement for the multi-hour 1e6 library run
# (AUDIT.md step 1).  Runs N independent short beamOn chunks
# SEQUENTIALLY — each is its own process, so RAM is freed
# between chunks and the 1e6 monolithic OOM never happens.
# Distinct RNG seed per chunk (default seed is fixed → chunks
# would otherwise be identical).  Then merges into the canonical
# output/{alplib_brems_flux,pi0_decays,calo_face_particles}.csv
# that steps 2-6 consume.
#
# Usage (from repo root):
#   bash scripts/local/run_library_local.sh
#   NCHUNKS=40 EVENTS=25000 bash scripts/local/run_library_local.sh
#
# Defaults: 20 chunks x 50000 = 1e6 electrons total.
# If a 50k chunk still crashes, drop EVENTS (e.g. 20000) and
# raise NCHUNKS to keep the product at 1e6.
# ============================================================
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

NCHUNKS="${NCHUNKS:-20}"
EVENTS="${EVENTS:-50000}"
DAMSA="${DAMSA:-./build/damsa}"
MACDIR="output/_chunk_macros"

[[ -x "$DAMSA" ]] || { echo "ERROR: $DAMSA not built. Run: (cd build && cmake .. && make -j)"; exit 1; }
mkdir -p "$MACDIR" output

total=$(( NCHUNKS * EVENTS ))
echo "=== local chunked library run ==="
echo "chunks=$NCHUNKS  events/chunk=$EVENTS  total=$total electrons"
echo "started: $(date)"

for i in $(seq 0 $((NCHUNKS-1))); do
    tag=$(printf "chunk%02d" "$i")
    mac="$MACDIR/${tag}.mac"
    # Distinct, reproducible seeds per chunk.
    s1=$(( 10007 + 2*i ))
    s2=$(( 20011 + 2*i ))
    {
        echo "/run/initialize"
        echo "/random/setSeeds $s1 $s2"
        echo "/damsa/setOutputPrefix ${tag}_"   # slash-free (CLAUDE.md gotcha)
        echo "/run/beamOn $EVENTS"
    } > "$mac"

    # Resume: skip a chunk whose 3 keepers already exist (from a prior run).
    if [[ -s "output/${tag}_alplib_brems_flux.csv" \
       && -s "output/${tag}_pi0_decays.csv" \
       && -s "output/${tag}_calo_face_particles.csv" ]]; then
        echo "--- [$((i+1))/$NCHUNKS] $tag  already done, skipping ---"
        continue
    fi

    echo "--- [$((i+1))/$NCHUNKS] $tag  seeds=($s1,$s2) ---"
    "$DAMSA" "$mac"

    # Keep ONLY the 3 files the merge/steps 2-6 need. The rest are the huge
    # per-electron exit dumps (all_particles/background/photon_flux ~GB/chunk)
    # + raw brems + .root histograms. Those exit dumps are consumed ONLY by the
    # separate geometry-optimization (Pareto/target-scan) workflow, which makes
    # its own via damsa_opt — nothing downstream of THIS library run reads them.
    # So prune immediately to keep disk flat.  Set KEEP_ALL=1 to skip pruning
    # (WARNING: ~7 GB per chunk).
    if [[ "${KEEP_ALL:-0}" != "1" ]]; then
        for g in output/${tag}_*; do
            case "$g" in
                output/${tag}_alplib_brems_flux.csv|\
                output/${tag}_pi0_decays.csv|\
                output/${tag}_calo_face_particles.csv) : ;;   # keep
                *) rm -f "$g" ;;
            esac
        done
    fi
    echo "    kept $(ls output/${tag}_* 2>/dev/null | wc -l) files; output/ now $(du -sh output 2>/dev/null | cut -f1)"
done

echo "=== chunks done, merging: $(date) ==="
python3 scripts/local/merge_library.py --nchunks "$NCHUNKS"

echo "=== merged canonical outputs ==="
ls -lh output/alplib_brems_flux.csv output/pi0_decays.csv output/calo_face_particles.csv
echo "done: $(date)"
echo
echo "Next: run AUDIT.md steps 2-6 (all quick, local-safe), or:"
echo "  bash scripts/local/run_postproc_local.sh"
