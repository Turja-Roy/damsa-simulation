#!/bin/bash
# AUDIT.md steps 2-6 — all quick, run locally after the merged
# canonical outputs exist (run_library_local.sh).  Consumes
# output/{alplib_brems_flux,pi0_decays,calo_face_particles}.csv.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

N="${N_ELECTRONS:-1000000}"   # keep in sync with NCHUNKS*EVENTS

for f in output/alplib_brems_flux.csv output/pi0_decays.csv output/calo_face_particles.csv; do
    [[ -s "$f" ]] || { echo "ERROR: missing $f — run run_library_local.sh first"; exit 1; }
done

# 2. signal pipeline (bugs 1.4, 1.5)
python3 scripts/pipeline/alp_signal_pipeline.py \
    --flux output/alplib_brems_flux.csv --auto-coupling

# 3. pi0-only accidental overlay per mode
for mode in dark lesa xleap interleaved; do
  python3 scripts/pipeline/pileup_overlay.py \
      --library output/pi0_decays.csv --n-library-electrons "$N" \
      --beam-mode "$mode" --gate-ns 1000 --mass-MeV 100 --mass-window-MeV 20
done

# 4. full SM accidental pool (calo-face)
for mode in dark lesa xleap interleaved; do
  python3 scripts/pipeline/pileup_overlay.py \
      --library output/calo_face_particles.csv --n-library-electrons "$N" \
      --beam-mode "$mode" --gate-ns 1000 --mass-MeV 100 --mass-window-MeV 20 \
      --calo-face --min-photon-E 5
done

# 5. re-inject decay photons (MT row bug 1.3 fixed)
./build/damsa_alp_inject output/alp_decay_photons_ma100MeV.csv macros/run_alp.mac

# 6. optional cross-check: pulsed direct sim vs overlay, dark only
./build/damsa macros/pulsed_dark.mac

echo "=== post-processing done: $(date) ==="
