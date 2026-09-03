#!/usr/bin/env bash
# DAMSA C++ migration - cluster verification runbook (Phases 0-3).
#
# Not meant to be run end to end unattended: work through it stage by stage and
# check each verdict before moving on. Every stage prints a MATCH / IDENTICAL
# line, or tells you what differed.
#
# Set these two, then source this file or copy blocks out of it.
set -u
LOCAL=/My-Drive/Research/damsa-simulation
CLUSTER=user@cluster.example.edu           # <-- edit
REMOTE=~/damsa-cpp                         # new folder, does not touch the old one

# ─────────────────────────────────────────────────────────────────────────────
# 0. Upload
# ─────────────────────────────────────────────────────────────────────────────
# Source + the alplib reference oracle + the flux the pipeline needs. Excludes
# build products and the 7.6 GB of old output, which the cluster regenerates.
rsync -avz --progress \
  --exclude 'build*/' --exclude 'output/' --exclude 'output_old/' \
  --exclude 'plots/'  --exclude 'plots_old/' --exclude '.git/' \
  --exclude '.venv/'  --exclude '__pycache__/' --exclude 'Ref/' --exclude 'Reading/' \
  "$LOCAL"/ "$CLUSTER:$REMOTE"/

# The 20 per-chunk flux CSVs are the input to the merge cross-check (~3.4 MB).
rsync -avz --progress \
  "$LOCAL"/output/chunk*_alplib_brems_flux.csv "$CLUSTER:$REMOTE/output/"

# The pi0 library, for the pileup cross-check (~1 MB).
rsync -avz "$LOCAL"/output/pi0_decays.csv "$CLUSTER:$REMOTE/output/"

# Optional: skip re-running Geant4 by shipping the merged flux directly (175 KB).
rsync -avz "$LOCAL"/output/alplib_brems_flux.csv "$CLUSTER:$REMOTE/output/"

# ── everything below runs ON the cluster ─────────────────────────────────────
ssh "$CLUSTER"   # then: cd ~/damsa-cpp

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build
# ─────────────────────────────────────────────────────────────────────────────
# Needs Geant4 (ui_all vis_all multithread) and ROOT >= 6.36 for RNTuple.
# module load geant4 root        # whatever your site uses
cmake -S . -B build -DGEANT4_BUILD_MULTITHREADED=ON
cmake --build build -j"$(nproc)"
# If cmake cannot find Geant4, pass it explicitly:
#   cmake -S . -B build -DGeant4_DIR=/path/to/lib/cmake/Geant4
# Expect 15 targets. Sanity:
root-config --version          # must be >= 6.36 for the RNTuple API used here

# ─────────────────────────────────────────────────────────────────────────────
# 2. Unit checks (seconds, no data needed)
# ─────────────────────────────────────────────────────────────────────────────
./build/damsa_io_test          # RNTuple round-trip is exact
./build/alp_test alplib        # physics identities: E-p conservation, m_gg == ma

# Phase 2: the C++ physics against alplib itself.
./build/alp_xcheck alplib  > cpp.txt
python3 tools/alp_xcheck.py > py.txt
python3 tools/alp_xcheck_diff.py cpp.txt py.txt    # expect "MATCH", exit 0

# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase 1 - storage. Short Geant4 run, both formats, then diff the pair.
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p verify && cd verify
printf '/run/initialize\n/run/beamOn 2000\n' > short.mac
../build/damsa short.mac > sim.log 2>&1

# Re-emit each RNTuple as CSV and compare to the CSV from the SAME run.
# These must be byte-identical; the RNTuple is the more precise of the two.
for p in "target_exit.root:all_particles_target_exit.csv" \
         "calo_face_particles.root:calo_face_particles.csv" \
         "pi0_decays.root:pi0_decays.csv"; do
  r=${p%%:*}; c=${p##*:}
  ../build/damsa_convert "output/$r" "check_$c" >/dev/null
  diff -q "output/$c" "check_$c" && echo "OK  $r == $c" || echo "FAIL $r != $c"
done
du -sh output/          # size win vs the CSVs
cd ..

# ─────────────────────────────────────────────────────────────────────────────
# 4. Phase 3 - merge. Byte-identical except the provenance line.
# ─────────────────────────────────────────────────────────────────────────────
# Work on a copy: the Python writes output/alplib_brems_flux.csv in place.
mkdir -p mergecheck/output && cp output/chunk*_alplib_brems_flux.csv mergecheck/output/
( cd mergecheck && ../build/damsa_merge --nchunks 20 --outdir output \
    && mv output/alplib_brems_flux.csv cpp_flux.csv )
python3 scripts/local/merge_library.py --nchunks 20
diff mergecheck/cpp_flux.csv output/alplib_brems_flux.csv
# Expect exactly one hunk: "MERGED from 20 chunks by damsa_merge" vs
# "... by merge_library.py". Any difference in a DATA row is a real failure.

# ─────────────────────────────────────────────────────────────────────────────
# 5. Phase 3 - ALP signal. Deterministic columns must match; sampled ones
#    are compared against the spread over seeds.
# ─────────────────────────────────────────────────────────────────────────────
./build/damsa_alp_signal --flux output/alplib_brems_flux.csv --alplib alplib \
    --auto-coupling --angle-samples 20 --no-4vec --no-sensitivity --xcheck \
    | grep -E '^[0-9]' > sig_cpp.txt
python3 tools/alp_signal_xcheck.py --flux output/alplib_brems_flux.csv \
    --angle-samples 20 | grep -E '^[0-9]' > sig_py.txt
paste sig_cpp.txt sig_py.txt | awk '{printf "ma=%-6s g: %s vs %s\n", $1, $2, $8}'
# coupling (col 2) must be identical digit for digit.
# theta_kin (col 3) and total_weight (col 4) agree to ~1e-9 / ~2e-8; that gap is
#   alplib storing its weights as float32, not a porting error.
# theta_MC (col 5) and accept (col 6) are sampled - expect a few percent.

# MC spread, to confirm the sampled columns are consistent rather than biased:
for sd in 1 2 3 4 5 6 7 8; do
  ./build/damsa_alp_signal --flux output/alplib_brems_flux.csv --alplib alplib \
    --auto-coupling --angle-samples 20 --seed $sd --no-4vec --no-sensitivity \
    --xcheck | grep -E '^[0-9]' | awk 'NF==6{print $1,$5,$6}'
done > seeds.txt
# The Python values should sit within ~3 sd of the per-mass mean of seeds.txt.

# ─────────────────────────────────────────────────────────────────────────────
# 6. Phase 3 - pileup. Library parse is deterministic and must match exactly.
# ─────────────────────────────────────────────────────────────────────────────
./build/damsa_convert output/pi0_decays.csv output/pi0_decays.root
for sd in 1 2 3; do
  ./build/damsa_pileup --library output/pi0_decays.root \
     --n-library-electrons 1000000 --use-geom-accept --beam-mode lesa \
     --n-trials 20000 --seed $sd --out-csv pu_cpp_$sd.csv | grep -E 'P\(has\)|R_acc'
  python3 scripts/pipeline/pileup_overlay.py --library output/pi0_decays.csv \
     --n-library-electrons 1000000 --use-geom-accept --beam-mode lesa \
     --n-trials 20000 --seed $sd | grep -iE 'calo photons total|R_acc'
done
# "148 photon-bearing electrons, 176 photons" must match exactly on both sides.
# R_acc is sampled: agreement at the sub-percent level is expected.
#
# NOTE: without --use-geom-accept both sides give ZERO. In the 1e6-electron
# library no pi0 gamma survives the material to the calo face (AtCalo = 0 for
# all 5532 decays). That is physics, not a failure.

# ─────────────────────────────────────────────────────────────────────────────
# 7. Production runs (only after the checks above pass)
# ─────────────────────────────────────────────────────────────────────────────
# 7a. Full library, 20 x 50k electrons as separate processes. THE LONG ONE.
bash scripts/local/run_library_local.sh
./build/damsa_merge --nchunks 20

# 7b. Full ALP signal: 8 masses x 4M decay pairs. ~80 s, ~1.9 GB of RNTuple
#     (the CSVs it replaces were 7.9 GB). The sensitivity scan adds ~4 min.
./build/damsa_alp_signal --flux output/alplib_brems_flux.csv --alplib alplib \
    --auto-coupling --outdir output --decay-samples 500

# 7c. Accidentals, per beam mode, on the full calo-face pool.
for m in dark lesa xleap interleaved; do
  ./build/damsa_pileup --library output/calo_face_particles.root --calo-face \
     --n-library-electrons 1000000 --beam-mode $m --n-trials 100000 \
     --out-csv output/pileup_$m.csv
done

# 7d. Re-injection through Geant4 still reads CSV, so convert one mass first.
#     (alp_generator.h keeps its CSV path; RNTuple support there is Phase 4.)
./build/damsa_convert output/alp_decay_photons_ma100MeV.root \
                      output/alp_decay_photons_ma100MeV.csv
./build/damsa_alp_inject output/alp_decay_photons_ma100MeV.csv

# ─────────────────────────────────────────────────────────────────────────────
# 8. Migrating the existing 1 GB CSVs (optional, only if you want them kept)
# ─────────────────────────────────────────────────────────────────────────────
for f in output/alp_decay_photons_ma*MeV.csv; do
  ./build/damsa_convert "$f" "${f%.csv}.root"
done
du -sh output/          # ~7.9 GB -> ~1.9 GB

# ─────────────────────────────────────────────────────────────────────────────
# 9. Pull results back
# ─────────────────────────────────────────────────────────────────────────────
# From your laptop. Exclude the multi-hundred-MB decay files unless you need them.
# rsync -avz --progress --exclude 'alp_decay_photons_*' \
#   "$CLUSTER:$REMOTE/output/" "$LOCAL/output_cluster/"
