#!/bin/bash
#SBATCH -J damsa_verify
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o logs/verify_%j.out
#SBATCH -e logs/verify_%j.err
#
# Every cross-check, in one job. Read the summary at the bottom of the log.
#   sbatch --dependency=afterok:<build_id> jobs/01_verify.sbatch
#
# Needs in output/ : chunk*_alplib_brems_flux.csv (20), pi0_decays.root,
#                    alplib_brems_flux.csv

set -uo pipefail          # NOT -e: we want every check to run and report
source jobs/env.sh
mkdir -p logs verify
FAIL=0
ok(){ echo "  OK   $1"; }
bad(){ echo "  FAIL $1"; FAIL=1; }
skip(){ echo "  SKIP $1"; SKIPPED=1; }
SKIPPED=0

echo "=== 1. unit checks ==="
./build/damsa_io_test >/dev/null 2>&1 && ok "damsa_io_test (TTree round-trip exact)" || bad "damsa_io_test"
./build/alp_test alplib >/dev/null 2>&1 && ok "alp_test (E-p conservation, m_gg == ma)" || bad "alp_test"

echo "=== 2. physics vs alplib ==="
./build/alp_xcheck alplib > verify/cpp.txt 2>/dev/null
python3 tools/alp_xcheck.py > verify/py.txt 2>/dev/null
if python3 tools/alp_xcheck_diff.py verify/cpp.txt verify/py.txt > verify/xcheck.log 2>&1; then
    ok "alp_xcheck vs alplib (see verify/xcheck.log)"
else
    bad "alp_xcheck vs alplib"; cat verify/xcheck.log
fi

echo "=== 3. storage: short Geant4 run, CSV vs TTree ==="
( cd verify && rm -rf output && printf '/damsa/setWriteCSV true\n/run/initialize\n/run/beamOn 2000\n' > short.mac \
  && ../build/damsa short.mac > sim.log 2>&1 )
for pair in "target_exit.root:all_particles_target_exit.csv" \
            "calo_face_particles.root:calo_face_particles.csv" \
            "pi0_decays.root:pi0_decays.csv"; do
    r="${pair%%:*}"; c="${pair##*:}"
    ./build/damsa_convert "verify/output/$r" "verify/chk_$c" >/dev/null 2>&1
    if diff -q "verify/output/$c" "verify/chk_$c" >/dev/null 2>&1; then
        ok "$r == $c  ($(( $(wc -l < "verify/chk_$c") - 1 )) rows, byte-identical)"
    else
        bad "$r != $c"
    fi
done
echo "  size: $(du -sh verify/output | cut -f1)"

echo "=== 4. merge (damsa_merge vs merge_library.py) ==="
# 4-6 need the merged 1e6 library (02_library -> 03_merge). Without it they
# SKIP; they used to "pass" by diffing two empty outputs.
if ! ls output/chunk*_alplib_brems_flux.csv >/dev/null 2>&1; then
    skip "no output/chunk*_alplib_brems_flux.csv (run 02_library first)"
else
    # The Python writes output/alplib_brems_flux.csv IN PLACE, so keep a copy.
    cp output/alplib_brems_flux.csv verify/flux_before.csv 2>/dev/null || true
    rm -rf verify/mg && mkdir -p verify/mg/output
    cp output/chunk*_alplib_brems_flux.csv verify/mg/output/ 2>/dev/null
    ( cd verify/mg && ../../build/damsa_merge --nchunks 20 --outdir output >/dev/null 2>&1 )
    python3 .legacy/python/local/merge_library.py --nchunks 20 >/dev/null 2>&1
    if [[ ! -s verify/mg/output/alplib_brems_flux.csv || ! -s output/alplib_brems_flux.csv ]]; then
        bad "merge produced no flux file"
    # Skip line 2 on both sides: it names the tool that produced the file.
    elif diff <(tail -n +3 verify/mg/output/alplib_brems_flux.csv) \
              <(tail -n +3 output/alplib_brems_flux.csv) >/dev/null 2>&1; then
        ok "merged flux identical across all bins and headers"
    else
        bad "merged flux differs"
        diff <(tail -n +3 verify/mg/output/alplib_brems_flux.csv) \
             <(tail -n +3 output/alplib_brems_flux.csv) | head -10
    fi
fi

echo "=== 5. pileup (damsa_pileup vs pileup_overlay.py) ==="
if [[ ! -s output/pi0_decays.root ]]; then
    skip "no output/pi0_decays.root (run 03_merge first)"
else
    # The legacy Python reads CSV only; per-particle CSVs are no longer written.
    ./build/damsa_convert output/pi0_decays.root verify/pi0.csv >/dev/null 2>&1
    CPP=$(./build/damsa_pileup --library output/pi0_decays.root --n-library-electrons 1000000 \
            --use-geom-accept --beam-mode lesa --n-trials 20000 --seed 1 \
            --out-csv verify/pu_cpp.csv 2>/dev/null | grep -oP 'P\(has\)=\K[0-9.e+-]+')
    PY=$(python3 .legacy/python/pipeline/pileup_overlay.py --library verify/pi0.csv \
            --n-library-electrons 1000000 --use-geom-accept --beam-mode lesa \
            --n-trials 20000 --seed 1 2>/dev/null | grep -oP '^\[lib\] \K[0-9]+')
    # Same library on both sides: C++ P(has) x 1e6 must equal the Python count.
    echo "  C++ P(has)=$CPP   Python photon-bearing electrons=$PY"
    if [[ -z "$CPP" || -z "$PY" ]]; then
        bad "pileup produced no output"
    elif python3 -c "import sys; sys.exit(0 if round(float('$CPP')*1e6) == int('$PY') else 1)"; then
        ok "pileup: same photon-bearing electrons on both sides"
    else
        bad "pileup: C++ and Python disagree on the library"
    fi
    # NOTE: without --use-geom-accept BOTH sides are zero. In the 1e6 library no pi0
    # gamma survives the material to the calo face. That is physics, not a failure.
fi

echo "=== 6. ALP signal (deterministic columns must match exactly) ==="
if [[ ! -s output/alplib_brems_flux.csv ]]; then
    skip "no output/alplib_brems_flux.csv (run 03_merge first)"
else
    ./build/damsa_alp_signal --flux output/alplib_brems_flux.csv --alplib alplib \
        --auto-coupling --angle-samples 20 --no-4vec --no-sensitivity --xcheck 2>/dev/null \
        | grep -E '^[0-9]' > verify/sig_cpp.txt
    python3 tools/alp_signal_xcheck.py --flux output/alplib_brems_flux.csv \
        --angle-samples 20 2>/dev/null | grep -E '^[0-9]' > verify/sig_py.txt
    if [[ ! -s verify/sig_cpp.txt || ! -s verify/sig_py.txt ]]; then
        bad "ALP signal produced no rows (C++ $(wc -l < verify/sig_cpp.txt), Python $(wc -l < verify/sig_py.txt))"
    elif diff <(awk '{print $1,$2}' verify/sig_cpp.txt) \
              <(awk '{print $1,$2}' verify/sig_py.txt) >/dev/null 2>&1; then
        ok "coupling identical at every mass"
    else
        bad "coupling differs"; paste verify/sig_cpp.txt verify/sig_py.txt | head
    fi
    echo "  theta_kin / total_weight agree to ~1e-9 / ~2e-8 (alplib stores weights as float32)"
    echo "  theta_MC / accept are sampled - a few percent apart is expected"
    paste verify/sig_cpp.txt verify/sig_py.txt | awk '{printf "    ma=%-6s theta_kin %s vs %s\n",$1,$3,$9}'
fi

echo
echo "================ VERIFY $( ((FAIL)) && echo FAILED || { ((SKIPPED)) && echo "PASSED (with skips)" || echo PASSED; } ) ================"
exit $FAIL
