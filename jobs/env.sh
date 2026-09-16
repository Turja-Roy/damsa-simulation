# Shared environment for all DAMSA batch jobs.
# Sourced by every jobs/*.sbatch. Safe to run standalone:  bash jobs/env.sh
#
# This cluster (UTA Tier3 ATLAS) has NO geant4/root/python modules -- `module
# avail` lists only mpi, lmod and settarg. Everything comes from a CVMFS LCG
# view, which is what build/CMakeCache.txt in the previous repo points at:
#     Geant4_DIR = /cvmfs/sft.cern.ch/lcg/views/LCG_110/.../lib64/cmake/Geant4
#     CLHEP_DIR  = /cvmfs/sft.cern.ch/lcg/views/LCG_110/.../lib/CLHEP-2.4.7.2
#
# Order matters, and getting it wrong fails in a confusing way:
#   1. LCG view      -- gcc13, Geant4, ROOT, CLHEP, tbb
#   2. miniforge     -- after LCG, with PYTHONPATH/PYTHONHOME cleared
#   3. project venv  -- last, so its numpy/scipy win over the LCG stack's
#   4. PYTHONPATH    -- repo root, so `import alplib` resolves
#
# Overrides: DAMSA_LCG_VIEW, DAMSA_VENV, DAMSA_CONDA.

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORKDIR" || { echo "[env] cannot cd to $WORKDIR"; exit 1; }

# ── 1. LCG view ─────────────────────────────────────────────────────────────
# REQUIRED; do not rely on the submitting shell. build/damsa* link against this
# view, and a job submitted from a shell that had not sourced it dies with
# "libtbb.so.12: cannot open shared object file" -- SLURM's default --export=ALL
# makes that failure intermittent, so it must be explicit here.
LCG_VIEW="${DAMSA_LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt}"
if [[ -r "$LCG_VIEW/setup.sh" ]]; then
    # shellcheck disable=SC1091
    source "$LCG_VIEW/setup.sh"
    echo "[env] LCG view: $LCG_VIEW"
else
    echo "[env] ERROR: no LCG view at $LCG_VIEW"
    echo "[env]   /cvmfs is autofs -- on the login node it may need a touch first:"
    echo "[env]     ls /cvmfs/sft.cern.ch/lcg/views/ | tail"
    echo "[env]   then set DAMSA_LCG_VIEW to the right one."
fi

# ── 2-3. Python ─────────────────────────────────────────────────────────────
# The LCG view puts its own Python on PATH; clear PYTHONPATH/PYTHONHOME before
# activating conda so the two stacks do not cross-contaminate.
unset PYTHONPATH PYTHONHOME
for _conda in "${DAMSA_CONDA:-}" "$HOME/miniforge/bin/activate" "$HOME/miniconda3/bin/activate"; do
    if [[ -n "$_conda" && -r "$_conda" ]]; then
        # shellcheck disable=SC1091
        source "$_conda" && echo "[env] conda: $_conda"
        break
    fi
done
for _venv in "${DAMSA_VENV:-}" "$WORKDIR/.venv" "$WORKDIR/../damsa-simulation/.venv" \
             "$HOME/damsa-simulation/.venv"; do
    if [[ -n "$_venv" && -r "$_venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$_venv/bin/activate" && echo "[env] venv: $_venv"
        break
    fi
done

# 4. alplib lives at the repo root, so the repo root must be importable.
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"

# alplib imports pkg_resources, which setuptools >= 81 removed.
if ! python3 -c "import pkg_resources" >/dev/null 2>&1; then
    echo "[env] installing setuptools<81 (alplib needs pkg_resources)"
    pip install --quiet "setuptools<81" || echo "[env] WARNING: that pip install failed"
fi

# One thread per allocated CPU. damsa.cpp reads SLURM_CPUS_PER_TASK itself; this
# keeps any OpenMP/BLAS underneath from oversubscribing on top of it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ── Report and gate ─────────────────────────────────────────────────────────
_root=$(root-config --version 2>/dev/null || echo MISSING)
_g4v=$(geant4-config --version 2>/dev/null || echo MISSING)
_py=$(python3 -c 'import numpy; print("numpy "+numpy.__version__)' 2>/dev/null || echo "numpy MISSING")

echo "[env] host=$(hostname)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "[env] workdir=$WORKDIR"
echo "[env] root=$_root  geant4=$_g4v"
echo "[env] $(cmake --version 2>/dev/null | head -1)  $(g++ --version 2>/dev/null | head -1)"
echo "[env] python=$(python3 --version 2>&1 | cut -d' ' -f2)  $_py  ($(command -v python3))"

# Any ROOT version works: storage is TTree, identical from 6.26 to 6.40.
_bad=0
[[ "$_root" == MISSING ]] && { echo "[env] ERROR: root-config not on PATH"; _bad=1; }
if [[ "$_g4v" == MISSING ]]; then
    export Geant4_DIR="${Geant4_DIR:-${DAMSA_GEANT4_DIR:-$LCG_VIEW/lib64/cmake/Geant4}}"
    if [[ -r "$Geant4_DIR/Geant4Config.cmake" ]]; then
        echo "[env] geant4-config not on PATH, using Geant4_DIR=$Geant4_DIR"
    else
        echo "[env] ERROR: no geant4-config and no Geant4Config.cmake at $Geant4_DIR"
        _bad=1
    fi
fi
[[ "$_py" == "numpy MISSING" ]] && echo "[env] WARNING: no numpy -- the Python half of 01_verify will not run"

if (( _bad )); then
    cat <<'MSG'
[env]
[env] The toolchain comes from CVMFS, not modules. Check the view exists:
[env]     ls /cvmfs/sft.cern.ch/lcg/views/ | tail
[env]     ls /cvmfs/sft.cern.ch/lcg/views/LCG_110/
[env] If /cvmfs is empty here, it is autofs and may only mount on compute nodes:
[env]     srun -p normal -n1 -t 2:00 bash -c 'ls /cvmfs/sft.cern.ch/lcg/views/ | tail'
MSG
    (return 0 2>/dev/null) && return 1 || exit 1
fi
