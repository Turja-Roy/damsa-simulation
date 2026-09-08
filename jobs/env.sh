# Shared environment for all DAMSA batch jobs.
# Sourced by every jobs/*.sbatch. Safe to run standalone to test:  bash jobs/env.sh
#
# Override module names without editing this file:
#   DAMSA_GEANT4_MODULE=geant4/11.2 DAMSA_ROOT_MODULE=root/6.30 sbatch jobs/00_build.sbatch

# ── Bootstrap the module system ─────────────────────────────────────────────
# `module` is a shell FUNCTION installed by /etc/profile.d/modules.sh (Environment
# Modules) or Lmod's init script. A non-interactive shell -- `bash jobs/env.sh`,
# or a Slurm batch script, neither of which is a login shell -- does not inherit
# it, so `module load` fails with "command not found". Source the init explicitly.
if ! type module >/dev/null 2>&1; then
    for _init in /etc/profile.d/modules.sh \
                 /etc/profile.d/lmod.sh \
                 /etc/profile.d/z00_lmod.sh \
                 /usr/share/lmod/lmod/init/bash \
                 /usr/share/Modules/init/bash \
                 /opt/ohpc/admin/lmod/lmod/init/bash \
                 "${MODULESHOME:-/nonexistent}/init/bash"; do
        if [[ -r "$_init" ]]; then
            # shellcheck disable=SC1090
            source "$_init" && break
        fi
    done
fi
if type module >/dev/null 2>&1; then
    echo "[env] module system: available"
else
    echo "[env] module system: NOT FOUND (tried the usual init paths)"
    echo "[env]   if your login shell has \`module\`, find where it comes from:"
    echo "[env]     type module | head -3   ;   echo \$MODULESHOME"
    echo "[env]   then add that init path to the loop in jobs/env.sh"
fi

# ── Load the toolchain ──────────────────────────────────────────────────────
# Tries the named module, then progressively less specific names, and stops at
# the first that works. Errors are shown, not swallowed.
damsa_load() {
    local label=$1; shift
    local m
    for m in "$@"; do
        [[ -z "$m" ]] && continue
        if module load "$m" >/dev/null 2>&1; then
            echo "[env] $label: loaded '$m'"
            return 0
        fi
    done
    echo "[env] $label: no module loaded (tried: $*)"
    return 1
}

if type module >/dev/null 2>&1; then
    damsa_load geant4 "${DAMSA_GEANT4_MODULE:-}" geant4 geant4/11.3 geant4/11.2 geant4/10.7 || true
    damsa_load root   "${DAMSA_ROOT_MODULE:-}"   root   root/6.30   root/6.28   root/6.26   || true
    damsa_load python "${DAMSA_PYTHON_MODULE:-}" python python3 python/3.10               || true
    damsa_load cmake  "${DAMSA_CMAKE_MODULE:-}"  cmake                                     || true
fi

# Geant4 data files. The module usually does this; this covers sites where it
# does not. Set DAMSA_GEANT4_ENV to point at a specific geant4env.sh.
for _g4 in "${DAMSA_GEANT4_ENV:-}" \
           "$(geant4-config --prefix 2>/dev/null)/share/Geant4/geant4make/geant4env.sh" \
           /opt/geant4/10.7/share/Geant4/geant4make/geant4env.sh; do
    if [[ -n "$_g4" && -r "$_g4" ]]; then
        # shellcheck disable=SC1090
        source "$_g4" >/dev/null 2>&1 && echo "[env] geant4 data env: $_g4"
        break
    fi
done

# One thread per allocated CPU. damsa.cpp reads SLURM_CPUS_PER_TASK itself; this
# keeps any OpenMP/BLAS underneath from oversubscribing on top of it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORKDIR" || { echo "[env] cannot cd to $WORKDIR"; exit 1; }

# ── Report and gate ─────────────────────────────────────────────────────────
_root=$(root-config --version 2>/dev/null || echo MISSING)
_g4v=$(geant4-config --version 2>/dev/null || echo MISSING)
_cmake=$(cmake --version 2>/dev/null | head -1 || echo MISSING)

echo "[env] host=$(hostname)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "[env] workdir=$WORKDIR"
echo "[env] root=$_root  geant4=$_g4v"
echo "[env] $_cmake"

# Any ROOT version is fine: storage is TTree, which reads and writes identically
# from 6.26 through 6.40. (RNTuple would have required >= 6.36.)
_bad=0
[[ "$_root" == MISSING ]] && { echo "[env] ERROR: root-config not on PATH"; _bad=1; }

# Geant4 does not need geant4-config on PATH: CMake only needs Geant4Config.cmake,
# which -DGeant4_DIR can point at directly. Export Geant4_DIR (or DAMSA_GEANT4_DIR)
# and 00_build.sbatch will pass it through.
if [[ "$_g4v" == MISSING ]]; then
    export Geant4_DIR="${Geant4_DIR:-${DAMSA_GEANT4_DIR:-}}"
    if [[ -n "$Geant4_DIR" && -r "$Geant4_DIR/Geant4Config.cmake" ]]; then
        echo "[env] geant4-config not on PATH, using Geant4_DIR=$Geant4_DIR"
    else
        echo "[env] ERROR: geant4-config not on PATH and Geant4_DIR is not a"
        echo "[env]        directory containing Geant4Config.cmake"
        echo "[env]        find it with:  find / -name Geant4Config.cmake 2>/dev/null | head"
        _bad=1
    fi
fi

if (( _bad )); then
    cat <<'MSG'
[env]
[env] Find the right module names on this node and re-run:
[env]     module avail 2>&1 | less        # everything
[env]     module -t avail 2>&1 | grep -i -e root -e geant
[env]     module spider root              # Lmod sites
[env]
[env] Then either edit the damsa_load lines in jobs/env.sh, or pass the names in:
[env]     DAMSA_ROOT_MODULE=root/6.30 DAMSA_GEANT4_MODULE=geant4/11.2 \
[env]         sbatch jobs/00_build.sbatch
[env]
[env] If your OTHER working repo builds on this cluster, copy what it does:
[env]     grep -rn -e "module load" -e thisroot -e geant4env ~/<other-repo> ~/.bashrc
MSG
    # `return` when sourced, `exit` when executed, so testing this by hand
    # cannot kill an interactive shell.
    (return 0 2>/dev/null) && return 1 || exit 1
fi
