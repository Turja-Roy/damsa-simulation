# Shared environment for all DAMSA batch jobs. Sourced, not executed.
#
# EDIT THESE to match the modules actually available on the node.
# Check with:  module avail geant4 ; module avail root
#
# ROOT: any version works. The storage layer is TTree, chosen precisely so
# root/6.26 and root/6.40 both read and write the same files (RNTuple would
# have needed >= 6.36). If a newer ROOT is available, prefer it — nothing here
# depends on the version.

module load geant4/10.7  2>/dev/null || echo "[env] WARNING: geant4 module not loaded"
module load root/6.26    2>/dev/null || echo "[env] WARNING: root module not loaded"
module load python/3.10  2>/dev/null || true

# Geant4 data/env. Adjust the path if the module already does this.
if [[ -f /opt/geant4/10.7/share/Geant4/geant4make/geant4env.sh ]]; then
    source /opt/geant4/10.7/share/Geant4/geant4make/geant4env.sh
fi

# One thread per allocated CPU. damsa.cpp reads SLURM_CPUS_PER_TASK itself;
# this keeps any OpenMP/BLAS underneath from oversubscribing on top of it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORKDIR" || exit 1

echo "[env] host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "[env] workdir=$WORKDIR"
echo "[env] root=$(root-config --version 2>/dev/null || echo MISSING)"
