#!/usr/bin/env bash
# Gather everything needed to configure jobs/env.sh for this cluster.
# Run on the LOGIN NODE (it is all cheap):   bash jobs/diagnose.sh 2>&1 | tee env_report.txt
#
# The point is to find out how YOUR shell gets its toolchain, since a batch job
# is not a login shell and inherits none of it automatically.

echo "########## 1. shell / module system ##########"
echo "SHELL=$SHELL"
echo "MODULESHOME=${MODULESHOME:-<unset>}"
echo "LMOD_CMD=${LMOD_CMD:-<unset>}"
echo "--- is 'module' defined in an interactive login shell? ---"
bash -lic 'type module 2>&1 | head -5' 2>/dev/null
echo "--- module init scripts present on this node ---"
for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
         /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash \
         /opt/ohpc/admin/lmod/lmod/init/bash "${MODULESHOME:-/nonexistent}/init/bash"; do
    [[ -r "$f" ]] && echo "  FOUND $f"
done

echo
echo "########## 2. available modules ##########"
bash -lic 'module -t avail' 2>&1 | grep -i -e root -e geant -e python -e cmake | sort -u | head -40
echo "--- (if that is empty, the full list) ---"
bash -lic 'module -t avail' 2>&1 | head -40

echo
echo "########## 3. toolchain already on PATH (login shell) ##########"
bash -lic 'command -v root-config    && root-config --version'    2>&1 | tail -2
bash -lic 'command -v geant4-config  && geant4-config --version'  2>&1 | tail -2
bash -lic 'command -v cmake          && cmake --version | head -1' 2>&1 | tail -2
bash -lic 'command -v python3        && python3 --version'        2>&1 | tail -2

echo
echo "########## 4. how does your OTHER repo build? ##########"
# The working setup is the best documentation. Adjust the paths if needed.
for d in ~/damsa-simulation ~/DAMSA* ~/damsa*; do
    [[ -d "$d" ]] || continue
    echo "--- $d ---"
    grep -rn --include='*.sh' --include='*.sbatch' --include='*.slurm' \
         -e 'module load' -e 'thisroot' -e 'geant4env' -e 'Geant4_DIR' "$d" 2>/dev/null | head -10
done
echo "--- your shell startup files ---"
grep -n -e 'module load' -e thisroot -e geant4env -e Geant4_DIR \
    ~/.bashrc ~/.bash_profile ~/.profile 2>/dev/null | head -20

echo
echo "########## 5. Geant4 / ROOT installs on disk ##########"
# NEVER search $HOME or /cvmfs here: home is NFS and /cvmfs is autofs, so a
# depth-6 walk can hang for an hour. Stay on local filesystems, bounded depth,
# with a hard timeout.
echo "--- what is under /opt ---"
timeout 20 ls /opt 2>/dev/null | head -20
echo "--- Geant4Config.cmake (this is what -DGeant4_DIR needs) ---"
timeout 60 find /opt /usr/local /usr/lib64 /usr/share -xdev -maxdepth 5 \
    -name Geant4Config.cmake 2>/dev/null | head -5
echo "--- thisroot.sh / ROOTConfig.cmake ---"
timeout 60 find /opt /usr/local /usr/lib64 /usr/share -xdev -maxdepth 5 \
    \( -name thisroot.sh -o -name ROOTConfig.cmake \) 2>/dev/null | head -5
echo "--- CVMFS repos (listing only, never descend) ---"
timeout 10 ls /cvmfs 2>/dev/null | head || echo "  (no /cvmfs, or autofs did not respond)"

echo
echo "########## 6. slurm ##########"
sinfo -s 2>&1 | head -10
echo "--- your account/partitions ---"
sacctmgr -n show assoc user="$USER" format=Account,Partition 2>&1 | head -10

echo
echo "########## done ##########"
echo "Send section 2, 3, 4 and 5 back and jobs/env.sh can be pinned to this node."
