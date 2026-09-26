#!/usr/bin/env python3
"""
Pileup overlay / shower-library tool for DAMSA — Level B Strategy 2 (plan.md §3.3, §3.5).

The dense LESA beam modes (LESA laser, X-LEAP, interleaved) put 10^4–10^8
electrons in a single calorimeter readout gate. Firing them all through Geant4
per event is intractable (the generator guard `GateTooDense` blocks it). This
tool instead OVERLAYS a single-electron shower library offline:

  1. Library  : one non-pulsed Geant4 run (1 electron / event) → output/pi0_decays.csv.
                Each eventID = one electron's worth of pi0 -> gamma gamma decays,
                with both photon 4-vectors and "reached calo" truth flags.
                The library is per-electron physics, so ONE library serves ALL
                beam modes — the mode only changes how many electrons per gate.

  2. Overlay  : for each synthetic readout gate, sample the gate occupancy for
                the chosen beam mode (same logic as gate_sampler.h), pool the
                calo-reaching photons from that many drawn library electrons,
                and count ACCIDENTAL coincidences: two photons from DIFFERENT
                pi0 decays whose pair invariant mass falls in an ALP-like window
                and whose summed energy passes the calo threshold.

Accidental rate (plan.md §3.6):
    R_acc[Hz] = (gates_with_a_fake / N_trials) x kicker_rate_Hz

This is a TRUTH-LEVEL estimate (photon directions/energies, no shower
clustering or energy smearing). It is the right first-order observable for the
accidental background; a full per-cell calo overlay would refine the energy
resolution but needs cell-level scoring that does not exist yet.

Two library modes:
  default      : pi0_decays.csv — only pi0->gg daughters are fake candidates.
  --calo-face  : calo_face_particles.csv — EVERY photon crossing the calo
                 entrance (brems, shower, pi0) is a candidate; this is the full
                 SM accidental pool and gives the physically complete rate.

Usage:
    # build the library first (any non-pulsed run; mode is irrelevant here):
    ./build/damsa macros/run.mac            # -> output/pi0_decays.csv, 100k electrons

    # then overlay for a given mode:
    python scripts/pipeline/pileup_overlay.py \
        --library output/pi0_decays.csv --n-library-electrons 100000 \
        --beam-mode lesa --gate-ns 1000 --mass-MeV 100 --mass-window-MeV 20
"""

import sys
import argparse
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Beam modes — MIRROR of DamsaConfig::BeamSpecFor in src/config/damsa_config.h.
# Keep in sync by hand (small table, changes rarely).
#   (bunch_charge_e, kicker_rate_Hz, bunches_per_kick, bunch_spacing_s)
# ──────────────────────────────────────────────────────────────────────────────
BEAM_MODES = {
    "dark":        (0.07,    929e3, 100, 5.4e-9),
    "lesa":        (4200.0,  929e3, 18,  26.9e-9),
    "xleap":       (167000.0, 929e3, 1,  1.08e-6),
    "interleaved": (6.2e8,   100.0, 1,   10e-3),
}


def sample_gate_occupancy(spec, gate_s, poisson, rng, size):
    """Vectorized SampleGateOccupancy (gate_sampler.h) for `size` gates at once.

    n_bunches = min(floor(gate/spacing), bunches_per_kick), >= 1.
    occupancy = sum over bunches of Poisson(bunch_charge)  [or fixed round].
    Returns an int array of electrons-per-gate, shape (size,).
    """
    q, rate, nkick, spacing = spec
    nb = int(np.floor(gate_s / spacing))
    nb = max(1, min(nb, nkick))
    if poisson:
        # Sum of nb iid Poisson(q) = Poisson(nb*q). One draw per gate.
        return rng.poisson(nb * q, size=size)
    return np.full(size, int(round(nb * q)) * 1, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Library loading
# ──────────────────────────────────────────────────────────────────────────────

def load_library(csv_path, use_geom_accept=False):
    """Parse pi0_decays.csv into per-electron lists of calo-reaching photons.

    Each library photon = (E_MeV, ux, uy, uz, parent_tag) where parent_tag
    uniquely identifies the parent pi0 (eventID, pi0TrackID) so accidental pairs
    (different parents) can be distinguished from a single pi0's own gamma gamma.

    Returns:
        photon_sets : list of np.ndarray, one per electron that produced >=1
                      calo photon. Each array has columns [E, ux, uy, uz, tag].
        n_calo_photons : total calo photons in the library.
    """
    # Columns (see pi0DecayData.h WriteCSV):
    #  0 eventID 1 pi0TrackID 2-4 vtx 5 g1TID 6 e1 7-9 p1 10 g2TID 11 e2 12-14 p2
    #  15 openingAngle 16 pi0E 17 g1AtCalo 18 g2AtCalo 19 g1Geom 20 g2Geom 21 caloE
    acc1_col, acc2_col = (19, 20) if use_geom_accept else (17, 18)

    per_electron = {}        # eventID -> list of [E, ux, uy, uz, tag]
    tag_counter = 0
    n_calo_photons = 0

    with open(csv_path) as f:
        header = f.readline()  # skip
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = line.split(',')
            if len(c) < 21:
                continue
            try:
                evt = int(float(c[0]))
                pi0 = int(float(c[1]))
                tag = hash((evt, pi0)) & 0x7FFFFFFF
                bucket = per_electron.setdefault(evt, [])
                # gamma 1
                if int(float(c[acc1_col])) == 1:
                    E = float(c[6]); ux, uy, uz = float(c[7]), float(c[8]), float(c[9])
                    bucket.append([E, ux, uy, uz, tag]); n_calo_photons += 1
                # gamma 2
                if int(float(c[acc2_col])) == 1:
                    E = float(c[11]); ux, uy, uz = float(c[12]), float(c[13]), float(c[14])
                    bucket.append([E, ux, uy, uz, tag]); n_calo_photons += 1
            except (ValueError, IndexError):
                continue

    photon_sets = [np.array(v, dtype=float) for v in per_electron.values() if v]
    return photon_sets, n_calo_photons


def load_calo_face_library(csv_path, min_photon_E):
    """Parse calo_face_particles.csv into per-electron photon lists.

    Unlike the pi0 library, this includes EVERY photon that physically crossed
    the calo entrance plane (bremsstrahlung, shower photons, pi0 daughters, …),
    which is the full SM fake-candidate pool. All photons of one electron share
    one parent tag (its eventID): pairs within a single electron's shower are
    correlated single-electron background, not accidentals, and are counted
    only across different draws.

    Columns (FluxData.h WriteCaloFaceCSV):
      0 pdg 1 energy_MeV 2 time_ns 3-5 x,y,z 6-8 px,py,pz 9 weight 10 trackID 11 eventID
    """
    per_electron = {}
    n_photons = 0

    with open(csv_path) as f:
        f.readline()  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = line.split(',')
            if len(c) < 12:
                continue
            try:
                if int(float(c[0])) != 22:
                    continue
                E = float(c[1])
                if E < min_photon_E:
                    continue
                evt = int(float(c[11]))
                per_electron.setdefault(evt, []).append(
                    [E, float(c[6]), float(c[7]), float(c[8]), float(evt)])
                n_photons += 1
            except (ValueError, IndexError):
                continue

    photon_sets = [np.array(v, dtype=float) for v in per_electron.values() if v]
    return photon_sets, n_photons


# ──────────────────────────────────────────────────────────────────────────────
# Accidental pair counting within one gate
# ──────────────────────────────────────────────────────────────────────────────

def count_fakes_in_gate(photons, mass_lo, mass_hi, esum_thr, max_pairs, rng):
    """Count accidental (different-source) gamma-gamma pairs passing the ALP window.

    photons : (K, 6) array [E, ux, uy, uz, tag, draw] — directions unit norm.
    `draw` is the index of the library draw the photon came from: the library is
    sampled WITH replacement, so the same library electron drawn twice
    represents two independent real electrons — its photons must pair as
    accidentals even though their tags collide. Same-source = same tag AND same
    draw (one real pi0's gamma-gamma, or one electron's own shower in calo-face
    mode).
    Returns the (possibly subsample-scaled) number of fake pairs in this gate.
    """
    K = len(photons)
    if K < 2:
        return 0.0

    E = photons[:, 0]
    u = photons[:, 1:4]
    tag = photons[:, 4]
    draw = photons[:, 5]

    # All unordered pairs; subsample if too many to keep runtime bounded.
    i_idx, j_idx = np.triu_indices(K, k=1)
    n_all = len(i_idx)
    scale = 1.0
    if n_all > max_pairs:
        sel = rng.choice(n_all, size=max_pairs, replace=False)
        i_idx, j_idx = i_idx[sel], j_idx[sel]
        scale = n_all / max_pairs

    # Reject same-source pairs (not accidental).
    diff_parent = (tag[i_idx] != tag[j_idx]) | (draw[i_idx] != draw[j_idx])
    i_idx, j_idx = i_idx[diff_parent], j_idx[diff_parent]
    if len(i_idx) == 0:
        return 0.0

    E1, E2 = E[i_idx], E[j_idx]
    cos12 = np.sum(u[i_idx] * u[j_idx], axis=1)
    cos12 = np.clip(cos12, -1.0, 1.0)
    minv = np.sqrt(np.maximum(2.0 * E1 * E2 * (1.0 - cos12), 0.0))
    esum = E1 + E2

    passed = (minv >= mass_lo) & (minv <= mass_hi) & (esum >= esum_thr)
    return float(np.count_nonzero(passed)) * scale


# ──────────────────────────────────────────────────────────────────────────────
# Main overlay loop
# ──────────────────────────────────────────────────────────────────────────────

def run_overlay(photon_sets, n_lib_electrons, spec, args, rng):
    kicker_rate = spec[1]
    p_has = len(photon_sets) / float(n_lib_electrons)   # P(electron yields calo photon)

    occ = sample_gate_occupancy(spec, args.gate_ns * 1e-9, not args.fixed,
                                rng, args.n_trials)

    # Number of photon-bearing electrons per gate ~ Binomial(occupancy, p_has).
    # (Most electrons make no calo photon; only these contribute.)
    occ_clip = np.minimum(occ, np.iinfo(np.int64).max)
    n_bearers = rng.binomial(occ_clip, p_has)

    n_sets = len(photon_sets)
    total_fakes = 0.0
    gates_with_fake = 0
    max_K_seen = 0

    warned = False
    for g in range(args.n_trials):
        m = int(n_bearers[g])
        if m < 1:
            continue
        pick = rng.integers(0, n_sets, size=m)
        # Append a draw-index column: repeated draws of the same library
        # electron are independent real electrons, distinguished by draw index.
        photons = np.vstack([
            np.column_stack([photon_sets[p],
                             np.full(len(photon_sets[p]), k, dtype=float)])
            for k, p in enumerate(pick)
        ])
        K = len(photons)
        max_K_seen = max(max_K_seen, K)
        if K > args.max_photons_per_gate:
            if not warned:
                print(f"[warn] gate has {K} calo photons (> --max-photons-per-gate="
                      f"{args.max_photons_per_gate}). This mode is in the saturated-"
                      f"pileup regime where the accidental-pair model breaks down; "
                      f"treat results as a lower bound.", file=sys.stderr)
                warned = True
            # subsample the photon pool to keep it tractable
            sub = rng.choice(K, size=args.max_photons_per_gate, replace=False)
            photons = photons[sub]

        fakes = count_fakes_in_gate(photons, args.mass_MeV - args.mass_window_MeV,
                                    args.mass_MeV + args.mass_window_MeV,
                                    args.threshold_MeV, args.max_pairs, rng)
        if fakes > 0:
            total_fakes += fakes
            gates_with_fake += 1

    frac_gate = gates_with_fake / args.n_trials
    r_acc = frac_gate * kicker_rate
    fake_pair_rate = (total_fakes / args.n_trials) * kicker_rate

    return {
        "mean_occupancy": float(np.mean(occ)),
        "mean_bearers": float(np.mean(n_bearers)),
        "max_calo_photons_gate": max_K_seen,
        "p_has": p_has,
        "gates_with_fake": gates_with_fake,
        "frac_gates_with_fake": frac_gate,
        "total_fakes": total_fakes,
        "R_acc_Hz": r_acc,
        "fake_pair_rate_Hz": fake_pair_rate,
        "kicker_rate_Hz": kicker_rate,
    }


def main():
    ap = argparse.ArgumentParser(description="DAMSA pileup overlay / accidental coincidence tool")
    ap.add_argument("--library", required=True,
                    help="pi0_decays.csv (default) or calo_face_particles.csv "
                         "(with --calo-face) from a non-pulsed run")
    ap.add_argument("--n-library-electrons", type=int, required=True,
                    help="electrons fired to build the library (= beamOn of that run)")
    ap.add_argument("--calo-face", action="store_true",
                    help="library is calo_face_particles.csv: use ALL photons at "
                         "the calo face (full SM fake pool), not only pi0 daughters")
    ap.add_argument("--min-photon-E", type=float, default=1.0,
                    help="min single-photon energy [MeV] for calo-face library "
                         "(default 1.0)")
    ap.add_argument("--beam-mode", choices=list(BEAM_MODES), default="dark")
    ap.add_argument("--gate-ns", type=float, default=1000.0, help="readout gate [ns] (default 1us)")
    ap.add_argument("--n-trials", type=int, default=100000, help="synthetic gates to simulate")
    ap.add_argument("--mass-MeV", type=float, default=100.0, help="signal ALP mass window centre")
    ap.add_argument("--mass-window-MeV", type=float, default=20.0, help="+/- mass window")
    ap.add_argument("--threshold-MeV", type=float, default=5.0, help="min summed pair energy")
    ap.add_argument("--use-geom-accept", action="store_true",
                    help="use geometric-acceptance flag instead of physical AtCalo")
    ap.add_argument("--fixed", action="store_true", help="fixed occupancy instead of Poisson")
    ap.add_argument("--max-pairs", type=int, default=200000,
                    help="cap on pairs evaluated per gate (subsample+scale above this)")
    ap.add_argument("--max-photons-per-gate", type=int, default=20000,
                    help="cap on calo photons pooled per gate (saturated regime guard)")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    spec = BEAM_MODES[args.beam_mode]

    print(f"[lib] loading {args.library} ...")
    if args.calo_face:
        photon_sets, n_calo = load_calo_face_library(args.library, args.min_photon_E)
    else:
        photon_sets, n_calo = load_library(args.library, args.use_geom_accept)
    if not photon_sets:
        sys.exit("[lib] no calo-reaching photons in library — run a bigger non-pulsed "
                 "sim, or pass --use-geom-accept for a looser acceptance flag.")
    print(f"[lib] {len(photon_sets)} electrons with calo photons / "
          f"{args.n_library_electrons} fired; {n_calo} calo photons total")
    print(f"[lib] mean calo photons per electron: {n_calo/args.n_library_electrons:.3e}")

    print(f"\n[overlay] mode={args.beam_mode} gate={args.gate_ns} ns "
          f"trials={args.n_trials} mass={args.mass_MeV}+/-{args.mass_window_MeV} MeV")
    r = run_overlay(photon_sets, args.n_library_electrons, spec, args, rng)

    print("\n=== Accidental coincidence result ===")
    print(f"  mean electrons / gate      : {r['mean_occupancy']:.3e}")
    print(f"  mean photon-bearing e/gate : {r['mean_bearers']:.3e}")
    print(f"  max calo photons in a gate : {r['max_calo_photons_gate']}")
    print(f"  gates with >=1 fake        : {r['gates_with_fake']} / {args.n_trials} "
          f"({r['frac_gates_with_fake']:.3e})")
    print(f"  kicker (gate) rate         : {r['kicker_rate_Hz']:.3e} Hz")
    print(f"  R_acc (fake-bearing gates) : {r['R_acc_Hz']:.3e} Hz")
    print(f"  fake-pair rate             : {r['fake_pair_rate_Hz']:.3e} Hz")
    print("=====================================")


if __name__ == "__main__":
    main()
