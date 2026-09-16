#!/usr/bin/env python3
"""
ALP Signal Pipeline for DAMSA — Production-quality, no proxy masses.

Computes Primakoff ALP production and decay using either:
  (A) Geant4 bremsstrahlung flux from output/alplib_brems_flux.csv [preferred]
  (B) Analytic Bethe-Heitler spectrum [fallback / fast mode]

Outputs:
  - Opening angle distributions per mass point
  - Sensitivity curve (excluded coupling vs mass)
  - Decay photon 4-vector CSVs for Geant4 re-injection (alp_generator.h)

Usage:
    python alp_signal_pipeline.py --flux output/alplib_brems_flux.csv --nprimaries 10000
    python alp_signal_pipeline.py --analytic --mass 100 --coupling 1e-3  # no Geant4 run needed
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from alplib.fluxes import FluxPrimakoffIsotropic
    from alplib.materials import Material
    from alplib.generators import PhotonEventGenerator
    from alplib.constants import CHARGE_COULOMBS, S_PER_DAY, METER_BY_MEV
except ImportError:
    sys.exit("alplib not found. Clone https://github.com/athompson-git/alplib into the project root.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT = True
except ImportError:
    PLOT = False


# ──────────────────────────────────────────────────────────────────────────────
# Geometry — update these to match your run's optimal configuration
# ──────────────────────────────────────────────────────────────────────────────
BEAM_ENERGY_MEV   = 8000.0    # 8 GeV LCLS-II electron beam
# Delivered LESA-Laser current (plan.md §1: 4200 e/bunch × 18 bunches × 929 kHz
# ≈ 11.2 nA). Used ONLY by the analytic Bethe-Heitler fallback; the Geant4 flux
# path reads its normalization from the CSV header written by FluxData.h.
BEAM_CURRENT_A    = 4200 * 18 * 929e3 * CHARGE_COULOMBS
ELECTRONS_PER_S   = BEAM_CURRENT_A / CHARGE_COULOMBS

TARGET_LENGTH_M   = 0.10      # 10 cm tungsten dump
TARGET_HALF_M     = TARGET_LENGTH_M / 2.0   # 5 cm — reference for decay vertex (target centre)
VDC_M             = 0.30      # vacuum decay chamber length [m]  (user-tunable, scan range 0.30–0.60 m)
MAGNET_M          = 0.12      # magnet + tracker region length [m]  (fixed hardware)
# DET_DIST_M: target centre → calo face = half-target + VDC + magnet.
# Used as the transverse-projection plane (photons must land on the calo face).
DET_DIST_M        = TARGET_HALF_M + VDC_M + MAGNET_M
# ALP decay window, measured from the production point (target centre):
# decays before the target exit die in tungsten (photons absorbed, cf. the
# pi0 study); decays after the calo face are invisible. alplib's propagate()
# uses surv over det_dist then decay within det_length, so:
#   det_dist   = DECAY_ZMIN_M  (target centre → target exit)
#   det_length = DECAY_ZMAX_M − DECAY_ZMIN_M  (target exit → calo face)
# (Pre-2026-07-10 this was det_dist=0.47/det_length=0.44 — decay modelled
#  INSIDE the CsI volume; wrong fiducial region for DAMSA.)
DECAY_ZMIN_M      = TARGET_HALF_M
DECAY_ZMAX_M      = TARGET_HALF_M + VDC_M + MAGNET_M
DET_AREA_M2       = 0.0144    # 12 cm × 12 cm calorimeter face [m²]
DET_HALF_X_M      = 0.06      # half-side of calorimeter face along x [m]
DET_HALF_Y_M      = 0.06      # half-side of calorimeter face along y [m]
EXPOSURE_DAYS     = 30.0

# Default mass and coupling grids
MASS_GRID_MEV   = np.array([1, 5, 10, 20, 50, 100, 200, 500])
COUPLING_GRID   = np.logspace(-8, -2, 60)   # GeV^-1


# ──────────────────────────────────────────────────────────────────────────────
# Bremsstrahlung spectrum helpers
# ──────────────────────────────────────────────────────────────────────────────

def bethe_heitler_spectrum(k_MeV, E0_MeV=BEAM_ENERGY_MEV, n_bins=500):
    """
    Analytic thin-target bremsstrahlung photon spectrum (Tsai/Bethe-Heitler,
    complete-screening limit).

    Returns [[E_MeV, photons_per_second], ...] normalised to one primary
    electron × ELECTRONS_PER_S.

    dN/dk ∝ (1/k) × [4/3 − (4/3)(k/E₀) + (k/E₀)²]  per radiation length

    The absolute normalisation: one e⁻ emits ≈1 photon per X₀ with the 1/k
    distribution.  We keep the spectral shape and scale by electrons/s.
    """
    k_min = 1.0          # MeV — Primakoff negligible below this
    k_max = 0.999 * E0_MeV
    k_edges = np.linspace(k_min, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    dk = k_edges[1:] - k_edges[:-1]

    x = k_centers / E0_MeV
    # Tsai formula (Eq. 3.83 in PDG review on passage of particles through matter)
    dNdk = (4.0/3.0 - 4.0*x/3.0 + x**2) / k_centers  # per e⁻ per X₀

    # Integrate over each bin width to get photons/bin/e⁻/X₀
    counts_per_electron = dNdk * dk

    # Scale to photons/second
    rates = counts_per_electron * ELECTRONS_PER_S

    return np.column_stack([k_centers, rates])


def load_geant4_brems_flux(csv_path, n_primaries):
    """
    Load Geant4 bremsstrahlung flux file (output/alplib_brems_flux.csv).

    The file has comment lines starting with # and data rows:
        energy_MeV,rate_per_second
    where rate = count/n_primaries * electrons_per_second (already scaled by
    run.h:WriteAlplibBremsFlux).
    """
    data = []
    beam_mode = None
    beam_current_A = None
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # The C++ writer (FluxData.h) stamps the beam normalization into
                # the header. Read it here so Python never re-derives or
                # double-applies the current — the rates below are ALREADY in
                # photons/s at this delivered current.
                if 'Beam mode:' in line:
                    beam_mode = line.split(':', 1)[1].strip()
                elif 'Beam current [A]:' in line:
                    try:
                        beam_current_A = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    e_mev = float(parts[0])
                    rate  = float(parts[1])
                    if e_mev > 0 and rate > 0:
                        data.append([e_mev, rate])
                except ValueError:
                    continue  # skip header if present

    if not data:
        raise RuntimeError(f"No usable data in {csv_path}")

    arr = np.array(data)

    # The file is already in photons/second at the beam current stamped in its
    # header; no rescaling needed here. To study a different LESA mode, rerun
    # Geant4 with that mode (or rescale linearly by the current ratio).
    print(f"[flux] Loaded {len(arr)} bins from {csv_path}")
    if beam_mode is not None:
        cur = f"{beam_current_A:.3e} A" if beam_current_A is not None else "unknown"
        print(f"[flux] Beam mode: {beam_mode}  (delivered current {cur})")
    elif beam_current_A is not None:
        print(f"[flux] Beam current: {beam_current_A:.3e} A")
    else:
        print("[flux] WARNING: no beam-mode/current header found "
              "(old flux file?) — assuming rates are already normalized.")
    print(f"[flux] Energy range: {arr[:,0].min():.1f} – {arr[:,0].max():.1f} MeV")
    print(f"[flux] Total rate: {arr[:,1].sum():.3e} photons/s")
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Coupling auto-selection and exact mean opening angle
# ──────────────────────────────────────────────────────────────────────────────

def pick_safe_coupling(ma_MeV, photon_flux, target_decay_length_m=5.0):
    """
    Return g (GeV⁻¹) so that the mean ALP boosted decay length ≈ target_decay_length_m,
    ensuring surv_prob ≈ 1 (ALPs reach the detector).  DAMSA detector distance is
    ~0.52 m (target centre → calo face), so 5 m gives surv_prob ≈ 1 − 0.52/5 ≈ 0.9.

        Γ = g² mₐ³ / (64π),  L_decay = (Eₐ/mₐ)(ℏc/Γ)
        ⇒ g² = 64π · ℏc · Eₐ / (L_decay · mₐ⁴)       [MeV⁻²]
    (mₐ⁴: one power from Γ ∝ mₐ³ and one from the boost Eₐ/mₐ.)
    """
    E = photon_flux[:, 0]
    w = photon_flux[:, 1]
    mask = E > ma_MeV
    if mask.sum() == 0:
        return 1e-4
    Ea_typ = np.average(E[mask], weights=w[mask])       # MeV

    g_sq_MeV = (64.0 * np.pi * METER_BY_MEV * Ea_typ
                / (target_decay_length_m * ma_MeV**4))
    g_MeV = np.sqrt(g_sq_MeV)
    return g_MeV * 1000.0                               # MeV⁻¹ → GeV⁻¹


def expected_mean_theta_per_alp(Ea, ma, n_integration=200):
    """
    Exact ⟨θ_open⟩ for one ALP of (Ea, ma), averaged uniformly over rest-frame
    cos θ* ∈ [−1, 1]:
        cos θ_open(u) = 1 − 2 mₐ² / (Eₐ² − pₐ² u²)
    In the UR limit this approaches π mₐ/Eₐ, NOT 2 mₐ/Eₐ (which is the minimum).
    """
    if Ea <= ma:
        return 0.0
    pa2 = Ea*Ea - ma*ma
    u = np.linspace(-1.0, 1.0, n_integration)
    denom = Ea*Ea - pa2 * u*u
    denom = np.clip(denom, ma*ma, None)
    cos_theta = 1.0 - 2.0 * ma*ma / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.trapezoid(theta, u) / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Core alplib calculation
# ──────────────────────────────────────────────────────────────────────────────

def run_alplib(photon_flux, ma_MeV, coupling_GeV, n_samples=5000):
    """Run FluxPrimakoffIsotropic for one (mass, coupling) point.

    photon_flux: [[E_MeV, photons/s], ...] entering the tungsten target.
    coupling_GeV: g_agammagamma in GeV^-1, converted internally to MeV^-1 for alplib.
    Returns (flux_obj, generator).
    """
    # Unit conversion: 1 GeV^-1 = 1e-3 MeV^-1
    coupling_MeV = coupling_GeV / 1000.0

    # alplib's primakoff_sigma (Creswick, screened) is an mₐ ≪ E_γ
    # approximation: no mass dependence beyond the production threshold.
    if ma_MeV > 500.0:
        print(f"  [warn] ma={ma_MeV:.0f} MeV: alplib primakoff_sigma neglects "
              f"mass effects in the cross section (valid for ma << E_gamma); "
              f"production rate increasingly overestimated at this mass.")

    flux_obj = FluxPrimakoffIsotropic(
        photon_flux   = photon_flux,
        target        = Material("W"),
        det_dist      = DECAY_ZMIN_M,
        det_length    = DECAY_ZMAX_M - DECAY_ZMIN_M,
        det_area      = DET_AREA_M2,
        axion_mass    = ma_MeV,
        axion_coupling= coupling_MeV,    # MeV^-1
        n_samples     = n_samples,
    )
    flux_obj.simulate()
    # is_isotropic=False: the default (True) multiplies every weight by the
    # isotropic solid-angle acceptance det_area/(4π·det_dist²) ≈ 5e-3, which
    # models an isotropic ALP source (reactor/sun). DAMSA ALPs inherit the
    # direction of forward-collimated 8 GeV bremsstrahlung photons — alplib
    # itself samples the decays with θ_ALP = 0 (forward) in
    # simulate_decay_4vectors — so the isotropic factor is the wrong model
    # AND double-counts the transverse_acceptance_mask() applied downstream.
    # Acceptance is instead applied per decay by the transverse mask (Python)
    # or by the real Geant4 geometry (re-injection path).
    flux_obj.propagate(is_isotropic=False)

    gen = PhotonEventGenerator(flux_obj, Material("CsI"))
    return flux_obj, gen


def sample_decay_vertex_z(p41_list, p42_list, ma_MeV, coupling_GeV, rng=None,
                          z_min_m=None, z_max_m=None):
    """
    Sample a decay-vertex z (metres from target centre) per γγ pair from the
    exponential decay law truncated to the fiducial window [z_min, z_max]
    (defaults: DECAY_ZMIN_M → DECAY_ZMAX_M, i.e. target exit → calo face).

    The alplib weight already contains P(decay in window); this is the
    CONDITIONAL vertex distribution given that decay, so weights stay valid
    (importance sampling). ALP energy/boost is reconstructed per pair from
    E_a = E1 + E2, p_a = |p1 + p2|.
    """
    if rng is None:
        rng = np.random.default_rng(1234)
    z0 = DECAY_ZMIN_M if z_min_m is None else z_min_m
    z1 = DECAY_ZMAX_M if z_max_m is None else z_max_m

    g_MeV = coupling_GeV / 1000.0
    gamma_w = g_MeV**2 * ma_MeV**3 / (64.0 * np.pi)      # MeV

    Ea = np.array([p1.energy() + p2.energy() for p1, p2 in zip(p41_list, p42_list)])
    pa = np.sqrt(np.maximum(Ea**2 - ma_MeV**2, 1e-30))
    L = (pa / ma_MeV) * (METER_BY_MEV / gamma_w)          # lab decay length [m]
    L = np.maximum(L, 1e-12)

    u = rng.uniform(size=len(Ea))
    a = np.exp(-z0 / L)
    b = np.exp(-z1 / L)
    return -L * np.log(a - u * (a - b))


def signal_events(photon_flux, ma_MeV, coupling_GeV, n_samples=500,
                  threshold_MeV=5.0, n_decay_samples=10, rng=None):
    """Return total signal events for given exposure.

    Samples a→γγ decays with a per-event decay vertex inside the fiducial
    window (target exit → calo face), requires BOTH photons to hit the calo
    face from that vertex (transverse_acceptance_mask) AND each photon to
    carry at least threshold_MeV (calorimeter detection threshold — a photon
    below threshold is missing energy and the pair is not reconstructable).
    This replaces gen.decays(), which applied no transverse acceptance at all.
    """
    flux_obj, gen = run_alplib(photon_flux, ma_MeV, coupling_GeV, n_samples)
    if len(flux_obj.axion_energy) == 0:
        return 0.0
    p41, p42, wgts = gen.simulate_decay_4vectors(
        days_exposure=EXPOSURE_DAYS, n_samples=n_decay_samples)
    if len(wgts) == 0:
        return 0.0
    wgts = np.asarray(wgts)
    vz = sample_decay_vertex_z(p41, p42, ma_MeV, coupling_GeV, rng)
    accept = transverse_acceptance_mask(p41, p42, det_dist_m=DET_DIST_M,
                                        vertex_z_m=vz)
    e1 = np.array([p.energy() for p in p41])
    e2 = np.array([p.energy() for p in p42])
    sel = accept & (e1 >= threshold_MeV) & (e2 >= threshold_MeV)
    return float(wgts[sel].sum())


# ──────────────────────────────────────────────────────────────────────────────
# Transverse detector acceptance (analytic, ignores material)
# ──────────────────────────────────────────────────────────────────────────────

def transverse_acceptance_mask(p41_list, p42_list,
                               det_dist_m=DET_DIST_M,
                               half_x=DET_HALF_X_M,
                               half_y=DET_HALF_Y_M,
                               vertex_z_m=0.0):
    """
    Straight-line propagate each γγ pair from (0, 0, vertex_z) to z = det_dist_m
    and return a boolean array marking pairs where BOTH photons land inside the
    calorimeter face (|x| ≤ half_x, |y| ≤ half_y).

    vertex_z_m: scalar OR per-pair array of decay-vertex z positions (metres
    from target centre) — use sample_decay_vertex_z() for the physical
    distribution. z_decay is NOT hardcoded to the target centre any more:
    a decay late in the VDC sees a much wider angular acceptance than one at
    the target, so this matters for wide-angle (soft / heavy-ALP) pairs.

    alplib's `decay_axion_weight` (as used here, propagate(is_isotropic=False))
    includes only the longitudinal survival × decay-in-window probability.
    It does NOT check whether each γ trajectory actually hits the calorimeter
    face. This function IS the geometric acceptance.

    Approximations:
      • No magnetic deflection (photons are neutral, so this is exact for γ).
      • Backward-going photons (pz ≤ 0) are rejected.
    """
    n = len(p41_list)
    vz = np.broadcast_to(np.asarray(vertex_z_m, dtype=float), (n,))
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        p1 = p41_list[i]
        p2 = p42_list[i]
        if p1.p3 <= 0.0 or p2.p3 <= 0.0:
            continue
        dz = det_dist_m - vz[i]
        if dz <= 0.0:
            continue
        x1 = (p1.p1 / p1.p3) * dz
        y1 = (p1.p2 / p1.p3) * dz
        x2 = (p2.p1 / p2.p3) * dz
        y2 = (p2.p2 / p2.p3) * dz
        if (abs(x1) <= half_x and abs(y1) <= half_y
                and abs(x2) <= half_x and abs(y2) <= half_y):
            mask[i] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Opening angle calculation
# ──────────────────────────────────────────────────────────────────────────────

def compute_opening_angles(photon_flux, ma_MeV, coupling_GeV=None, n_samples=500,
                           n_decay_samples=200):
    """
    Compute the weighted opening angle distribution for a → γγ.

    If coupling_GeV is None, pick a coupling that gives surv_prob ≈ 1 so the
    opening angle distribution isn't biased by detector acceptance (heavy ALPs
    otherwise all decay before reaching the calorimeter).

    Returns a dict with the full distribution and the in-acceptance subset:
        thetas, wgts, mean_mrad,                  — all decays
        thetas_in, wgts_in, mean_in_mrad,         — both photons hit calo face
        accept_frac,                              — Σ wgts_in / Σ wgts
        theta_kin_mrad, coupling_used.
    """
    if coupling_GeV is None:
        coupling_GeV = pick_safe_coupling(ma_MeV, photon_flux,
                                          target_decay_length_m=5.0)

    empty = {
        "thetas": np.array([]), "wgts": np.array([]), "mean_mrad": 0.0,
        "thetas_in": np.array([]), "wgts_in": np.array([]), "mean_in_mrad": 0.0,
        "accept_frac": 0.0, "Ea": np.array([]),
        "theta_kin_mrad": 0.0, "coupling_used": coupling_GeV,
    }

    flux_obj, gen = run_alplib(photon_flux, ma_MeV, coupling_GeV, n_samples)

    if len(flux_obj.axion_energy) == 0:
        return empty

    # Exact kinematic expectation from per-ALP integration (same ensemble)
    Ea_arr = np.asarray(flux_obj.axion_energy)
    alp_wgt = np.asarray(flux_obj.decay_axion_weight)
    if alp_wgt.sum() > 0:
        theta_kin_per = np.array([expected_mean_theta_per_alp(e, ma_MeV)
                                  for e in Ea_arr])
        theta_kin_mrad = 1000.0 * np.average(theta_kin_per, weights=alp_wgt)
    else:
        theta_kin_mrad = 0.0

    p41_list, p42_list, wgts = gen.simulate_decay_4vectors(
        days_exposure=EXPOSURE_DAYS,
        n_samples=n_decay_samples,
    )

    thetas = []
    for p1, p2 in zip(p41_list, p42_list):
        # alplib LorentzVector API: .p0=E, .p1=px, .p2=py, .p3=pz
        p1_vec = np.array([p1.p1, p1.p2, p1.p3])
        p2_vec = np.array([p2.p1, p2.p2, p2.p3])
        mag1 = np.linalg.norm(p1_vec)
        mag2 = np.linalg.norm(p2_vec)
        if mag1 < 1e-12 or mag2 < 1e-12:
            thetas.append(0.0)
            continue
        cos_theta = np.dot(p1_vec, p2_vec) / (mag1 * mag2)
        thetas.append(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    thetas = np.array(thetas)
    wgts   = np.array(wgts)

    if len(wgts) == 0 or wgts.sum() == 0:
        out = dict(empty)
        out["thetas"] = thetas
        out["wgts"] = wgts
        out["theta_kin_mrad"] = theta_kin_mrad
        return out

    mean_theta_mrad = 1000.0 * np.average(thetas, weights=wgts)

    # Apply analytic transverse acceptance cut: both γ must hit calo face,
    # projected from a per-event decay vertex sampled in the fiducial window.
    # Pass det_dist_m explicitly so it reads the current DET_DIST_M global
    # (which may have been updated by --vdc-length).
    vz = sample_decay_vertex_z(p41_list, p42_list, ma_MeV, coupling_GeV)
    accept = transverse_acceptance_mask(p41_list, p42_list,
                                        det_dist_m=DET_DIST_M, vertex_z_m=vz)
    thetas_in = thetas[accept]
    wgts_in   = wgts[accept]
    if wgts_in.sum() > 0:
        mean_in_mrad = 1000.0 * np.average(thetas_in, weights=wgts_in)
        accept_frac = float(wgts_in.sum() / wgts.sum())
    else:
        mean_in_mrad = 0.0
        accept_frac = 0.0

    # per-pair ALP energy — used by the decomposition plot (theta_min ~ E_a)
    Ea_pairs = np.array([p1.energy() + p2.energy()
                         for p1, p2 in zip(p41_list, p42_list)])

    return {
        "thetas": thetas, "wgts": wgts, "mean_mrad": mean_theta_mrad,
        "thetas_in": thetas_in, "wgts_in": wgts_in, "mean_in_mrad": mean_in_mrad,
        "accept_frac": accept_frac, "Ea": Ea_pairs,
        "theta_kin_mrad": theta_kin_mrad, "coupling_used": coupling_GeV,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Export decay photon 4-vectors for Geant4 re-injection
# ──────────────────────────────────────────────────────────────────────────────

def export_decay_4vectors(photon_flux, ma_MeV, coupling_GeV=None,
                          out_dir="output", n_samples=10, n_decay_samples=200):
    """
    Export γγ decay 4-vectors to CSV for DamsaALPDecayGenerator.

    If coupling_GeV is None, auto-select a safe coupling per mass so heavy ALPs
    actually reach the detector (otherwise the exported weights are 0 — useless
    for Geant4 re-injection).

    CSV columns: E1_MeV,px1,py1,pz1,E2_MeV,px2,py2,pz2,weight_evts_per_day,
                 decay_z_m
    decay_z_m = decay-vertex z in METRES from the target centre, sampled from
    the truncated exponential decay law inside the fiducial window
    (target exit → calo face). DamsaALPDecayGenerator fires the pair from
    (0, 0, vertex_z + decay_z_m) instead of the fixed target centre.
    """
    if coupling_GeV is None:
        coupling_GeV = pick_safe_coupling(ma_MeV, photon_flux,
                                          target_decay_length_m=5.0)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"alp_decay_photons_ma{ma_MeV:.0f}MeV.csv"

    flux_obj, gen = run_alplib(photon_flux, ma_MeV, coupling_GeV, n_samples)

    if len(flux_obj.axion_energy) == 0:
        print(f"  [ma={ma_MeV} MeV] No ALP events generated (mass > max photon energy?)")
        return

    p41_list, p42_list, wgts = gen.simulate_decay_4vectors(
        days_exposure=EXPOSURE_DAYS,
        n_samples=n_decay_samples,
    )
    vz = sample_decay_vertex_z(p41_list, p42_list, ma_MeV, coupling_GeV)

    rows = []
    for p1, p2, w, z in zip(p41_list, p42_list, wgts, vz):
        rows.append([
            p1.energy(), p1.p1, p1.p2, p1.p3,
            p2.energy(), p2.p1, p2.p2, p2.p3,
            w, z,
        ])

    arr = np.array(rows)
    np.savetxt(str(out_path), arr, delimiter=",",
               header="E1_MeV,px1,py1,pz1,E2_MeV,px2,py2,pz2,weight_evts_per_day,decay_z_m",
               comments="")
    print(f"  [ma={ma_MeV:.0f} MeV, g={coupling_GeV:.2e} GeV⁻¹] "
          f"{len(rows)} decay pairs → {out_path}"
          f"  (total weight: {np.array(wgts).sum():.3e} events over {EXPOSURE_DAYS} days)")


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity curve
# ──────────────────────────────────────────────────────────────────────────────

def compute_sensitivity(photon_flux, mass_grid=MASS_GRID_MEV,
                        coupling_grid=COUPLING_GRID,
                        cl_threshold=2.3,
                        threshold_MeV=5.0):
    """
    DAMSA sensitivity is a CLOSED contour in (mₐ, g) space:
      • low-g edge:  g too small ⇒ decay prob in detector ∝ g² → 0
      • high-g edge: g too large ⇒ ALPs decay before reaching detector
                     (cτβγ ≪ det_dist ⇒ surv_prob → 0)
      • a band of excluded g between the two edges where N_signal > cl_threshold.

    For each mass, scans the coupling grid and records the lower and upper
    edges of the excluded band.

    Returns (mass_grid, g_lower, g_upper) — NaN if not excluded.
    """
    g_lower = np.full(len(mass_grid), np.nan)
    g_upper = np.full(len(mass_grid), np.nan)

    for i, ma in enumerate(mass_grid):
        print(f"  Sensitivity scan: ma={ma} MeV ...")
        n_sig = np.zeros(len(coupling_grid))
        for j, gagg in enumerate(coupling_grid):
            n_sig[j] = signal_events(photon_flux, ma, gagg,
                                     n_samples=200, threshold_MeV=threshold_MeV)

        excluded = n_sig > cl_threshold
        if excluded.any():
            idx = np.where(excluded)[0]
            g_lower[i] = coupling_grid[idx[0]]
            g_upper[i] = coupling_grid[idx[-1]]
            n_max = n_sig[idx].max()
            print(f"    excluded band: "
                  f"[{g_lower[i]:.2e}, {g_upper[i]:.2e}] GeV⁻¹  (max N_sig={n_max:.1f})")
        else:
            print(f"    not excluded (max N_sig={n_sig.max():.2e})")

    return mass_grid, g_lower, g_upper


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

# E_a/m_a boost slices for the decomposition panels. theta_min = 2 asin(m/E)
# depends only on this ratio, so the same slices work for every mass:
# ratio bands ~ (theta_min): 1–1.2 (>113°), 1.2–1.6 (77–113°), 1.6–3 (39–77°),
# 3–8 (14–39°), 8–∞ (<14°).  Sequential blues, light = soft.
_DECOMP_SLICES = [(1.0, 1.2, "#c6dbef"), (1.2, 1.6, "#9ecae1"),
                  (1.6, 3.0, "#6baed6"), (3.0, 8.0, "#3182bd"),
                  (8.0, np.inf, "#08519c")]


def plot_opening_angle_decomposition(r, out_dir="plots"):
    """
    Per-mass decomposition of the full (no-acceptance) opening-angle
    distribution into ALP boost (E_a/m_a) slices.

    The inclusive distribution is quasi-flat and uninformative: every energy
    slice contributes a Jacobian peak at its own theta_min = 2 asin(m_a/E_a),
    and integrating over the soft-dominated spectrum smears the peaks across
    all of 0–180°. Showing the slices restores the physics.
    """
    if not PLOT or len(r["thetas"]) == 0 or r["wgts"].sum() <= 0:
        return
    ma = r["ma"]
    t_deg = np.degrees(r["thetas"])
    w = r["wgts"]
    ratio = r["Ea"] / ma
    bins = np.linspace(0.0, 180.0, 90)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(t_deg, bins=bins, weights=w, histtype="step", color="#555555",
            lw=1.8, label="all $E_a$ (inclusive)")
    for lo, hi, c in _DECOMP_SLICES:
        m = (ratio >= lo) & (ratio < hi)
        if m.sum() < 5 or w[m].sum() <= 0:
            continue
        counts, _ = np.histogram(t_deg[m], bins=bins, weights=w[m])
        theta_min = 2.0 * np.degrees(np.arcsin(min(1.0, 1.0 / lo)))
        hi_lab = "∞" if np.isinf(hi) else f"{hi:g}"
        ax.hist(t_deg[m], bins=bins, weights=w[m], histtype="step",
                color=c, lw=1.3)
        ax.annotate(f"$E_a/m_a$ {lo:g}–{hi_lab}",
                    xy=(min(theta_min + 4, 150), counts.max() * 1.35),
                    color=c, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"opening angle $\theta_{\gamma\gamma}$ [deg]")
    ax.set_ylabel(f"weighted events / bin  (expected in {EXPOSURE_DAYS:.0f} days)")
    ax.set_title(rf"$m_a$ = {ma:g} MeV, g = {r['coupling_used']:.2e} GeV$^{{-1}}$: "
                 r"opening angle by boost slice"
                 "\n"
                 r"each slice peaks at $\theta_{min}=2\arcsin(m_a/E_a)$;"
                 " inclusive sum is flat", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, frameon=False, loc="lower center")
    fig.tight_layout()
    out = Path(out_dir) / f"opening_angles_decomp_ma{ma:g}MeV.png"
    fig.savefig(str(out), dpi=150)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Opening angle decomposition → {out}")


def plot_opening_angles(results, out_dir="plots", overlay_masses=None):
    """
    Opening-angle plots:
      • opening_angles.png — in-acceptance overlay across masses (both photons
        hit the calo face). Peaked and narrow; this is what the detector sees.
      • opening_angles_decomp_ma<M>MeV.png — per-mass boost-slice
        decomposition of the full distribution (the inclusive no-cut curve is
        quasi-flat by construction and is only shown as the gray envelope).
    results: list of dicts from compute_opening_angles().
    overlay_masses: array of masses to include in overlay (None = all).
    """
    if not PLOT:
        return

    # Per-mass decomposition figures (all computed masses)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for r in results:
        plot_opening_angle_decomposition(r, out_dir=out_dir)

    # Filter to requested masses for the overlay
    if overlay_masses is not None:
        results = [r for r in results if r['ma'] in overlay_masses]
        if not results:
            print(f"[overlay] No matching results for masses {overlay_masses}")
            return

    fig, ax_in = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")

    for k, r in enumerate(results):
        if len(r["thetas_in"]) == 0 or r["wgts_in"].sum() <= 0:
            continue
        color = cmap(k / max(1, len(results) - 1))
        t_deg_in = np.degrees(r["thetas_in"])
        hi = max(np.percentile(t_deg_in, 99), 1.0) * 1.1
        bins_in = np.linspace(0.0, hi, 50)
        ax_in.hist(t_deg_in, weights=r["wgts_in"], bins=bins_in,
                   histtype='step', color=color, label=f"mₐ={r['ma']} MeV")

    ax_in.set_xlabel("Opening angle (degrees)")
    ax_in.set_ylabel("Weighted events / bin")
    ax_in.set_title(f"Both γ on calorimeter face "
                    f"({DET_HALF_X_M*200:.0f}×{DET_HALF_Y_M*200:.0f} cm² "
                    f"@ {DET_DIST_M*100:.0f} cm)")
    ax_in.set_yscale("log")
    ax_in.legend(fontsize=7)
    ax_in.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "opening_angles.pdf"))
    fig.savefig(str(Path(out_dir) / "opening_angles.png"), dpi=150)
    plt.close(fig)
    print(f"Opening angle plot (in-acceptance overlay) → {out_dir}/opening_angles.pdf")


def plot_sensitivity(mass_grid, g_lower, g_upper, out_dir="plots"):
    if not PLOT:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    mask = ~np.isnan(g_lower)
    fig, ax = plt.subplots(figsize=(8, 5))
    if mask.any():
        ax.fill_between(mass_grid[mask], g_lower[mask], g_upper[mask],
                        color='steelblue', alpha=0.3,
                        label=f"DAMSA Phase 1 ({EXPOSURE_DAYS:.0f} days)")
        ax.plot(mass_grid[mask], g_lower[mask], 'o-', color='steelblue')
        ax.plot(mass_grid[mask], g_upper[mask], 'o-', color='steelblue')
    ax.set_xlabel("mₐ (MeV)")
    ax.set_ylabel("g_aγγ (GeV⁻¹)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("90% CL sensitivity: g_aγγ vs mₐ")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "sensitivity_curve.pdf"))
    fig.savefig(str(Path(out_dir) / "sensitivity_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Sensitivity plot → {out_dir}/sensitivity_curve.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global VDC_M, DET_DIST_M  # noqa: PLW0603  — may be overridden by --vdc-length
    parser = argparse.ArgumentParser(description="DAMSA ALP signal pipeline")
    parser.add_argument("--flux", default=None,
                        help="Path to Geant4 alplib_brems_flux.csv")
    parser.add_argument("--nprimaries", type=int, default=10000,
                        help="Number of Geant4 primary events (for info only)")
    parser.add_argument("--analytic", action="store_true",
                        help="Use analytic Bethe-Heitler spectrum instead of Geant4 flux")
    parser.add_argument("--mass", type=float, default=None,
                        help="Single ALP mass in MeV (default: scan all)")
    parser.add_argument("--coupling", type=float, default=1e-3,
                        help="g_aγγ in GeV^-1 (default: 1e-3)")
    parser.add_argument("--auto-coupling", action="store_true",
                        help="Auto-pick coupling per mass so surv_prob ≈ 1 "
                             "(recommended for opening-angle and 4-vec export)")
    parser.add_argument("--no-4vec", action="store_true",
                        help="Skip 4-vector CSV export")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="Skip sensitivity curve calculation")
    parser.add_argument("--outdir", default="output",
                        help="Output directory for CSVs and plots")
    parser.add_argument("--vdc-length", type=float, default=None, metavar="METERS",
                        help="Vacuum decay chamber length in metres (default: %.2f m = %.0f cm). "
                             "Updates DET_DIST_M = TARGET_HALF + VDC + MAGNET." % (VDC_M, VDC_M*100))
    parser.add_argument("--overlay-mass", type=str, default=None,
                        help="Comma-separated masses to include in overlay plot "
                             "(e.g., '10,20,50,100,200')")
    parser.add_argument("--threshold-mev", type=float, default=5.0,
                        help="Per-photon calorimeter detection threshold in MeV "
                             "(both photons must exceed it; a photon below it is "
                             "missing energy). Default: 5 MeV")
    args = parser.parse_args()

    # ── Apply VDC length override ─────────────────────────────────────────────
    if args.vdc_length is not None:
        VDC_M      = args.vdc_length
        DET_DIST_M = TARGET_HALF_M + VDC_M + MAGNET_M
        print(f"[geometry] VDC length overridden to {VDC_M*100:.1f} cm → "
              f"DET_DIST_M = {DET_DIST_M*100:.1f} cm")

    # ── Load / build photon flux ──────────────────────────────────────────────
    if args.flux and not args.analytic:
        photon_flux = load_geant4_brems_flux(args.flux, args.nprimaries)
    else:
        print("[flux] Using analytic Bethe-Heitler bremsstrahlung spectrum")
        photon_flux = bethe_heitler_spectrum(1.0, E0_MeV=BEAM_ENERGY_MEV)
        print(f"[flux] {len(photon_flux)} bins, "
              f"E: {photon_flux[:,0].min():.1f}–{photon_flux[:,0].max():.1f} MeV, "
              f"total rate: {photon_flux[:,1].sum():.3e} photons/s")

    mass_grid = np.array([args.mass]) if args.mass else MASS_GRID_MEV

    # ── Calculation ranges (meeting item: document what is being computed) ──
    print("\n=== ALP signal calculation ranges ===")
    print(f"  Photon flux:      {photon_flux[:,0].min():.1f} – "
          f"{photon_flux[:,0].max():.1f} MeV "
          f"({photon_flux[:,1].sum():.3e} photons/s total)")
    print(f"  ALP masses:       {mass_grid.min():g} – {mass_grid.max():g} MeV "
          f"(per mass, ALP energies span m_a → max flux energy)")
    print(f"  Coupling grid:    {COUPLING_GRID.min():.1e} – "
          f"{COUPLING_GRID.max():.1e} GeV⁻¹ (sensitivity scan)")
    print(f"  Photon threshold: {args.threshold_mev:g} MeV per photon "
          f"(pairs with either γ below are counted as missing energy)")
    print(f"  Decay window:     z = {DECAY_ZMIN_M*100:.0f} – "
          f"{DECAY_ZMAX_M*100:.0f} cm from target centre "
          f"(target exit → calo face)")

    # Parse overlay masses for plotting
    overlay_masses = None
    if args.overlay_mass:
        overlay_masses = np.array([float(m) for m in args.overlay_mass.split(',')])
        print(f"[overlay] Plotting masses: {args.overlay_mass} MeV")

    # ── Opening angle scan ────────────────────────────────────────────────────
    auto_g = (args.coupling is None) or args.auto_coupling
    label = "auto-selected" if auto_g else f"g_aγγ = {args.coupling:.1e} GeV⁻¹"
    print(f"\n=== Opening angle scan ({label}) ===")
    print(f"  θ_MC      = mean over all alplib decays (no transverse cut)")
    print(f"  θ_in      = mean over γγ pairs whose photons both hit the calo face")
    print(f"  accept    = Σ wgts_in / Σ wgts (transverse acceptance fraction)\n")
    print(f"{'mₐ (MeV)':>10} {'g (GeV⁻¹)':>12} {'θ_MC (mrad)':>13}"
          f" {'θ_kin (mrad)':>14} {'MC/kin':>8}"
          f" {'θ_in (mrad)':>13} {'accept':>10}")
    print("-" * 95)
    angle_results = []
    for ma in mass_grid:
        g_in = None if auto_g else args.coupling
        # n_samples raised 10 → 500 (2026-07-10): 10 ALP energy samples badly
        # under-resolved the weighted opening-angle distribution.
        r = compute_opening_angles(photon_flux, ma, g_in,
                                   n_samples=500, n_decay_samples=300)
        r["ma"] = ma
        angle_results.append(r)
        ratio = (r["mean_mrad"] / r["theta_kin_mrad"]) if r["theta_kin_mrad"] > 0 else 0.0
        print(f"{ma:10.1f} {r['coupling_used']:12.2e} {r['mean_mrad']:13.2f}"
              f" {r['theta_kin_mrad']:14.2f} {ratio:8.3f}"
              f" {r['mean_in_mrad']:13.2f} {r['accept_frac']:10.3e}")

    if PLOT:
        plot_opening_angles(angle_results, out_dir=str(Path(args.outdir).parent / "plots"),
                          overlay_masses=overlay_masses)

    # ── Decay 4-vector export ─────────────────────────────────────────────────
    if not args.no_4vec:
        print(f"\n=== Exporting decay photon 4-vectors → {args.outdir}/ ===")
        for ma in mass_grid:
            g_used = None if auto_g else args.coupling
            export_decay_4vectors(photon_flux, ma, g_used,
                                  out_dir=args.outdir,
                                  n_samples=20, n_decay_samples=500)

    # ── Sensitivity curve ─────────────────────────────────────────────────────
    if not args.no_sensitivity:
        print(f"\n=== Sensitivity curve scan ===")
        masses, g_lo, g_hi = compute_sensitivity(photon_flux, mass_grid=mass_grid,
                                                 threshold_MeV=args.threshold_mev)
        print("\nExclusion summary (90% CL, closed band):")
        for ma, lo, hi in zip(masses, g_lo, g_hi):
            if np.isnan(lo):
                print(f"  ma={ma:6.1f} MeV → not excluded in coupling range")
            else:
                print(f"  ma={ma:6.1f} MeV → excluded band: "
                      f"[{lo:.2e}, {hi:.2e}] GeV⁻¹")
        if PLOT:
            plot_sensitivity(masses, g_lo, g_hi,
                             out_dir=str(Path(args.outdir).parent / "plots"))

    print("\nDone.")


if __name__ == "__main__":
    main()
