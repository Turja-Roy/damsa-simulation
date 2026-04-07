#!/usr/bin/env python3
"""
Validate ALP opening angle calculations before the APS presentation.

These are INTERNAL consistency checks — the idea is not to match an external
analytic weighting of the photon spectrum (which is ambiguous), but to verify
that alplib's MC 2-body decay produces opening angles consistent with the
ultra-relativistic formula θ ≈ 2mₐ/⟨Eₐ⟩, where ⟨Eₐ⟩ is computed from the
SAME ensemble of ALPs that alplib uses for its 4-vector simulation.

Three checks:
  (1) MC vs formula θ — should agree to ~5% for ma ≪ Eₐ
  (2) Signal rate ∝ g⁴ scaling (in the small-coupling regime)
  (3) Invariant mass reconstruction m_γγ ≈ mₐ

Important note on coupling regime
---------------------------------
For ma = 100 MeV and g_aγγ = 1e-3 GeV⁻¹, the ALP proper decay length is
cτ ≈ 4 mm, and boosted by γ ~ 10 → lab decay length ~ 4 cm, which is much
less than the 47 cm gap to the calorimeter.  Virtually all ALPs decay before
the detector (surv_prob ~ 0), driving signal → 0 and spoiling naive g-scaling
tests.  This script auto-selects a coupling where surv_prob ≈ 1 so the
validation is meaningful.

Usage:
    python validate_opening_angles.py --flux output/alplib_brems_flux.csv
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from alplib.fluxes import FluxPrimakoffIsotropic
    from alplib.materials import Material
    from alplib.generators import PhotonEventGenerator
    from alplib.constants import CHARGE_COULOMBS, S_PER_DAY, METER_BY_MEV
except ImportError:
    sys.exit("alplib not found.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT = True
except ImportError:
    PLOT = False

from alp_signal_pipeline import (
    bethe_heitler_spectrum,
    load_geant4_brems_flux,
    run_alplib,
    BEAM_ENERGY_MEV, DET_DIST_M, DET_LENGTH_M, DET_AREA_M2, EXPOSURE_DAYS,
)

MASS_GRID_MEV = np.array([10, 20, 50, 100, 200, 500])


# ──────────────────────────────────────────────────────────────────────────────
# Coupling auto-selection
# ──────────────────────────────────────────────────────────────────────────────

def pick_safe_coupling(ma_MeV, photon_flux, target_decay_length_m=5.0):
    """
    Return a coupling g (in GeV⁻¹) such that the mean ALP boosted decay length
    is ~target_decay_length_m.  This ensures surv_prob ≈ 1 for the validation
    checks (so the opening angles are not biased by the detector acceptance).

    cτ = ℏ / Γ,  Γ = g² mₐ³ / (64π)
    L_decay = βγ cτ ≈ (Eₐ/mₐ) × (ℏc / Γ)
    Solving for g²:
        g² = (Eₐ / mₐ²) × (ℏc × 64π) / (L_decay × mₐ)
           = 64π × ℏc × Eₐ / (L_decay × mₐ³)

    Units: ℏc in MeV·m = METER_BY_MEV.  g² in MeV⁻².
    """
    # Estimate typical ALP energy: use the photon-rate-weighted mean over E > ma
    E = photon_flux[:, 0]
    w = photon_flux[:, 1]
    mask = E > ma_MeV
    if mask.sum() == 0:
        return 1e-4  # fallback
    Ea_typ = np.average(E[mask], weights=w[mask])   # MeV

    # Solve g² so decay length matches target
    g_sq_MeV = (64.0 * np.pi * METER_BY_MEV * Ea_typ
                / (target_decay_length_m * ma_MeV**3))
    g_MeV = np.sqrt(g_sq_MeV)
    g_GeV = g_MeV * 1000.0      # convert MeV⁻¹ → GeV⁻¹
    return g_GeV


# ──────────────────────────────────────────────────────────────────────────────
# Exact per-ALP expected mean opening angle (averaged over isotropic rest-frame)
# ──────────────────────────────────────────────────────────────────────────────

def expected_mean_theta_per_alp(Ea, ma, n_integration=200):
    """
    Exact mean of θ_open for a single ALP with energy Ea and mass ma,
    averaged uniformly over rest-frame cos θ* ∈ [−1, 1]:

        cos θ_open(u) = 1 − 2 mₐ² / (Eₐ² − pₐ² u²)         where u = cos θ*
        θ_open(u)     = arccos(cos θ_open(u))

        ⟨θ_open⟩ = (1/2) ∫₋₁¹ du θ_open(u)

    This is the MEAN angle, not the minimum (2 mₐ/Eₐ).  In the
    ultra-relativistic limit it approaches π mₐ/Eₐ (a factor π/2 larger
    than the minimum).
    """
    if Ea <= ma:
        return 0.0
    pa2 = Ea*Ea - ma*ma
    u   = np.linspace(-1.0, 1.0, n_integration)
    denom = Ea*Ea - pa2 * u*u
    # guard against tiny denominator for β → 1, u → ±1
    denom = np.clip(denom, ma*ma, None)
    cos_theta = 1.0 - 2.0 * ma*ma / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.trapezoid(theta, u) / 2.0)  # (1/2) × integral


# ──────────────────────────────────────────────────────────────────────────────
# Internal consistency: θ_MC vs exact kinematic expectation
# ──────────────────────────────────────────────────────────────────────────────

def opening_angle_internal_check(photon_flux, ma_MeV):
    """
    Runs alplib and compares:
      • θ_MC       : weighted mean opening angle from simulate_decay_4vectors
      • θ_kin      : exact kinematic expectation — per-ALP ⟨θ_open⟩ computed
                     from (Eₐ, mₐ) by integrating over isotropic rest-frame
                     decay angles, then ensemble-averaged with the same
                     decay_axion_weight.
      • θ_UR       : ultra-relativistic approximation π mₐ/⟨Eₐ⟩ (for reference)

    θ_MC and θ_kin should agree to ~5% — this tests the MC 2-body decay
    against the exact analytic form using the SAME ensemble.
    """
    coupling_GeV = pick_safe_coupling(ma_MeV, photon_flux,
                                      target_decay_length_m=5.0)

    flux_obj, gen = run_alplib(photon_flux, ma_MeV, coupling_GeV,
                               n_samples=20)

    if len(flux_obj.axion_energy) == 0:
        return None, None, None, None, 0

    Ea  = np.asarray(flux_obj.axion_energy)
    wgt = np.asarray(flux_obj.decay_axion_weight)

    if wgt.sum() <= 0:
        return None, None, None, None, 0

    mean_Ea = np.average(Ea, weights=wgt)

    # Exact kinematic expectation per ALP, ensemble-averaged
    theta_kin_per = np.array([expected_mean_theta_per_alp(e, ma_MeV) for e in Ea])
    theta_kin_mrad = 1000.0 * np.average(theta_kin_per, weights=wgt)

    # Ultra-relativistic reference
    theta_ur_mrad = 1000.0 * np.pi * ma_MeV / mean_Ea

    p41_list, p42_list, mc_wgts = gen.simulate_decay_4vectors(
        days_exposure=EXPOSURE_DAYS, n_samples=400)

    mc_wgts = np.asarray(mc_wgts)
    if mc_wgts.sum() <= 0 or len(p41_list) == 0:
        return None, theta_kin_mrad, theta_ur_mrad, mean_Ea, 0

    thetas = np.zeros(len(p41_list))
    for i, (p1, p2) in enumerate(zip(p41_list, p42_list)):
        v1 = np.array([p1.p1, p1.p2, p1.p3])
        v2 = np.array([p2.p1, p2.p2, p2.p3])
        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)
        if m1 > 1e-12 and m2 > 1e-12:
            cos_t = np.dot(v1, v2) / (m1 * m2)
            thetas[i] = np.arccos(np.clip(cos_t, -1.0, 1.0))

    theta_mc_mrad = 1000.0 * np.average(thetas, weights=mc_wgts)

    return (theta_mc_mrad, theta_kin_mrad, theta_ur_mrad,
            mean_Ea, int(np.count_nonzero(wgt)))


# ──────────────────────────────────────────────────────────────────────────────
# Coupling scaling test — in the small-g regime where surv_prob ≈ 1
# ──────────────────────────────────────────────────────────────────────────────

def check_coupling_scaling(photon_flux, ma_MeV=10.0):
    """
    Verify n_signal ∝ g⁴ (production g² × decay-in-detector g²) in the
    short-baseline regime where ALPs don't decay prematurely.

    Uses ma = 10 MeV (light ALP, long lifetime) and small g values so that
    surv_prob ≈ 1 and decay_prob ∝ g².
    """
    print(f"\n--- Coupling scaling check (mₐ={ma_MeV} MeV, small-g regime) ---")

    # Very small couplings so decay length ≫ det_dist
    couplings = np.array([5e-7, 1e-6, 2e-6, 4e-6])   # GeV⁻¹
    signals = []
    for g in couplings:
        flux_obj, gen = run_alplib(photon_flux, ma_MeV, g, n_samples=200)
        n = gen.decays(days_exposure=1.0, threshold=1.0)
        signals.append(n)
        print(f"  g={g:.1e} GeV⁻¹ → {n:.3e} events/day")

    signals = np.array(signals)

    if signals[0] <= 0:
        print("  No signal at baseline g — cannot test scaling")
        return

    # Each doubling should give ~16×
    ratios = signals[1:] / signals[:-1]
    print(f"  Doubling ratios (expect ≈ 16): "
          + ", ".join(f"{r:.2f}" for r in ratios))

    n_pass = int(np.sum((ratios > 10.0) & (ratios < 22.0)))
    status = "PASS" if n_pass == len(ratios) else "FAIL"
    print(f"  {status} ({n_pass}/{len(ratios)} ratios in [10, 22])")


# ──────────────────────────────────────────────────────────────────────────────
# Invariant mass reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def check_invariant_mass(photon_flux, ma_MeV=100.0):
    """Reconstruct m_γγ from decay 4-vectors — must equal mₐ to MC precision."""
    print(f"\n--- Invariant mass check (mₐ={ma_MeV} MeV) ---")
    g_safe = pick_safe_coupling(ma_MeV, photon_flux, target_decay_length_m=5.0)
    flux_obj, gen = run_alplib(photon_flux, ma_MeV, g_safe, n_samples=5)

    if len(flux_obj.axion_energy) == 0:
        print("  No events generated, skipping.")
        return

    p41_list, p42_list, _ = gen.simulate_decay_4vectors(
        days_exposure=1.0, n_samples=20)

    m_gg = []
    for p1, p2 in zip(p41_list, p42_list):
        E_tot = p1.energy() + p2.energy()
        px_tot = p1.p1 + p2.p1
        py_tot = p1.p2 + p2.p2
        pz_tot = p1.p3 + p2.p3
        m2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
        if m2 >= 0:
            m_gg.append(np.sqrt(m2))

    if m_gg:
        m_arr = np.array(m_gg)
        frac_dev = abs(m_arr.mean() - ma_MeV) / ma_MeV
        status = "PASS" if frac_dev < 0.05 else "FAIL"
        print(f"  m_γγ: mean={m_arr.mean():.3f} MeV, "
              f"std={m_arr.std():.3f} MeV  (target {ma_MeV:.1f} MeV)  [{status}]")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux", default=None, help="Path to alplib_brems_flux.csv")
    parser.add_argument("--nprimaries", type=int, default=10000)
    parser.add_argument("--analytic", action="store_true",
                        help="Force analytic Bethe-Heitler spectrum")
    args = parser.parse_args()

    if args.flux and not args.analytic:
        photon_flux = load_geant4_brems_flux(args.flux, args.nprimaries)
    else:
        print("[spectrum] Using analytic Bethe-Heitler spectrum")
        photon_flux = bethe_heitler_spectrum(1.0, E0_MeV=BEAM_ENERGY_MEV)

    rate_mean_E = np.average(photon_flux[:, 0], weights=photon_flux[:, 1])
    # Hard-tail mean (photons > 100 MeV) for physical intuition
    hard = photon_flux[:, 0] > 100.0
    if hard.sum() > 0:
        hard_mean_E = np.average(photon_flux[hard, 0], weights=photon_flux[hard, 1])
    else:
        hard_mean_E = 0.0
    print(f"[spectrum] Rate-weighted mean Eγ: {rate_mean_E:.1f} MeV"
          f"  (hard tail > 100 MeV: {hard_mean_E:.1f} MeV)")
    print(f"[spectrum] Note: soft photons dominate by count, but only hard")
    print(f"[spectrum] photons with Eγ > mₐ contribute to ALP production.\n")

    # ── Internal consistency: θ_MC vs exact kinematic expectation ───────────
    print("Opening angle internal consistency check")
    print("(auto-selected coupling so ALPs survive to detector;")
    print(" θ_kin = exact ⟨θ_open⟩ from per-ALP integration over rest-frame;")
    print(" θ_UR  = ultra-relativistic reference π mₐ/⟨Eₐ⟩)")
    print(f"{'mₐ (MeV)':>10} {'⟨Eₐ⟩_dec':>11} {'θ_kin (mrad)':>14}"
          f" {'θ_MC (mrad)':>13} {'θ_UR (mrad)':>13}"
          f" {'MC/kin':>8} {'N_alp':>7}")
    print("-" * 80)

    theta_kin_vals = []
    theta_mc_vals  = []
    theta_ur_vals  = []

    for ma in MASS_GRID_MEV:
        theta_mc, theta_kin, theta_ur, mean_Ea, n_alp = \
            opening_angle_internal_check(photon_flux, ma)
        theta_kin_vals.append(theta_kin)
        theta_mc_vals.append(theta_mc)
        theta_ur_vals.append(theta_ur)

        if theta_mc is None or theta_kin is None:
            print(f"{ma:10.1f} {'—':>11} {'—':>14}"
                  f" {'N/A':>13} {'—':>13} {'—':>8} {n_alp:>7}")
            continue

        ratio = theta_mc / theta_kin
        status = "OK" if 0.9 < ratio < 1.1 else "WARN"
        print(f"{ma:10.1f} {mean_Ea:11.1f} {theta_kin:14.2f}"
              f" {theta_mc:13.2f} {theta_ur:13.2f}"
              f" {ratio:8.3f}  [{status}]  {n_alp:>7}")

    # ── Coupling scaling ──────────────────────────────────────────────────────
    check_coupling_scaling(photon_flux, ma_MeV=10.0)

    # ── Invariant mass ────────────────────────────────────────────────────────
    check_invariant_mass(photon_flux, ma_MeV=100.0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if PLOT:
        Path("plots").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        mk  = [i for i, v in enumerate(theta_kin_vals) if v]
        mmc = [i for i, v in enumerate(theta_mc_vals)  if v]
        mur = [i for i, v in enumerate(theta_ur_vals)  if v]
        if mk:
            ax.plot(MASS_GRID_MEV[mk], [theta_kin_vals[i] for i in mk],
                    'o--', color='royalblue', label='Exact kinematic ⟨θ⟩')
        if mmc:
            ax.plot(MASS_GRID_MEV[mmc], [theta_mc_vals[i] for i in mmc],
                    's-', color='crimson', label='alplib MC 4-vec decay')
        if mur:
            ax.plot(MASS_GRID_MEV[mur], [theta_ur_vals[i] for i in mur],
                    '^:', color='gray', alpha=0.7, label='UR limit π mₐ/⟨Eₐ⟩')

        ax.set_xlabel("mₐ (MeV)")
        ax.set_ylabel("Mean opening angle (mrad)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Opening angle internal consistency")
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig("plots/opening_angle_validation.pdf")
        fig.savefig("plots/opening_angle_validation.png", dpi=150)
        plt.close(fig)
        print("\nValidation plot → plots/opening_angle_validation.pdf")

    print("\n=== Validation complete ===")


if __name__ == "__main__":
    main()
