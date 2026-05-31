#!/usr/bin/env python3
"""
ALP opening angle diagnostic plots.

Four diagnostics:
  1. Event-by-event printout (first N decays) — raw Ea, theta_MC, formula predictions
  2. Full opening angle distribution for ma=100 MeV with mean/prediction markers
  3. Correlation plot: theta_formula vs theta_MC per ALP (three formula variants)
  4. ALP energy (Ea) distribution for each mass point

Saves all output to the angle_validation/ folder (this directory).

Usage:
    python angle_diagnostics.py [--flux path/to/alplib_brems_flux.csv]
                                [--analytic]
                                [--nprimaries N]
                                [--nevents N]     # events to print (default: all)
                                [--nsamples N]    # MC samples per mass (default 5000)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# resolve project root (two levels up from scripts/analysis/angle_validation/)
_here = Path(__file__).resolve().parent
_project_root = _here.parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "scripts" / "pipeline"))

try:
    from alplib.fluxes import FluxPrimakoffIsotropic
    from alplib.materials import Material
    from alplib.generators import PhotonEventGenerator
    from alplib.constants import CHARGE_COULOMBS, S_PER_DAY, METER_BY_MEV
except ImportError:
    sys.exit("alplib not found.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from alp_signal_pipeline import (
    bethe_heitler_spectrum,
    load_geant4_brems_flux,
    run_alplib,
    BEAM_ENERGY_MEV, DET_DIST_M, DET_LENGTH_M, DET_AREA_M2, EXPOSURE_DAYS,
)

OUT_DIR = _here          # save everything alongside this script
MASS_GRID_MEV = np.array([10, 20, 50, 100, 200, 500])

# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared across diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def pick_safe_coupling(ma_MeV, photon_flux, target_decay_length_m=5.0):
    E = photon_flux[:, 0]
    w = photon_flux[:, 1]
    mask = E > ma_MeV
    if mask.sum() == 0:
        return 1e-4
    Ea_typ = np.average(E[mask], weights=w[mask])
    g_sq_MeV = (64.0 * np.pi * METER_BY_MEV * Ea_typ
                / (target_decay_length_m * ma_MeV**3))
    return np.sqrt(g_sq_MeV) * 1000.0   # GeV⁻¹


def get_4vec_angle(p1, p2):
    """Opening angle (rad) between two alplib 4-vector objects."""
    v1 = np.array([p1.p1, p1.p2, p1.p3])
    v2 = np.array([p2.p1, p2.p2, p2.p3])
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-12 or m2 < 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)))


def formula_angles(Ea_arr, ma):
    """
    Returns three formula arrays (rad) for an array of ALP energies:
      min_angle : 2*ma/Ea          (minimum opening angle, ur approx)
      half_angle: ma/Ea            (another common reference, = min/2)
      pi_angle  : pi*ma/Ea         (ur mean over isotropic decay, = pi/2 * min)
    """
    Ea = np.asarray(Ea_arr, dtype=float)
    safe = np.where(Ea > ma, Ea, np.nan)
    return (2.0 * ma / safe,
            ma / safe,
            np.pi * ma / safe)


def weighted_median(values, weights):
    """Weighted median via sorted cumulative weight. Excludes zero-weight entries."""
    values, weights = np.asarray(values), np.asarray(weights)
    mask = weights > 0
    if mask.sum() == 0:
        return np.nan
    v, w = values[mask], weights[mask]
    idx = np.argsort(v)
    v_s, w_s = v[idx], w[idx]
    cdf = np.cumsum(w_s) / w_s.sum()
    return float(v_s[np.searchsorted(cdf, 0.5)])


def run_and_collect(photon_flux, ma_MeV, n_samples=600):
    """
    Run alplib for one mass, return (Ea_arr, angles_rad, mc_wgts, flux_obj).
    Per-event Ea is matched to each decay by tiling the flux_obj.axion_energy
    over n_samples repetitions (alplib repeats the ALP list n_samples times).
    """
    coupling_GeV = pick_safe_coupling(ma_MeV, photon_flux)
    flux_obj, gen = run_alplib(photon_flux, ma_MeV, coupling_GeV, n_samples=20)

    if len(flux_obj.axion_energy) == 0:
        return None, None, None, None

    p41_list, p42_list, mc_wgts = gen.simulate_decay_4vectors(
        days_exposure=EXPOSURE_DAYS, n_samples=n_samples)

    mc_wgts = np.asarray(mc_wgts, dtype=float)
    if mc_wgts.sum() <= 0 or len(p41_list) == 0:
        return None, None, None, flux_obj

    # alplib loops: for each ALP generates exactly n_samples decays
    # → output order is [ALP_0 × n_samples, ALP_1 × n_samples, ...]
    Ea_base  = np.asarray(flux_obj.axion_energy, dtype=float)
    Ea_tiled = np.repeat(Ea_base, n_samples)

    n_events = len(p41_list)
    angles = np.array([get_4vec_angle(p41_list[i], p42_list[i])
                       for i in range(n_events)])

    return Ea_tiled, angles, mc_wgts, flux_obj


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic 1 — event-by-event printout
# ──────────────────────────────────────────────────────────────────────────────

def diag_event_printout(photon_flux, ma_MeV=100.0, n_print=10, n_samples=200):
    print(f"\n{'='*80}")
    _label = "All" if n_print is None else f"First {n_print}"
    print(f"Diagnostic 1 — {_label} ALP decays  (ma={ma_MeV} MeV)")
    print(f"{'='*80}")

    Ea_arr, angles, mc_wgts, flux_obj = run_and_collect(
        photon_flux, ma_MeV, n_samples=n_samples)

    if Ea_arr is None:
        print("  No events generated.")
        return

    th_min, th_half, th_pi = formula_angles(Ea_arr, ma_MeV)

    # header
    hdr = (f"{'#':>4}  {'Ea (MeV)':>10}  {'theta_MC (mrad)':>16}"
           f"  {'2ma/Ea (mrad)':>14}  {'ma/Ea (mrad)':>13}"
           f"  {'pi*ma/Ea (mrad)':>16}  {'wgt':>10}")
    print(hdr)
    print("-" * len(hdr))

    n_print_eff = len(Ea_arr) if n_print is None else n_print
    for i in range(min(n_print_eff, len(Ea_arr))):
        print(f"{i+1:4d}  {Ea_arr[i]:10.2f}  {1000*angles[i]:16.4f}"
              f"  {1000*th_min[i]:14.4f}  {1000*th_half[i]:13.4f}"
              f"  {1000*th_pi[i]:16.4f}  {mc_wgts[i]:10.3e}")

    # weighted means
    w = mc_wgts
    print("\nWeighted means over all events:")
    print(f"  theta_MC      = {1000*np.average(angles, weights=w):.4f} mrad")
    print(f"  2*ma/Ea       = {1000*np.nanmean(th_min * w / w.sum()):>8.4f} mrad")
    print(f"  ma/Ea         = {1000*np.nanmean(th_half* w / w.sum()):>8.4f} mrad")
    print(f"  pi*ma/Ea      = {1000*np.nanmean(th_pi  * w / w.sum()):>8.4f} mrad")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic 2 — opening angle full distribution (ma=100 MeV)
# ──────────────────────────────────────────────────────────────────────────────

def diag_angle_distribution(photon_flux, ma_MeV=100.0, n_samples=600):
    print(f"\n{'='*80}")
    print(f"Diagnostic 2 — Opening angle distribution  (ma={ma_MeV} MeV)")
    print(f"{'='*80}")

    Ea_arr, angles, mc_wgts, flux_obj = run_and_collect(
        photon_flux, ma_MeV, n_samples=n_samples)

    if Ea_arr is None:
        print("  No events.")
        return

    angles_deg = np.degrees(angles)
    w = mc_wgts

    # Weighted statistics
    mean_mc   = np.average(angles_deg, weights=w)
    median_mc = weighted_median(angles_deg, w)
    # exact kinematic prediction (ensemble weighted mean)
    mean_Ea   = np.average(Ea_arr, weights=w)
    pred_min  = np.degrees(2.0 * ma_MeV / mean_Ea)    # 2*ma/<Ea>
    pred_half = np.degrees(ma_MeV / mean_Ea)           # ma/<Ea>
    pred_pi   = np.degrees(np.pi * ma_MeV / mean_Ea)  # pi*ma/<Ea>

    print(f"  Weighted mean   theta_MC = {mean_mc:.4f} deg")
    print(f"  Weighted median theta_MC = {median_mc:.4f} deg")
    print(f"  <Ea> (weighted)          = {mean_Ea:.2f} MeV")
    print(f"  2*ma/<Ea>                = {pred_min:.4f} deg")
    print(f"  ma/<Ea>                  = {pred_half:.4f} deg")
    print(f"  pi*ma/<Ea>               = {pred_pi:.4f} deg")

    # Per-event formula predictions (for skewness insight)
    th_min_arr, th_half_arr, th_pi_arr = formula_angles(Ea_arr, ma_MeV)
    valid = np.isfinite(th_min_arr)
    print(f"  Weighted mean 2*ma/Ea    = {np.degrees(np.average(th_min_arr[valid], weights=w[valid])):.4f} deg")
    print(f"  Weighted mean pi*ma/Ea   = {np.degrees(np.average(th_pi_arr[valid],  weights=w[valid])):.4f} deg")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, max(angles_deg.max() * 1.1, 5), 70)
    counts, edges = np.histogram(angles_deg, bins=bins, weights=w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]

    ax.bar(centers, counts / bw, width=bw, color='steelblue', alpha=0.7,
           label='alplib MC (weighted)')

    ax.axvline(mean_mc,   color='crimson',   lw=2,
               label=f'MC mean = {mean_mc:.3f}°')
    ax.axvline(median_mc, color='royalblue', lw=2, ls=(0, (3, 1, 1, 1)),
               label=f'MC median = {median_mc:.3f}°')
    ax.axvline(pred_min,  color='darkorange', lw=2, ls='--',
               label=f'2mₐ/⟨Eₐ⟩ = {pred_min:.3f}°')
    ax.axvline(pred_half, color='forestgreen', lw=2, ls=':',
               label=f'mₐ/⟨Eₐ⟩ = {pred_half:.3f}°')
    ax.axvline(pred_pi,   color='purple',  lw=2, ls='-.',
               label=f'π·mₐ/⟨Eₐ⟩ = {pred_pi:.3f}°')

    # skewness annotation
    from scipy.stats import skew as _skew
    sk = _skew(angles_deg)
    ax.text(0.97, 0.97, f'skewness = {sk:.3f}\nmean = {mean_mc:.3f}°\nmedian = {median_mc:.3f}°',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(0, 90)
    ax.set_xlabel('Opening angle θ [degrees]', fontsize=12)
    ax.set_ylabel('Weighted events / degree', fontsize=12)
    ax.set_title(f'ALP→γγ Opening Angle Distribution  (mₐ = {ma_MeV} MeV)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / f"diag2_angle_dist_ma{int(ma_MeV)}MeV.png"
    fig.savefig(out, dpi=180, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic 3 — correlation: per-ALP theta_formula vs theta_MC
# ──────────────────────────────────────────────────────────────────────────────

def diag_correlation(photon_flux, ma_MeV=100.0, n_samples=600):
    print(f"\n{'='*80}")
    print(f"Diagnostic 3 — θ_MC vs Eₐ with formula overlays  (ma={ma_MeV} MeV)")
    print(f"{'='*80}")

    Ea_arr, angles, mc_wgts, flux_obj = run_and_collect(
        photon_flux, ma_MeV, n_samples=n_samples)

    if Ea_arr is None:
        print("  No events.")
        return

    # Filter: valid Ea > ma (formula defined) and positive weight
    valid = (Ea_arr > ma_MeV) & np.isfinite(angles)
    Ea_v   = Ea_arr[valid]
    ang_v  = np.degrees(angles[valid])
    w_v    = mc_wgts[valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter: per-event (Ea, theta_MC), colored by weight
    sc = ax.scatter(Ea_v, ang_v, c=np.log10(w_v + 1e-30),
                    cmap='plasma', s=3, alpha=0.35, rasterized=True,
                    label='θ_MC per decay')
    plt.colorbar(sc, ax=ax, label='log₁₀(weight)')

    # Smooth formula curves over Ea range
    Ea_line = np.linspace(Ea_v.min(), Ea_v.max(), 500)
    formula_defs = [
        ('2mₐ/Eₐ',   np.degrees(2.0      * ma_MeV / Ea_line), 'darkorange'),
        ('mₐ/Eₐ',    np.degrees(          ma_MeV / Ea_line), 'forestgreen'),
        ('π·mₐ/Eₐ',  np.degrees(np.pi    * ma_MeV / Ea_line), 'purple'),
    ]
    for lab, theta_line, col in formula_defs:
        ax.plot(Ea_line, theta_line, '--', color=col, lw=2.0, label=lab)

    # Print weighted mean formula vs MC for reference
    th_min, th_half, th_pi = formula_angles(Ea_v, ma_MeV)
    for lab, th_arr in [('2mₐ/Eₐ', th_min), ('mₐ/Eₐ', th_half), ('π·mₐ/Eₐ', th_pi)]:
        frac_resid = (ang_v - np.degrees(th_arr)) / np.where(ang_v > 0, ang_v, np.nan)
        rms_frac = float(np.sqrt(np.nanmean(frac_resid**2)))
        print(f"  {lab:12s}  RMS frac. resid from MC = {rms_frac:.4f}")

    ax.set_xscale('linear')
    ax.set_xlabel('Eₐ [MeV]', fontsize=12)
    ax.set_ylabel('Opening angle θ [degrees]', fontsize=12)
    ax.set_title(f'θ_MC vs Eₐ with formula predictions  (mₐ = {ma_MeV} MeV)',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / f"diag3_correlation_ma{int(ma_MeV)}MeV.png"
    fig.savefig(out, dpi=180, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic 4 — Ea distribution for each mass point
# ──────────────────────────────────────────────────────────────────────────────

def diag_energy_distributions(photon_flux, n_samples=400):
    print(f"\n{'='*80}")
    print("Diagnostic 4 — ALP energy (Ea) distributions")
    print(f"{'='*80}")

    colors = {10: '#e41a1c', 20: '#377eb8', 50: '#4daf4a',
              100: '#984ea3', 200: '#ff7f00', 500: '#a65628'}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for idx, ma_MeV in enumerate(MASS_GRID_MEV):
        ax = axes_flat[idx]
        Ea_arr, angles, mc_wgts, flux_obj = run_and_collect(
            photon_flux, ma_MeV, n_samples=n_samples)

        if Ea_arr is None:
            ax.text(0.5, 0.5, 'No events', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_title(f'mₐ = {ma_MeV} MeV', fontsize=11)
            continue

        w = mc_wgts
        mean_Ea   = np.average(Ea_arr, weights=w)
        median_Ea = weighted_median(Ea_arr, w)

        col = colors.get(ma_MeV, 'gray')
        bins = np.logspace(np.log10(max(Ea_arr.min(), 1.0)),
                           np.log10(Ea_arr.max()), 50)
        counts, edges = np.histogram(Ea_arr, bins=bins, weights=w)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bw = edges[1:] - edges[:-1]

        ax.bar(centers, counts / bw, width=bw, color=col, alpha=0.75,
               align='center')
        ax.axvline(mean_Ea,   color='crimson',  ls='--',              lw=1.5,
                   label=f'mean = {mean_Ea:.0f} MeV')
        ax.axvline(median_Ea, color='navy',     ls=(0, (3, 1, 1, 1)), lw=1.5,
                   label=f'median = {median_Ea:.0f} MeV')

        ax.set_xscale('log')
        ax.set_xlabel('Eₐ [MeV]', fontsize=10)
        ax.set_ylabel('Weighted events / MeV', fontsize=9)
        ax.set_title(f'mₐ = {ma_MeV} MeV', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

        boost_mean = mean_Ea / ma_MeV
        print(f"  ma={ma_MeV:5.0f} MeV  <Ea>={mean_Ea:8.1f} MeV  "
              f"median_Ea={median_Ea:8.1f} MeV  "
              f"<gamma>={boost_mean:.1f}  N_decays={len(Ea_arr)}")

    fig.suptitle(f'ALP energy distributions  ({BEAM_ENERGY_MEV/1000:.0f} GeV beam)',
                 fontsize=14)
    fig.tight_layout()

    out = OUT_DIR / "diag4_Ea_distributions.png"
    fig.savefig(out, dpi=180, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux",       default=None)
    parser.add_argument("--analytic",   action="store_true")
    parser.add_argument("--nprimaries", type=int,   default=10000)
    parser.add_argument("--nevents",    type=int,   default=None,
                        help="events to print in Diag 1 (default: all)")
    parser.add_argument("--nsamples",   type=int,   default=5000,
                        help="MC decay samples per mass (default 5000)")
    parser.add_argument("--mass",       type=float, default=100.0,
                        help="ALP mass for diag 2+3 (default 100 MeV)")
    args = parser.parse_args()

    # ── Flux ──────────────────────────────────────────────────────────────────
    if args.flux and not args.analytic:
        photon_flux = load_geant4_brems_flux(args.flux, args.nprimaries)
    else:
        print("[spectrum] Using analytic Bethe-Heitler spectrum")
        photon_flux = bethe_heitler_spectrum(1.0, E0_MeV=BEAM_ENERGY_MEV)

    # ── Run all diagnostics ───────────────────────────────────────────────────
    diag_event_printout(photon_flux, ma_MeV=args.mass,
                        n_print=args.nevents, n_samples=args.nsamples)
    diag_angle_distribution(photon_flux, ma_MeV=args.mass,
                            n_samples=args.nsamples)
    diag_correlation(photon_flux, ma_MeV=args.mass, n_samples=args.nsamples)
    diag_energy_distributions(photon_flux, n_samples=args.nsamples)

    print(f"\n=== All diagnostics complete. Output in: {OUT_DIR} ===")


if __name__ == "__main__":
    main()
