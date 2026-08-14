#!/usr/bin/env python3
"""
Fast Pareto scan for DAMSA gap optimization (no Geant4 re-runs).

Background vs gap:
  Propagates each of the 2.3M particles recorded at the target exit
  in a straight line to the calorimeter face (12x12 cm) for each gap.
  Weighted background = Σ(weight × norm_factor) for photons + 10×neutrons.

Signal vs gap:
  Generates ALP decay events once with alplib (Primakoff flux from target-exit
  photon flux). For each gap, samples decay positions from the exponential
  lifetime distribution and checks geometric acceptance on the calo face.
  Separability criterion: opening angle ≥ angle_cut_deg  OR
                          both photon energies ≥ energy_cut_MeV.

Produces:
  output/pareto_scan.csv          — per-gap table
  plots/pareto_front.png          — Pareto front (main slide figure)
  plots/pareto_gap_curves.png     — signal & background vs gap separately

Usage:
  python scripts/fast_pareto_scan.py --ma-list 5 10 20 --n-samples 20000
"""

import argparse

# Delivered LESA-Laser beam current in uA (plan.md §1): 4200 e/bunch x 18 bunches x 929 kHz x e.
LESA_DELIVERED_UA = 4200 * 18 * 929e3 * 1.602176634e-19 * 1e6

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT = True
except ImportError:
    PLOT = False

# ─── Geometry (must match construction.cpp) ──────────────────────────────────
# Layout: Target → VDC (variable) → Magnet (fixed 12 cm) → Calo
# zPos starts at -50 cm; target occupies -50→-40 cm (10 cm thick).
# "gap" = VDC length only.  Calo entrance = target_exit + VDC + magnet.
TARGET_EXIT_Z_MM   = -400.0   # target rear face z, mm
MAGNET_LENGTH_MM   =  120.0   # fixed magnet length = 12 cm
CALO_HALF_WIDTH_MM =   60.0   # calorimeter half-width = 6 cm

CHARGE_C           = 1.602176634e-19
SECONDS_PER_DAY    = 86400.0


# ─── Background: analytic propagation ────────────────────────────────────────

def propagate_background(particles_df: pd.DataFrame,
                         gaps_cm: np.ndarray,
                         norm_factor: float,
                         neutron_weight: float = 10.0,
                         calo_half_width_mm: float = CALO_HALF_WIDTH_MM) -> np.ndarray:
    """
    For each gap, straight-line-propagate target-exit particles to the calo
    face and count weighted background (photons + neutron_weight × neutrons).
    """
    # Select photons and neutrons only
    photon_mask  = particles_df['pdg'].values == 22
    neutron_mask = particles_df['pdg'].values == 2112
    keep_mask    = photon_mask | neutron_mask

    df = particles_df[keep_mask].reset_index(drop=True)
    pdg  = df['pdg'].values
    x    = df['x_mm'].values
    y    = df['y_mm'].values
    z    = df['z_mm'].values        # should be ~ TARGET_EXIT_Z_MM
    px   = df['px'].values          # unit direction
    py   = df['py'].values
    pz   = df['pz'].values
    w_mc = df['weight'].values      # MC weight (1 per bkg particle in EB mode)

    # Neutrons get extra weight
    species_weight = np.where(pdg == 2112, neutron_weight, 1.0)

    bkg_vs_gap = np.zeros(len(gaps_cm))

    for i, gap_cm in enumerate(gaps_cm):
        # Calo entrance = target exit + VDC (gap) + magnet (fixed)
        z_calo_mm = TARGET_EXIT_Z_MM + gap_cm * 10.0 + MAGNET_LENGTH_MM   # mm

        # Straight-line propagation: dt = (z_calo - z) / pz
        # (pz must be > 0 for forward particles)
        fwd = pz > 0
        dt = np.where(fwd, (z_calo_mm - z) / np.where(fwd, pz, 1.0), np.inf)

        x_calo = x + dt * px
        y_calo = y + dt * py

        hits = (np.abs(x_calo) <= calo_half_width_mm) & \
               (np.abs(y_calo) <= calo_half_width_mm) & fwd

        # Sum weighted: MC weight × species weight × per-primary normalisation
        bkg_vs_gap[i] = np.sum(w_mc[hits] * species_weight[hits]) * norm_factor

    return bkg_vs_gap


# ─── Signal: alplib + geometric acceptance ───────────────────────────────────

def auto_coupling(flux_array: np.ndarray, ma_MeV: float,
                  target_decay_length_m: float = 0.6) -> float:
    """
    Pick g (GeV⁻¹) so the mean boosted decay length ≈ target_decay_length_m.
    This ensures ALPs actually decay in the gap (20–100 cm) rather than
    flying through the whole detector — making the geometric acceptance
    calculation meaningful regardless of absolute physics coupling.

    target_decay_length_m=0.6 m is the middle of the scan range (60 cm).
    """
    from alplib.constants import METER_BY_MEV
    E = flux_array[:, 0]
    w = flux_array[:, 1]
    mask = E > ma_MeV
    if mask.sum() == 0:
        return 1e-4
    Ea_typ = np.average(E[mask], weights=w[mask])
    g_sq_MeV = (64.0 * np.pi * METER_BY_MEV * Ea_typ
                / (target_decay_length_m * ma_MeV**3))
    g_MeV = np.sqrt(max(g_sq_MeV, 0.0))
    return g_MeV * 1000.0   # → GeV⁻¹


def run_alplib_once(flux_file: str, n_primaries: int, beam_current_uA: float,
                    ma_MeV: float, coupling_GeV: float,
                    exposure_days: float, n_samples: int,
                    nominal_gap_cm: float = 47.0):
    """
    Generate ALP decay events once with alplib. Returns list of event dicts.
    """
    from scripts.alplib.alplib_signal_plots import load_flux_for_alplib
    import alplib.fluxes as af
    import alplib.materials as am
    import alplib.generators as ag
    from alplib.decay import W_gg
    from alplib.constants import HBAR, C_LIGHT

    flux_array = load_flux_for_alplib(flux_file, n_primaries, beam_current_uA)

    # Auto-select coupling so mean decay length ≈ mid-range gap (60 cm),
    # overriding only if caller explicitly passed coupling_GeV <= 0.
    if coupling_GeV <= 0:
        coupling_GeV = auto_coupling(flux_array, ma_MeV, target_decay_length_m=0.6)
        print(f"  [alplib] Auto-coupling for ma={ma_MeV} MeV: "
              f"g = {coupling_GeV:.3e} GeV⁻¹")

    coupling_MeV = coupling_GeV / 1000.0

    max_E = flux_array[:, 0].max()
    if ma_MeV >= max_E:
        print(f"  [alplib] ma={ma_MeV} MeV >= max photon E={max_E:.1f} MeV — no production")
        return [], 0.0

    det_dist_m  = nominal_gap_cm / 100.0
    det_length_m = nominal_gap_cm / 100.0
    det_area    = 0.12 * 0.12   # 12 cm × 12 cm

    flux_obj = af.FluxPrimakoffIsotropic(
        photon_flux   = flux_array,
        target        = am.Material("W"),
        det_dist      = det_dist_m,
        det_length    = det_length_m,
        det_area      = det_area,
        axion_mass    = ma_MeV,
        axion_coupling= coupling_MeV,
        n_samples     = n_samples,
    )
    flux_obj.simulate()

    if len(flux_obj.axion_energy) == 0:
        print(f"  [alplib] No ALP energy bins produced")
        return [], 0.0

    decay_width = W_gg(coupling_MeV, ma_MeV)
    tau_rest    = HBAR / decay_width          # seconds

    flux_obj.propagate(decay_width)
    generator = ag.PhotonEventGenerator(flux_obj, am.Material("CsI"))
    n_events  = generator.decays(days_exposure=exposure_days, threshold=0.1)

    p4_1, p4_2, weights = generator.simulate_decay_4vectors(
        days_exposure=exposure_days, n_samples=n_samples
    )

    # alplib returns a FLAT list of n_samples 4-vectors (not per energy bin).
    # Derive kinematics directly from the 4-vectors to avoid the per-bin
    # indexing bug in the original script.
    events = []
    for idx in range(len(p4_1)):
        w = weights[idx] if idx < len(weights) else 0.0
        if w <= 0:
            continue

        p1 = p4_1[idx];  p2 = p4_2[idx]
        E1, px1, py1, pz1 = p1.p0, p1.p1, p1.p2, p1.p3
        E2, px2, py2, pz2 = p2.p0, p2.p1, p2.p2, p2.p3
        if E1 <= 0 or E2 <= 0:
            continue

        # ALP 4-momentum by conservation
        E_alp = E1 + E2
        if E_alp <= ma_MeV:
            continue

        # True opening angle from 3-momenta dot product
        mag1 = np.sqrt(px1**2 + py1**2 + pz1**2)
        mag2 = np.sqrt(px2**2 + py2**2 + pz2**2)
        if mag1 < 1e-12 or mag2 < 1e-12:
            continue
        cos_theta = (px1*px2 + py1*py2 + pz1*pz2) / (mag1 * mag2)
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta_deg = np.degrees(np.arccos(cos_theta))

        gamma = E_alp / ma_MeV
        beta  = np.sqrt(max(1.0 - (ma_MeV / E_alp)**2, 0.0))
        lab_decay_length_cm = gamma * beta * C_LIGHT * tau_rest   # cm

        events.append({
            'weight':              w,
            'alp_E_MeV':           E_alp,
            'E1_MeV':              E1,
            'E2_MeV':              E2,
            'opening_angle_deg':   theta_deg,
            'lab_decay_length_cm': lab_decay_length_cm,
        })

    print(f"  [alplib] Built {len(events)} decay samples  "
          f"(n_events_exposure={n_events:.3g})")
    return events, n_events


def geometric_acceptance_vs_gap(events, gaps_cm,
                                angle_cut_deg=10.0, energy_cut_MeV=100.0,
                                calo_half_width_cm=6.0, rng=None):
    """
    For each gap, MC-sample decay positions and compute:
      accepted_fraction  = P(both photons hit calo)
      separable_fraction = P(hits AND separable)
    Returns two arrays, each shape (len(gaps_cm),).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    weights   = np.array([e['weight']              for e in events])
    theta_deg = np.array([e['opening_angle_deg']   for e in events])
    E1        = np.array([e['E1_MeV']              for e in events])
    E2        = np.array([e['E2_MeV']              for e in events])
    L_decay   = np.array([e['lab_decay_length_cm'] for e in events])

    total_w = weights.sum()
    if total_w <= 0:
        return np.zeros(len(gaps_cm)), np.zeros(len(gaps_cm))

    # Separability mask (independent of gap)
    sep_mask = (theta_deg >= angle_cut_deg) | \
               ((theta_deg < angle_cut_deg) & (E1 >= energy_cut_MeV) & (E2 >= energy_cut_MeV))

    accepted_frac  = np.zeros(len(gaps_cm))
    separable_frac = np.zeros(len(gaps_cm))

    for i, gap_cm in enumerate(gaps_cm):
        # Sample decay z positions for every event
        u = rng.random(len(events))
        z_decay = -L_decay * np.log(np.clip(1.0 - u, 1e-30, None))  # cm from target exit

        # Decay must be inside the VDC (before magnet entrance)
        in_gap = (z_decay > 0) & (z_decay < gap_cm)

        # Distance from decay point to calo entrance.
        # Calo is at (VDC_length + magnet_length) past target exit.
        magnet_cm    = MAGNET_LENGTH_MM / 10.0
        dist_to_calo = (gap_cm + magnet_cm) - z_decay   # cm (always > 0 when in_gap)

        # Photon half-separation at calo face (symmetric opening angle)
        theta_rad   = np.radians(theta_deg)
        half_sep_cm = theta_rad * dist_to_calo / 2.0

        both_hit = in_gap & (half_sep_cm <= calo_half_width_cm)

        acc_w  = (weights * both_hit).sum()
        sep_w  = (weights * both_hit * sep_mask).sum()

        accepted_frac[i]  = acc_w  / total_w
        separable_frac[i] = sep_w  / total_w

    return accepted_frac, separable_frac


# ─── Pareto helpers ──────────────────────────────────────────────────────────

def pareto_mask(x_min, y_max):
    """
    Identify Pareto-optimal points for minimising x and maximising y.
    A point is dominated if another has lower-or-equal x AND higher-or-equal y.
    """
    n = len(x_min)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x_min[j] <= x_min[i] and y_max[j] >= y_max[i] and \
               (x_min[j] < x_min[i] or y_max[j] > y_max[i]):
                is_pareto[i] = False
                break
    return is_pareto


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_pareto_front(df: pd.DataFrame, ma_list, plot_dir: Path,
                      beam_uA: float, exposure_days: float):
    if not PLOT:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(ma_list)))
    fig, ax = plt.subplots(figsize=(8, 6))

    for ma, col in zip(ma_list, colors):
        sub = df[df['ma_MeV'] == ma].copy()
        if sub.empty:
            continue

        bkg  = sub['bkg_exposure'].values
        sig  = sub['separable_fraction'].values
        gaps = sub['gap_cm'].values

        # Normalise background to [0, 1] relative to minimum gap (worst case)
        # for display; label shows actual values in a colour bar or annotation.
        ax.scatter(bkg, sig, c=[col]*len(sub), zorder=5, s=60,
                   label=f'ma={ma} MeV')
        ax.plot(bkg, sig, '-', color=col, alpha=0.4, linewidth=1)

        # Label Pareto points
        pm = pareto_mask(bkg, sig)
        ax.scatter(bkg[pm], sig[pm], c=[col]*pm.sum(),
                   edgecolors='black', linewidths=1.5, zorder=6, s=80)

        # Annotate a few gap values
        for idx in range(0, len(gaps), max(1, len(gaps)//4)):
            ax.annotate(f'{gaps[idx]:.0f}cm',
                        (bkg[idx], sig[idx]),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=7, color=col)

    ax.set_xlabel('Weighted background at calo face (photons + 10×n, full exposure)', fontsize=11)
    ax.set_ylabel('Signal separable fraction', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(
        f'DAMSA gap Pareto front  —  '
        f'I_avg={beam_uA} µA, T={exposure_days:.0f} d\n'
        f'Separability: Δθ ≥ 10° or both E ≥ 100 MeV  |  '
        f'Pareto-optimal points outlined in black',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = plot_dir / 'pareto_front.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_gap_curves(df: pd.DataFrame, ma_list, plot_dir: Path,
                    beam_uA: float, exposure_days: float):
    if not PLOT:
        return

    # Only keep masses that have nonzero separable signal
    active_ma = [ma for ma in ma_list
                 if df[df['ma_MeV'] == ma]['separable_fraction'].max() > 0]
    if not active_ma:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(active_ma)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_sig, ax_bkg, ax_fom = axes

    # ── Background (identical for all masses, normalise to gap_min) ──────────
    ref_sub = df[df['ma_MeV'] == active_ma[0]].copy()
    bkg_abs  = ref_sub['bkg_exposure'].values
    bkg_ref  = bkg_abs[0]
    bkg_norm = bkg_abs / bkg_ref * 100.0   # percentage of gap_min background
    ax_bkg.plot(ref_sub['gap_cm'], bkg_norm, 'o-', color='steelblue', linewidth=2)
    ax_bkg.axhline(100, color='grey', linestyle=':', linewidth=1)
    ax_bkg.set_xlabel('Gap [cm]', fontsize=11)
    ax_bkg.set_ylabel('Background at calo face\n(% of gap=30 cm level)', fontsize=11)
    ax_bkg.set_title('Background reduction vs gap\n(brem photons + 10×neutrons, Geant4)', fontsize=10)
    ax_bkg.set_yscale('log')
    ax_bkg.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax_bkg.grid(True, alpha=0.3, which='both')

    for ma, col in zip(active_ma, colors):
        sub  = df[df['ma_MeV'] == ma].copy()
        gaps = sub['gap_cm'].values
        sig  = sub['separable_fraction'].values
        bkg  = sub['bkg_exposure'].values

        # Normalise to peak (gap=20 cm) so all masses visible on same axis
        sig_ref  = max(sig[0], 1e-30)
        sig_norm = sig / sig_ref * 100.0

        ax_sig.plot(gaps, sig_norm, 'o-', color=col, label=f'ma={ma:.0f} MeV')

        # FoM: signal retained / background retained (both normalised to gap=20)
        # = (sig/sig0) / (bkg/bkg0)^0.5  — reward bkg reduction
        fom = (sig / sig_ref) / np.sqrt(bkg / bkg_ref + 1e-30)
        fom_norm = fom / max(fom.max(), 1e-30)
        ax_fom.plot(gaps, fom_norm, 'o-', color=col, label=f'ma={ma:.0f} MeV')

        peak_idx = np.argmax(fom_norm)
        ax_fom.axvline(gaps[peak_idx], color=col, linestyle='--', alpha=0.35)
        ax_fom.annotate(f'{gaps[peak_idx]:.0f} cm',
                        (gaps[peak_idx], fom_norm[peak_idx]),
                        textcoords='offset points', xytext=(4, 2),
                        fontsize=8, color=col)

    ax_sig.axhline(100, color='grey', linestyle=':', linewidth=1)
    ax_sig.set_xlabel('Gap [cm]', fontsize=11)
    ax_sig.set_ylabel('Signal separable fraction\n(% of gap=20 cm level)', fontsize=11)
    ax_sig.set_title('Signal acceptance vs gap\n(separability: Δθ≥10° or both E≥100 MeV)', fontsize=10)
    ax_sig.set_yscale('log')
    ax_sig.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax_sig.legend(fontsize=9)
    ax_sig.grid(True, alpha=0.3, which='both')

    ax_fom.set_xlabel('Gap [cm]', fontsize=11)
    ax_fom.set_ylabel('S/√B  (normalised)', fontsize=11)
    ax_fom.set_title('Figure of merit S/√B vs gap\n(optimal = highest value)', fontsize=10)
    ax_fom.legend(fontsize=9)
    ax_fom.grid(True, alpha=0.3)

    fig.suptitle(
        f'DAMSA gap optimisation scan  —  I_avg={beam_uA} µA CW  |  T={exposure_days:.0f} d  '
        f'|  Separability: Δθ ≥ 20° or both E ≥ 100 MeV',
        fontsize=10
    )
    plt.tight_layout()
    out = plot_dir / 'pareto_gap_curves.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--particles-csv', default='output/all_particles_target_exit.csv')
    parser.add_argument('--flux-csv',      default='output/photon_flux_target_exit.csv')
    parser.add_argument('--n-primaries',   type=int,   default=100000)
    parser.add_argument('--beam-uA',       type=float, default=LESA_DELIVERED_UA)
    parser.add_argument('--exposure-days', type=float, default=30.0)
    parser.add_argument('--ma-list',       type=float, nargs='+', default=[5, 10, 20])
    parser.add_argument('--coupling',      type=float, default=-1,
                        help='ALP-photon coupling in GeV^-1. '
                             'Pass -1 (default) to auto-select per mass so '
                             'decay length ≈ mid-range gap (60 cm).')
    parser.add_argument('--n-samples',     type=int,   default=10000,
                        help='alplib MC samples per mass')
    parser.add_argument('--angle-cut',     type=float, default=20.0,
                        help='Opening angle separability cut [deg]')
    parser.add_argument('--energy-cut',    type=float, default=100.0,
                        help='Per-photon energy separability cut [MeV]')
    parser.add_argument('--gap-min',       type=float, default=30.0)
    parser.add_argument('--gap-max',       type=float, default=60.0)
    parser.add_argument('--gap-step',      type=float, default=2.0)
    parser.add_argument('--calo-half-width-cm', type=float, default=CALO_HALF_WIDTH_MM / 10.0,
                        help='Calorimeter half-width in cm (default: 6.0 = 12 cm square)')
    parser.add_argument('--out-csv',       default='output/pareto_scan.csv')
    parser.add_argument('--plot-dir',      default='plots/pareto')
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gaps_cm = np.arange(args.gap_min, args.gap_max + 0.1, args.gap_step)

    # ── Background ────────────────────────────────────────────────────────────
    print(f"\nLoading particles: {args.particles_csv}")
    particles_df = pd.read_csv(args.particles_csv)
    print(f"  {len(particles_df)} particles loaded")

    eps             = (args.beam_uA * 1e-6) / CHARGE_C
    exposure_s      = args.exposure_days * SECONDS_PER_DAY
    norm_factor     = (eps * exposure_s) / args.n_primaries
    print(f"  Beam: {args.beam_uA} µA → {eps:.3e} e/s")
    print(f"  Exposure: {args.exposure_days} d → {exposure_s:.3e} s")
    print(f"  Background norm factor: {norm_factor:.3e}")

    calo_hw_cm = args.calo_half_width_cm
    calo_hw_mm = calo_hw_cm * 10.0
    print(f"\nCalorimeter: {calo_hw_cm*2:.1f} cm × {calo_hw_cm*2:.1f} cm square face")
    print("\nPropagating background across gaps...")
    bkg_vs_gap = propagate_background(particles_df, gaps_cm, norm_factor,
                                      calo_half_width_mm=calo_hw_mm)
    print(f"  Gap range: {gaps_cm[0]:.0f}–{gaps_cm[-1]:.0f} cm")
    print(f"  Background range: {bkg_vs_gap.min():.3e} – {bkg_vs_gap.max():.3e}")

    # ── Signal per mass ───────────────────────────────────────────────────────
    all_rows = []

    for ma in args.ma_list:
        print(f"\n{'='*55}")
        print(f"ALP mass: ma = {ma} MeV,  g = {args.coupling:.1e} GeV^-1")
        print(f"{'='*55}")

        events, n_total = run_alplib_once(
            flux_file        = args.flux_csv,
            n_primaries      = args.n_primaries,
            beam_current_uA  = args.beam_uA,
            ma_MeV           = ma,
            coupling_GeV     = args.coupling,
            exposure_days    = args.exposure_days,
            n_samples        = args.n_samples,
            nominal_gap_cm   = float(np.mean(gaps_cm)),
        )

        if not events:
            print(f"  Skipping ma={ma}: no events generated")
            continue

        acc_frac, sep_frac = geometric_acceptance_vs_gap(
            events, gaps_cm,
            angle_cut_deg      = args.angle_cut,
            energy_cut_MeV     = args.energy_cut,
            calo_half_width_cm = calo_hw_cm,
            rng                = rng,
        )

        for i, gap_cm in enumerate(gaps_cm):
            acc_val = acc_frac[i]
            sep_eff = sep_frac[i] / acc_val if acc_val > 0 else 0.0
            all_rows.append({
                'ma_MeV':             ma,
                'gap_cm':             gap_cm,
                'bkg_exposure':       bkg_vs_gap[i],
                'accepted_fraction':  acc_val,
                'separable_fraction': sep_frac[i],
                'sep_efficiency':     sep_eff,
                'n_alp_events':       n_total,
            })

        print(f"  Accepted fraction:   {acc_frac.min():.3f} – {acc_frac.max():.3f}")
        print(f"  Separable fraction:  {sep_frac.min():.4f} – {sep_frac.max():.4f}")
        opt_idx = np.argmax(sep_frac / (bkg_vs_gap + 1))
        print(f"  Best gap (sep/bkg):  {gaps_cm[opt_idx]:.0f} cm")

    if not all_rows:
        print("No results to save.")
        return 1

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")
    print(df.to_string(index=False))

    plot_pareto_front(df, args.ma_list, plot_dir, args.beam_uA, args.exposure_days)
    plot_gap_curves(df,  args.ma_list, plot_dir, args.beam_uA, args.exposure_days)

    return 0


if __name__ == '__main__':
    sys.exit(main())
