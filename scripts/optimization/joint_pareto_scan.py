#!/usr/bin/env python3
"""
Joint analytic scan of VDC length × calorimeter XY size for DAMSA.

No new Geant4 runs required — re-uses a fixed target-exit particle CSV and
bremsstrahlung flux CSV.  Computes background by straight-line propagation
and ALP signal by re-sampling alplib decay photons with the appropriate
geometric acceptance cut for each (VDC, calo) combination.

Geometry:
    Target → VDC (variable, 'gap') → Magnet (12 cm fixed) → Calorimeter
    fCaloEntranceZ = TARGET_EXIT_Z + VDC + MAGNET

The 'gap' throughout this script refers to VDC length only (not total gap).

Outputs (under --output-dir):
    <label>_grid.csv         — full (vdc, calo, mass, objectives) table
    <label>_pareto.csv       — Pareto-non-dominated rows from grid
    plots/<label>_heatmap_*.png  — signal/background heatmaps over VDC × calo
    plots/<label>_pareto_front.png

Usage:
    python scripts/joint_pareto_scan.py
    python scripts/joint_pareto_scan.py \\
        --particles output/all_particles_target_exit.csv \\
        --flux      output/alplib_brems_flux.csv \\
        --vdc-min 30 --vdc-max 40 --vdc-step 2 \\
        --calo-min 12 --calo-max 20 --calo-step 2 \\
        --ma-list 10 20 50 100 200 \\
        --label Tz10 \\
        --output-dir output/joint_pareto
"""

import argparse
import sys
from itertools import product
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

# ─── Fixed geometry ──────────────────────────────────────────────────────────
TARGET_EXIT_Z_MM  = -400.0   # target rear face z [mm]; -50 cm + 10 cm target
MAGNET_LENGTH_MM  =  120.0   # fixed magnet length [mm]

CHARGE_C          = 1.602176634e-19
SECONDS_PER_DAY   = 86400.0


# ─── Background propagation (same physics as fast_pareto_scan.py) ────────────

def propagate_background_grid(particles_df: pd.DataFrame,
                              vdc_values_cm: np.ndarray,
                              calo_values_cm: np.ndarray,
                              norm_factor: float,
                              neutron_weight: float = 10.0) -> np.ndarray:
    """
    Straight-line propagate target-exit particles to calo face for each
    (VDC, calo) combination.

    Returns
    -------
    np.ndarray shape (len(vdc_values_cm), len(calo_values_cm))
    """
    photon_mask  = particles_df['pdg'].values == 22
    neutron_mask = particles_df['pdg'].values == 2112
    keep_mask    = photon_mask | neutron_mask

    df = particles_df[keep_mask].reset_index(drop=True)
    pdg = df['pdg'].values
    x   = df['x_mm'].values
    y   = df['y_mm'].values
    z   = df['z_mm'].values
    px  = df['px'].values
    py  = df['py'].values
    pz  = df['pz'].values
    w   = df['weight'].values

    species_w = np.where(pdg == 2112, neutron_weight, 1.0)

    fwd = pz > 0
    result = np.zeros((len(vdc_values_cm), len(calo_values_cm)))

    for iv, vdc_cm in enumerate(vdc_values_cm):
        z_calo_mm = TARGET_EXIT_Z_MM + vdc_cm * 10.0 + MAGNET_LENGTH_MM

        dt      = np.where(fwd, (z_calo_mm - z) / np.where(fwd, pz, 1.0), np.inf)
        x_calo  = x + dt * px
        y_calo  = y + dt * py

        for ic, calo_cm in enumerate(calo_values_cm):
            hw_mm = calo_cm / 2.0 * 10.0   # half-width in mm
            hits  = (np.abs(x_calo) <= hw_mm) & \
                    (np.abs(y_calo) <= hw_mm) & fwd
            result[iv, ic] = np.sum(w[hits] * species_w[hits]) * norm_factor

    return result


# ─── Signal acceptance (same physics as fast_pareto_scan.py) ─────────────────

def geometric_acceptance_grid(events: list,
                               vdc_values_cm: np.ndarray,
                               calo_values_cm: np.ndarray,
                               angle_cut_deg: float = 10.0,
                               energy_cut_MeV: float = 100.0,
                               rng: np.random.Generator = None
                               ) -> tuple:
    """
    For each (VDC, calo) pair, compute accepted_fraction and separable_fraction.

    Returns
    -------
    acc  : np.ndarray shape (len(vdc_values_cm), len(calo_values_cm))
    sep  : np.ndarray shape (len(vdc_values_cm), len(calo_values_cm))
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
        return (np.zeros((len(vdc_values_cm), len(calo_values_cm))),
                np.zeros((len(vdc_values_cm), len(calo_values_cm))))

    # Separability (independent of geometry)
    sep_mask = (theta_deg >= angle_cut_deg) | \
               ((theta_deg < angle_cut_deg) &
                (E1 >= energy_cut_MeV) & (E2 >= energy_cut_MeV))

    # Sample decay positions once (reuse across calo sizes for same VDC)
    magnet_cm = MAGNET_LENGTH_MM / 10.0

    acc = np.zeros((len(vdc_values_cm), len(calo_values_cm)))
    sep = np.zeros((len(vdc_values_cm), len(calo_values_cm)))

    for iv, vdc_cm in enumerate(vdc_values_cm):
        u       = rng.random(len(events))
        z_decay = -L_decay * np.log(np.clip(1.0 - u, 1e-30, None))
        in_gap  = (z_decay > 0) & (z_decay < vdc_cm)
        dist_to_calo = (vdc_cm + magnet_cm) - z_decay   # cm; positive when in_gap
        theta_rad    = np.radians(theta_deg)
        half_sep_cm  = theta_rad * dist_to_calo / 2.0

        for ic, calo_cm in enumerate(calo_values_cm):
            hw_cm    = calo_cm / 2.0
            both_hit = in_gap & (half_sep_cm <= hw_cm)

            acc[iv, ic] = (weights * both_hit).sum() / total_w
            sep[iv, ic] = (weights * both_hit * sep_mask).sum() / total_w

    return acc, sep


# ─── Pareto utilities ─────────────────────────────────────────────────────────

def is_pareto_optimal(objectives: np.ndarray, minimize: list) -> np.ndarray:
    """
    Return boolean mask of non-dominated rows.

    Parameters
    ----------
    objectives : shape (N, k)  — k objective values
    minimize   : list of bool, length k — True if that objective should be minimised
    """
    N = len(objectives)
    is_pareto = np.ones(N, dtype=bool)
    # Flip maximised objectives so we only need to handle minimisation
    obj = objectives.copy()
    for j, mn in enumerate(minimize):
        if not mn:
            obj[:, j] = -obj[:, j]

    for i in range(N):
        if not is_pareto[i]:
            continue
        dominated = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False

    return is_pareto


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_heatmaps(grid_df: pd.DataFrame, vdc_values: np.ndarray,
                  calo_values: np.ndarray, ma: float,
                  plot_dir: Path, label: str):
    if not PLOT:
        return

    sub = grid_df[grid_df['ma_MeV'] == ma]
    if sub.empty:
        return

    for col, title, cmap in [
        ('sep_efficiency',   'Separability efficiency',    'viridis'),
        ('separable_fraction', 'Signal separable fraction', 'viridis'),
        ('bkg_exposure',       'Weighted background (log)', 'magma_r'),
        ('fom',                'FoM = sep_frac / √bkg',    'plasma'),
    ]:
        pivot = sub.pivot(index='vdc_cm', columns='calo_cm', values=col)

        fig, ax = plt.subplots(figsize=(7, 5))
        data = pivot.values
        if col == 'bkg_exposure':
            data = np.log10(np.where(data > 0, data, np.nan))
            title = 'Weighted background (log₁₀)'

        im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap,
                       extent=[calo_values[0] - 1, calo_values[-1] + 1,
                                vdc_values[0] - 1,  vdc_values[-1]  + 1])
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Calo XY size [cm]', fontsize=11)
        ax.set_ylabel('VDC length [cm]', fontsize=11)
        ax.set_xticks(calo_values)
        ax.set_yticks(vdc_values)
        ax.set_title(f'{title}\nma={ma:.0f} MeV  ({label})', fontsize=10)

        fname = plot_dir / f'{label}_heatmap_{col}_ma{ma:.0f}MeV.png'
        plt.tight_layout()
        plt.savefig(fname, dpi=130)
        plt.close()
        print(f"  Saved: {fname}")


def plot_pareto(pareto_df: pd.DataFrame, plot_dir: Path, label: str,
                beam_uA: float, exposure_days: float):
    if not PLOT or pareto_df.empty:
        return

    masses = sorted(pareto_df['ma_MeV'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(masses)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for ma, col in zip(masses, colors):
        sub = pareto_df[pareto_df['ma_MeV'] == ma]
        ax.scatter(sub['bkg_exposure'], sub['separable_fraction'],
                   c=[col] * len(sub), label=f'ma={ma:.0f} MeV',
                   edgecolors='black', linewidths=0.8, s=70, zorder=5)
        for _, row in sub.iterrows():
            ax.annotate(
                f"V{row['vdc_cm']:.0f}/C{row['calo_cm']:.0f}",
                (row['bkg_exposure'], row['separable_fraction']),
                textcoords='offset points', xytext=(4, 2), fontsize=6, color=col,
            )

    ax.set_xlabel('Weighted background (full exposure)', fontsize=11)
    ax.set_ylabel('Signal separable fraction', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(
        f'Joint VDC × Calo Pareto front — {label}\n'
        f'I={beam_uA} µA, T={exposure_days:.0f} d  |  '
        f'Labels: V=VDC cm / C=Calo cm',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = plot_dir / f'{label}_pareto_front.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--particles', default='output/all_particles_target_exit.csv',
                        help='Target-exit particle CSV from Geant4')
    parser.add_argument('--flux',      default='output/alplib_brems_flux.csv',
                        help='Bremsstrahlung photon flux CSV for alplib')
    parser.add_argument('--n-primaries',  type=int,   default=100000)
    parser.add_argument('--beam-uA',      type=float, default=62.5)
    parser.add_argument('--exposure-days',type=float, default=30.0)
    parser.add_argument('--vdc-min',   type=float, default=30.0, help='VDC min [cm]')
    parser.add_argument('--vdc-max',   type=float, default=40.0, help='VDC max [cm]')
    parser.add_argument('--vdc-step',  type=float, default=2.0)
    parser.add_argument('--calo-min',  type=float, default=12.0, help='Calo XY min [cm]')
    parser.add_argument('--calo-max',  type=float, default=20.0, help='Calo XY max [cm]')
    parser.add_argument('--calo-step', type=float, default=2.0)
    parser.add_argument('--ma-list',   type=float, nargs='+',
                        default=[5, 10, 20, 50, 100, 200],
                        help='ALP masses to scan [MeV]')
    parser.add_argument('--coupling',  type=float, default=-1,
                        help='Coupling in GeV^-1; -1 = auto per mass')
    parser.add_argument('--n-samples', type=int,   default=10000)
    parser.add_argument('--angle-cut', type=float, default=10.0,
                        help='Separability opening angle cut [deg]')
    parser.add_argument('--energy-cut',type=float, default=100.0,
                        help='Separability per-photon energy cut [MeV]')
    parser.add_argument('--label',     default='Tz10',
                        help='Label prepended to output files (e.g. Tz14)')
    parser.add_argument('--output-dir',default='output/joint_pareto')
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir  = Path(args.output_dir)
    plot_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    vdc_values  = np.arange(args.vdc_min,  args.vdc_max  + 0.1, args.vdc_step)
    calo_values = np.arange(args.calo_min, args.calo_max + 0.1, args.calo_step)

    print(f"VDC range:  {vdc_values[0]:.0f} – {vdc_values[-1]:.0f} cm "
          f"({len(vdc_values)} points)")
    print(f"Calo range: {calo_values[0]:.0f} – {calo_values[-1]:.0f} cm "
          f"({len(calo_values)} points)")
    print(f"Mass list:  {args.ma_list}")
    print(f"Grid size:  {len(vdc_values) * len(calo_values) * len(args.ma_list)} "
          f"({len(vdc_values)}×{len(calo_values)} geo × {len(args.ma_list)} masses)")

    # ── Background grid (independent of mass) ────────────────────────────────
    print(f"\nLoading particles: {args.particles}")
    particles_df = pd.read_csv(args.particles)
    print(f"  {len(particles_df):,} particles loaded")

    eps        = (args.beam_uA * 1e-6) / CHARGE_C
    exposure_s = args.exposure_days * SECONDS_PER_DAY
    norm       = (eps * exposure_s) / args.n_primaries
    print(f"  Norm factor: {norm:.3e}")

    print("\nComputing background grid (VDC × calo)...")
    bkg_grid = propagate_background_grid(
        particles_df, vdc_values, calo_values, norm
    )
    print(f"  Background range: {bkg_grid.min():.3e} – {bkg_grid.max():.3e}")

    # ── Load flux (brems flux CSV, headerless, already rate-scaled) ──────────
    from scripts.pipeline.alp_signal_pipeline import load_geant4_brems_flux
    from scripts.optimization.fast_pareto_scan import auto_coupling
    import alplib.fluxes as af
    import alplib.materials as am
    import alplib.generators as ag
    from alplib.decay import W_gg
    from alplib.constants import HBAR, C_LIGHT

    print(f"\nLoading flux: {args.flux}")
    flux_array = load_geant4_brems_flux(args.flux, args.n_primaries)

    nominal_det_dist_m = (float(np.mean(vdc_values)) + MAGNET_LENGTH_MM / 10.0) / 100.0

    def generate_events(ma_MeV: float, coupling_GeV: float) -> tuple:
        """Run alplib for one mass, return (events list, n_total)."""
        g_GeV = coupling_GeV if coupling_GeV > 0 \
                else auto_coupling(flux_array, ma_MeV, target_decay_length_m=0.6)
        g_MeV = g_GeV / 1000.0

        max_E = flux_array[:, 0].max()
        if ma_MeV >= max_E:
            print(f"  [alplib] ma={ma_MeV} MeV >= max photon E={max_E:.1f} MeV")
            return [], 0.0

        det_area = (args.calo_max / 100.0) ** 2   # use max calo for initial sampling

        flux_obj = af.FluxPrimakoffIsotropic(
            photon_flux    = flux_array,
            target         = am.Material("W"),
            det_dist       = nominal_det_dist_m,
            det_length     = nominal_det_dist_m,
            det_area       = det_area,
            axion_mass     = ma_MeV,
            axion_coupling = g_MeV,
            n_samples      = args.n_samples,
        )
        flux_obj.simulate()

        if len(flux_obj.axion_energy) == 0:
            return [], 0.0

        decay_width = W_gg(g_MeV, ma_MeV)
        tau_rest    = HBAR / decay_width
        flux_obj.propagate(decay_width)
        generator = ag.PhotonEventGenerator(flux_obj, am.Material("CsI"))
        n_total   = generator.decays(days_exposure=args.exposure_days, threshold=0.1)

        p4_1, p4_2, weights = generator.simulate_decay_4vectors(
            days_exposure=args.exposure_days, n_samples=args.n_samples
        )

        events = []
        for idx in range(len(p4_1)):
            w = weights[idx] if idx < len(weights) else 0.0
            if w <= 0:
                continue
            p1 = p4_1[idx]; p2 = p4_2[idx]
            E1, px1, py1, pz1 = p1.p0, p1.p1, p1.p2, p1.p3
            E2, px2, py2, pz2 = p2.p0, p2.p1, p2.p2, p2.p3
            if E1 <= 0 or E2 <= 0:
                continue
            E_alp = E1 + E2
            if E_alp <= ma_MeV:
                continue
            mag1 = np.sqrt(px1**2 + py1**2 + pz1**2)
            mag2 = np.sqrt(px2**2 + py2**2 + pz2**2)
            if mag1 < 1e-12 or mag2 < 1e-12:
                continue
            cos_t = float(np.clip((px1*px2 + py1*py2 + pz1*pz2) / (mag1*mag2), -1, 1))
            theta_deg = np.degrees(np.arccos(cos_t))
            gamma = E_alp / ma_MeV
            beta  = np.sqrt(max(1.0 - (ma_MeV/E_alp)**2, 0.0))
            L_cm  = gamma * beta * C_LIGHT * tau_rest
            events.append({
                'weight': w, 'alp_E_MeV': E_alp,
                'E1_MeV': E1, 'E2_MeV': E2,
                'opening_angle_deg': theta_deg,
                'lab_decay_length_cm': L_cm,
            })
        print(f"  [alplib] {len(events)} samples, n_total={n_total:.3g}")
        return events, n_total

    all_rows = []

    for ma in args.ma_list:
        print(f"\n{'='*55}")
        print(f"ALP mass: {ma} MeV")
        print(f"{'='*55}")

        events, n_total = generate_events(ma, args.coupling)

        if not events:
            print(f"  Skipping ma={ma}: no events generated")
            continue

        acc_grid, sep_grid = geometric_acceptance_grid(
            events, vdc_values, calo_values,
            angle_cut_deg  = args.angle_cut,
            energy_cut_MeV = args.energy_cut,
            rng            = rng,
        )

        for iv, vdc_cm in enumerate(vdc_values):
            for ic, calo_cm in enumerate(calo_values):
                bkg = bkg_grid[iv, ic]
                sep = sep_grid[iv, ic]
                fom = sep / np.sqrt(bkg + 1e-30)
                acc_val = acc_grid[iv, ic]
                sep_eff = sep / acc_val if acc_val > 0 else 0.0
                all_rows.append({
                    'ma_MeV':             ma,
                    'vdc_cm':             vdc_cm,
                    'calo_cm':            calo_cm,
                    'bkg_exposure':       bkg,
                    'accepted_fraction':  acc_val,
                    'separable_fraction': sep,
                    'sep_efficiency':     sep_eff,
                    'n_alp_events':       n_total,
                    'fom':                fom,
                })

        print(f"  acc_grid:  {acc_grid.min():.4f} – {acc_grid.max():.4f}")
        print(f"  sep_grid:  {sep_grid.min():.4f} – {sep_grid.max():.4f}")

    if not all_rows:
        print("No results generated. Check flux CSV and mass list.")
        return

    grid_df = pd.DataFrame(all_rows)

    # ── Pareto front (minimize bkg, maximize separable_fraction) ─────────────
    pareto_rows = []
    for ma in grid_df['ma_MeV'].unique():
        sub = grid_df[grid_df['ma_MeV'] == ma].reset_index(drop=True)
        obj = sub[['bkg_exposure', 'separable_fraction']].values
        mask = is_pareto_optimal(obj, minimize=[True, False])
        pareto_rows.append(sub[mask])

    pareto_df = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()

    # ── Save outputs ──────────────────────────────────────────────────────────
    grid_path   = out_dir / f'{args.label}_grid.csv'
    pareto_path = out_dir / f'{args.label}_pareto.csv'

    grid_df.to_csv(grid_path, index=False)
    pareto_df.to_csv(pareto_path, index=False)

    print(f"\nSaved grid:   {grid_path}  ({len(grid_df)} rows)")
    print(f"Saved pareto: {pareto_path}  ({len(pareto_df)} Pareto-optimal rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    for ma in args.ma_list:
        plot_heatmaps(grid_df, vdc_values, calo_values, ma, plot_dir, args.label)

    plot_pareto(pareto_df, plot_dir, args.label, args.beam_uA, args.exposure_days)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Best configurations per mass (highest FoM on Pareto front) ===")
    for ma in sorted(pareto_df['ma_MeV'].unique()):
        sub = pareto_df[pareto_df['ma_MeV'] == ma]
        best = sub.loc[sub['fom'].idxmax()]
        print(f"  ma={ma:5.0f} MeV | VDC={best['vdc_cm']:.0f} cm "
              f"calo={best['calo_cm']:.0f} cm | "
              f"sep={best['separable_fraction']:.4f} "
              f"bkg={best['bkg_exposure']:.3e} "
              f"FoM={best['fom']:.4e}")


if __name__ == '__main__':
    main()
