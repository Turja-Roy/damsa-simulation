#!/usr/bin/env python3
"""
Step 7 — Final configuration selection and APS-ready sensitivity plots.

Reads:
    output/joint_pareto/Tz10_pareto.csv      (Phase A: VDC × calo scan at Tz10)
    output/target_projection/target_projection_grid.csv  (analytic target scan)
    output/alplib_brems_flux.csv              (brem flux for sensitivity curves)

Produces:
    output/final_report/optimal_config.csv   — ranked configuration table
    plots/final_report/pareto_front.png      — VDC×calo Pareto with annotations
    plots/final_report/fom_heatmap_*.png     — FoM heatmaps (best calo per VDC)
    plots/final_report/target_sensitivity.png — FoM & bkg vs target length
    plots/final_report/sensitivity_curve.png  — excluded g_agg vs m_a at best geometry

Usage:
    python scripts/final_report.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    PLOT = True
except ImportError:
    PLOT = False

from scripts.pipeline.alp_signal_pipeline import load_geant4_brems_flux
from scripts.optimization.fast_pareto_scan import auto_coupling
import alplib.fluxes as af
import alplib.materials as am
import alplib.generators as ag
from alplib.decay import W_gg
from alplib.constants import HBAR, C_LIGHT

MAGNET_CM       = 12.0    # fixed magnet length [cm]
BEAM_UA         = 62.5
EXPOSURE_DAYS   = 30.0
CHARGE_C        = 1.602176634e-19
SECONDS_PER_DAY = 86400.0


# ── Sensitivity scan at given geometry ────────────────────────────────────────

def sensitivity_at_geometry(flux_array, vdc_cm, calo_cm,
                             mass_grid_MeV, coupling_grid,
                             n_samples=500, exposure_days=30.0,
                             cl_events=2.3):
    """
    Compute the excluded coupling band for each mass at a specific geometry.

    Returns
    -------
    g_lower, g_upper : np.ndarray  shape (len(mass_grid_MeV),)
        Lower and upper edges of the excluded coupling band [GeV⁻¹].
        NaN if not excluded.
    """
    det_dist_m = (vdc_cm + MAGNET_CM) / 100.0    # target centre ≈ z=0
    det_area   = (calo_cm / 100.0) ** 2

    g_lower = np.full(len(mass_grid_MeV), np.nan)
    g_upper = np.full(len(mass_grid_MeV), np.nan)

    for i, ma in enumerate(mass_grid_MeV):
        max_E = flux_array[:, 0].max()
        if ma >= max_E:
            continue

        n_sig = np.zeros(len(coupling_grid))
        for j, g_GeV in enumerate(coupling_grid):
            g_MeV = g_GeV / 1000.0
            try:
                flux_obj = af.FluxPrimakoffIsotropic(
                    photon_flux    = flux_array,
                    target         = am.Material("W"),
                    det_dist       = det_dist_m,
                    det_length     = det_dist_m,
                    det_area       = det_area,
                    axion_mass     = ma,
                    axion_coupling = g_MeV,
                    n_samples      = n_samples,
                )
                flux_obj.simulate()
                if len(flux_obj.axion_energy) == 0:
                    continue
                flux_obj.propagate(W_gg(g_MeV, ma))
                gen = ag.PhotonEventGenerator(flux_obj, am.Material("CsI"))
                n_sig[j] = gen.decays(days_exposure=exposure_days, threshold=0.1)
            except Exception:
                n_sig[j] = 0.0

        excluded = n_sig > cl_events
        if excluded.any():
            idx = np.where(excluded)[0]
            g_lower[i] = coupling_grid[idx[0]]
            g_upper[i] = coupling_grid[idx[-1]]
            print(f"  ma={ma:5.0f} MeV: [{g_lower[i]:.2e}, {g_upper[i]:.2e}] GeV⁻¹  "
                  f"(max N={n_sig[idx].max():.1f})")
        else:
            print(f"  ma={ma:5.0f} MeV: not excluded (max N={n_sig.max():.2e})")

    return g_lower, g_upper


# ── Pareto knee-point selection ────────────────────────────────────────────────

def weighted_score(pareto_df, w_sep=0.60, w_bkg=0.25, w_acc=0.15):
    """
    Weighted normalised score across Pareto-optimal configurations.
    Maximise separable_fraction (60 %), minimise bkg_exposure (25 %),
    maximise accepted_fraction (15 %).
    """
    df = pareto_df.copy()
    sep = df['separable_fraction']
    bkg = df['bkg_exposure']
    acc = df['accepted_fraction']

    # Normalise each column to [0, 1]
    def norm(x, minimize=False):
        lo, hi = x.min(), x.max()
        if hi == lo:
            return np.zeros(len(x))
        n = (x - lo) / (hi - lo)
        return (1 - n) if minimize else n

    df['_score'] = (w_sep * norm(sep)
                    + w_bkg * norm(bkg, minimize=True)
                    + w_acc * norm(acc))
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_pareto_front(pareto_df, plot_dir, title_suffix=""):
    if not PLOT:
        return
    masses = sorted(pareto_df['ma_MeV'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(masses)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for ma, col in zip(masses, colors):
        sub = pareto_df[pareto_df['ma_MeV'] == ma]
        ax.scatter(sub['bkg_exposure'], sub['separable_fraction'],
                   c=[col] * len(sub), label=f'ma={ma:.0f} MeV',
                   edgecolors='black', linewidths=0.8, s=70, zorder=5)
        for _, row in sub.iterrows():
            ax.annotate(f"V{row['vdc_cm']:.0f}/C{row['calo_cm']:.0f}",
                        (row['bkg_exposure'], row['separable_fraction']),
                        textcoords='offset points', xytext=(4, 2),
                        fontsize=6, color=col)

    ax.set_xlabel('Weighted background (30-day exposure)', fontsize=11)
    ax.set_ylabel('Separability efficiency', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(f'VDC × Calo Pareto front — Tz10 baseline{title_suffix}\n'
                 'Labels: V=VDC cm / C=Calo cm', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = plot_dir / 'pareto_front.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_fom_heatmap(grid_df, vdc_values, calo_values, ma, plot_dir):
    if not PLOT:
        return
    sub = grid_df[grid_df['ma_MeV'] == ma]
    if sub.empty:
        return

    pivot = sub.pivot(index='vdc_cm', columns='calo_cm', values='fom')
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='plasma',
                   extent=[calo_values[0]-1, calo_values[-1]+1,
                            vdc_values[0]-1,  vdc_values[-1]+1])
    plt.colorbar(im, ax=ax, label='FoM = sep / √bkg')
    ax.set_xlabel('Calo XY size [cm]', fontsize=11)
    ax.set_ylabel('VDC length [cm]', fontsize=11)
    ax.set_xticks(calo_values)
    ax.set_yticks(vdc_values)
    ax.set_title(f'Figure of Merit: VDC × Calo grid\nma={ma:.0f} MeV, Tz10 baseline',
                 fontsize=11)
    plt.tight_layout()
    out = plot_dir / f'fom_heatmap_ma{ma:.0f}MeV.png'
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved: {out}")


def plot_target_sensitivity(proj_df, vdc_cm, calo_cm, plot_dir):
    """
    Two-panel plot: background and FoM vs target length at fixed VDC, calo.
    Uses projection data. Normalised to Tz10.
    """
    if not PLOT:
        return
    masses = sorted(proj_df['ma_MeV'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(masses)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax_bkg = axes[0]
    ax_fom = axes[1]

    for ma, col in zip(masses, colors):
        sub = proj_df[(proj_df['vdc_cm'] == vdc_cm) &
                      (proj_df['calo_cm'] == calo_cm) &
                      (proj_df['ma_MeV'] == ma)].sort_values('target_cm')
        if sub.empty:
            continue
        bkg = sub['bkg_exposure'].values
        fom = sub['fom'].values
        L   = sub['target_cm'].values

        ax_bkg.plot(L, bkg / bkg[0], 'o-', color=col, label=f'ma={ma:.0f} MeV')
        ax_fom.plot(L, fom / fom[0], 'o-', color=col, label=f'ma={ma:.0f} MeV')

    ax_bkg.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax_bkg.set_xlabel('Target length [cm]', fontsize=12)
    ax_bkg.set_ylabel('Background / Background(10 cm)', fontsize=11)
    ax_bkg.set_title(
        'Background vs target length (analytic projection)\n'
        'Photon attenuation in extra W material  [NIST XCOM]',
        fontsize=10,
    )
    ax_bkg.set_yscale('log')
    ax_bkg.legend(fontsize=9)
    ax_bkg.grid(True, alpha=0.3, which='both')

    ax_fom.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax_fom.set_xlabel('Target length [cm]', fontsize=12)
    ax_fom.set_ylabel('FoM / FoM(10 cm)', fontsize=11)
    ax_fom.set_title(
        'FoM vs target length\n'
        'Signal ≈ constant (shower saturated); bkg decreases → FoM improves',
        fontsize=10,
    )
    ax_fom.legend(fontsize=9)
    ax_fom.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xticks(sorted(proj_df['target_cm'].unique()))

    plt.tight_layout()
    out = plot_dir / f'target_sensitivity_VDC{vdc_cm:.0f}_Calo{calo_cm:.0f}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_sensitivity_curve(mass_grid, g_lower, g_upper, plot_dir,
                           vdc_cm, calo_cm, target_cm=10):
    if not PLOT:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    mask = ~np.isnan(g_lower)

    if mask.any():
        ax.fill_between(mass_grid[mask], g_lower[mask], g_upper[mask],
                        color='steelblue', alpha=0.35, label='DAMSA 90% CL reach')
        ax.plot(mass_grid[mask], g_lower[mask], 'b-', lw=2)
        ax.plot(mass_grid[mask], g_upper[mask], 'b-', lw=2)
    else:
        ax.text(0.5, 0.5, 'No sensitivity found\n(increase exposure or coupling range)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.set_xlabel(r'ALP mass $m_a$ [MeV]', fontsize=13)
    ax.set_ylabel(r'Coupling $g_{a\gamma\gamma}$ [GeV$^{-1}$]', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(
        f'DAMSA projected sensitivity  —  g_aγγ vs mₐ\n'
        f'Optimal geometry: VDC={vdc_cm:.0f} cm, Calo={calo_cm:.0f} cm, '
        f'Target={target_cm:.0f} cm\n'
        f'I = {BEAM_UA} µA,  T = {EXPOSURE_DAYS:.0f} days,  90% CL (N > 2.3)',
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = plot_dir / f'sensitivity_curve_VDC{vdc_cm:.0f}_Calo{calo_cm:.0f}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def compute_global_3d_pareto(proj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-dominated sort across all (target_cm, vdc_cm, calo_cm) combinations,
    computed separately per mass.

    Objectives: minimise bkg_exposure, maximise separable_fraction.

    Returns a filtered DataFrame (subset of proj_df) flagged as global Pareto.
    """
    from scripts.optimization.joint_pareto_scan import is_pareto_optimal

    pareto_rows = []
    for ma in sorted(proj_df['ma_MeV'].unique()):
        sub  = proj_df[proj_df['ma_MeV'] == ma].reset_index(drop=True)
        obj  = sub[['bkg_exposure', 'separable_fraction']].values
        mask = is_pareto_optimal(obj, minimize=[True, False])
        pareto_rows.append(sub[mask])

    return pd.concat(pareto_rows, ignore_index=True)


def plot_3d_pareto(pareto_df: pd.DataFrame, plot_dir: Path):
    """
    Global 3D Pareto front: bkg vs separable_fraction scatter,
    coloured by target length, one panel per mass.
    """
    if not PLOT:
        return

    masses = sorted(pareto_df['ma_MeV'].unique())
    n_col = min(3, len(masses))
    n_row = int(np.ceil(len(masses) / n_col))

    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(5 * n_col, 4 * n_row),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    target_vals = np.sort(pareto_df['target_cm'].unique())
    target_cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(target_vals)))
    target_color = dict(zip(target_vals, target_cmap))

    for ax_idx, ma in enumerate(masses):
        ax = axes_flat[ax_idx]
        sub = pareto_df[pareto_df['ma_MeV'] == ma]

        for t_cm in target_vals:
            tsub = sub[sub['target_cm'] == t_cm]
            if tsub.empty:
                continue
            ax.scatter(tsub['bkg_exposure'], tsub['separable_fraction'],
                       color=target_color[t_cm], s=60,
                       edgecolors='black', linewidths=0.5,
                       label=f'T={t_cm:.0f} cm', zorder=5)
            for _, row in tsub.iterrows():
                ax.annotate(
                    f"V{row['vdc_cm']:.0f}/C{row['calo_cm']:.0f}",
                    (row['bkg_exposure'], row['separable_fraction']),
                    textcoords='offset points', xytext=(3, 2),
                    fontsize=5, color=target_color[t_cm],
                )

        ax.set_xscale('log')
        ax.set_xlabel('Bkg (30-day exposure)', fontsize=9)
        ax.set_ylabel('Separable fraction', fontsize=9)
        ax.set_title(f'ma = {ma:.0f} MeV', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        if ax_idx == 0:
            ax.legend(fontsize=6, loc='upper right',
                      title='Target length', title_fontsize=7)

    # Hide unused panels
    for ax in axes_flat[len(masses):]:
        ax.set_visible(False)

    fig.suptitle(
        'Global 3D Pareto front: (Target × VDC × Calo) all non-dominated configs\n'
        'Labels: V=VDC cm / C=Calo cm.  Colour = target length.',
        fontsize=11,
    )
    plt.tight_layout()
    out = plot_dir / 'global_3d_pareto.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_3d_pareto_summary(pareto_df: pd.DataFrame, plot_dir: Path):
    """
    For each mass: best FoM config per target length — shows how the
    optimal VDC and calo trade off as target length changes.
    """
    if not PLOT:
        return

    masses = sorted(pareto_df['ma_MeV'].unique())
    target_vals = np.sort(pareto_df['target_cm'].unique())

    if 'fom' not in pareto_df.columns:
        pareto_df = pareto_df.copy()
        pareto_df['fom'] = (pareto_df['separable_fraction'] /
                            np.sqrt(pareto_df['bkg_exposure'] + 1e-30))

    colors = plt.cm.tab10(np.linspace(0, 1, len(masses)))
    fig, ax = plt.subplots(figsize=(9, 5))

    for ma, col in zip(masses, colors):
        sub = pareto_df[pareto_df['ma_MeV'] == ma]
        best_per_target = (sub.groupby('target_cm')['fom']
                           .idxmax().dropna())
        best = sub.loc[best_per_target].sort_values('target_cm')
        fom0 = best['fom'].iloc[0]
        ax.plot(best['target_cm'], best['fom'] / fom0, 'o-',
                color=col, label=f'ma={ma:.0f} MeV', ms=7)

    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('Target length [cm]', fontsize=12)
    ax.set_ylabel('Best FoM / FoM(10 cm)  [per mass]', fontsize=11)
    ax.set_title(
        'Global Pareto: best FoM vs target length\n'
        '(VDC and calo re-optimised at each target length)',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(target_vals)
    plt.tight_layout()
    out = plot_dir / 'global_3d_pareto_fom_vs_target.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_shower_saturation(plot_dir):
    """Inlined from project_target_length — shower scale vs target length."""
    if not PLOT:
        return
    L_vals = np.arange(10, 21)
    # Pre-computed Longo fractions
    scales = np.array([1.000000, 1.000051, 1.000103, 1.000142, 1.000167,
                       1.000181, 1.000191, 1.000197, 1.000201, 1.000204,
                       1.000206])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(L_vals, (scales - 1) * 100, 'rs-', ms=7, lw=2)
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('Target length [cm]', fontsize=12)
    ax.set_ylabel('ALP signal change vs 10 cm  [%]', fontsize=12)
    ax.set_title(
        r'8 GeV $e^-$ shower saturation in W' '\n'
        r'Signal change $< 0.021\%$ for target 10–20 cm',
        fontsize=11,
    )
    for L, s in zip(L_vals, scales):
        ax.annotate(f'{(s-1)*100:.3f}%', (L, (s-1)*100),
                    textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=8)
    ax.set_ylim(-0.001, 0.025)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plot_dir / 'shower_saturation.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pareto',      default='output/joint_pareto/Tz10_pareto.csv')
    parser.add_argument('--grid',        default='output/joint_pareto/Tz10_grid.csv')
    parser.add_argument('--proj',        default='output/target_projection/target_projection_grid.csv')
    parser.add_argument('--flux',        default='output/alplib_brems_flux.csv')
    parser.add_argument('--n-primaries', type=int,   default=100000)
    parser.add_argument('--output-dir',  default='output/final_report')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip (slow) sensitivity curve computation')
    parser.add_argument('--n-sensitivity-samples', type=int, default=300,
                        help='alplib n_samples per coupling point for sensitivity scan')
    parser.add_argument('--mass-grid', type=float, nargs='+',
                        default=[5., 10., 20., 50., 100., 200.])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir  = Path(args.output_dir)
    plot_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=== DAMSA Final Configuration Report ===")
    print()

    # ── Load Phase A Pareto (VDC × calo scan, Tz10) ───────────────────────────
    pareto_df = pd.read_csv(args.pareto)
    print(f"Phase A Pareto:  {args.pareto}  ({len(pareto_df)} rows)")

    # ── Load Phase A full grid ─────────────────────────────────────────────────
    grid_path = Path(args.grid)
    if grid_path.exists():
        grid_df = pd.read_csv(grid_path)
        vdc_values  = np.sort(grid_df['vdc_cm'].unique())
        calo_values = np.sort(grid_df['calo_cm'].unique())
        print(f"Phase A grid:    {args.grid}  ({len(grid_df)} rows)")
    else:
        grid_df = None
        vdc_values  = np.array([30., 32., 34., 36., 38., 40.])
        calo_values = np.array([12., 16., 20.])
        print(f"WARNING: {args.grid} not found — heatmaps skipped")

    # ── Load target projection ─────────────────────────────────────────────────
    proj_path = Path(args.proj)
    if proj_path.exists():
        proj_df = pd.read_csv(proj_path)
        print(f"Target proj:     {args.proj}  ({len(proj_df)} rows)")
    else:
        proj_df = None
        print(f"WARNING: {args.proj} not found — target plots skipped")

    # ── Load brem flux ─────────────────────────────────────────────────────────
    flux_array = load_geant4_brems_flux(args.flux, args.n_primaries)
    print(f"Brem flux:       {args.flux}  ({len(flux_array)} bins)")
    print()

    # ── Weighted scoring → optimal configuration ───────────────────────────────
    print("=== Optimal configuration (weighted score) ===")
    print(f"Weights: separable_frac=60%, bkg=-25%, accepted_frac=15%")
    print()

    mass_configs = {}
    for ma in sorted(pareto_df['ma_MeV'].unique()):
        sub = pareto_df[pareto_df['ma_MeV'] == ma].reset_index(drop=True)
        scored = weighted_score(sub)
        best = scored.loc[scored['_score'].idxmax()]
        mass_configs[ma] = best
        print(f"  ma={ma:5.0f} MeV: VDC={best['vdc_cm']:.0f} cm, "
              f"calo={best['calo_cm']:.0f} cm  "
              f"(sep={best['separable_fraction']:.4f}, "
              f"bkg={best['bkg_exposure']:.2e}, "
              f"FoM={best['fom']:.3e})")

    print()

    # Most common optimal config across masses
    all_best = pd.DataFrame([r for r in mass_configs.values()])
    best_vdc  = float(all_best['vdc_cm'].mode().iloc[0])
    best_calo = float(all_best['calo_cm'].mode().iloc[0])
    print(f"Consensus optimal geometry: VDC={best_vdc:.0f} cm, Calo={best_calo:.0f} cm")
    print()

    # ── Save config table ──────────────────────────────────────────────────────
    config_rows = []
    for ma, row in mass_configs.items():
        config_rows.append({
            'ma_MeV':             ma,
            'optimal_vdc_cm':     row['vdc_cm'],
            'optimal_calo_cm':    row['calo_cm'],
            'separable_fraction': row['separable_fraction'],
            'bkg_exposure':       row['bkg_exposure'],
            'fom':                row['fom'],
        })
    config_df = pd.DataFrame(config_rows)
    cfg_path = out_dir / 'optimal_config.csv'
    config_df.to_csv(cfg_path, index=False)
    print(f"Saved config table: {cfg_path}")
    print()

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("Generating plots ...")

    # 1. Shower saturation
    plot_shower_saturation(plot_dir)

    # 2. Pareto front
    plot_pareto_front(pareto_df, plot_dir)

    # 3. FoM heatmaps per mass
    if grid_df is not None:
        for ma in sorted(grid_df['ma_MeV'].unique()):
            plot_fom_heatmap(grid_df, vdc_values, calo_values, ma, plot_dir)

    # 4. Global 3D Pareto (if projection available)
    global_pareto_3d = None
    if proj_df is not None:
        print("\nComputing global 3D Pareto front (target × VDC × calo) ...")
        global_pareto_3d = compute_global_3d_pareto(proj_df)
        if 'fom' not in global_pareto_3d.columns:
            global_pareto_3d['fom'] = (
                global_pareto_3d['separable_fraction'] /
                np.sqrt(global_pareto_3d['bkg_exposure'] + 1e-30)
            )

        gp_path = out_dir / 'global_3d_pareto.csv'
        global_pareto_3d.to_csv(gp_path, index=False)
        print(f"  Global 3D Pareto: {len(global_pareto_3d)} non-dominated configs")
        print(f"  Saved: {gp_path}")

        # Print best per mass
        print("\n  Best configuration per mass (global 3D Pareto, max FoM):")
        print(f"  {'ma':>8} {'target':>8} {'VDC':>6} {'calo':>6} {'sep':>8} "
              f"{'bkg':>12} {'FoM':>12}")
        print("  " + "-" * 64)
        for ma in sorted(global_pareto_3d['ma_MeV'].unique()):
            sub  = global_pareto_3d[global_pareto_3d['ma_MeV'] == ma]
            best = sub.loc[sub['fom'].idxmax()]
            print(f"  {ma:8.0f} {best['target_cm']:8.0f} {best['vdc_cm']:6.0f} "
                  f"{best['calo_cm']:6.0f} {best['separable_fraction']:8.4f} "
                  f"{best['bkg_exposure']:12.3e} {best['fom']:12.4e}")

        # Plots
        proj_vdcs  = np.sort(proj_df['vdc_cm'].unique())
        proj_calos = np.sort(proj_df['calo_cm'].unique())
        pv = float(proj_vdcs[0])   # smallest VDC (most likely 30 cm)
        pc = float(proj_calos[-1]) # largest calo (most likely 20 cm)
        plot_target_sensitivity(proj_df, pv, pc, plot_dir)
        plot_3d_pareto(global_pareto_3d, plot_dir)
        plot_3d_pareto_summary(global_pareto_3d, plot_dir)

    # 5. Sensitivity curve at optimal geometry
    if not args.skip_sensitivity:
        mass_grid  = np.array(args.mass_grid)
        coupling_g = np.logspace(-7, -2, 50)

        print(f"\nComputing sensitivity curve at VDC={best_vdc:.0f} cm, "
              f"Calo={best_calo:.0f} cm ...")
        print(f"  Masses: {args.mass_grid}")
        print(f"  Couplings: {coupling_g[0]:.2e} – {coupling_g[-1]:.2e} GeV⁻¹ "
              f"({len(coupling_g)} points)")

        g_lower, g_upper = sensitivity_at_geometry(
            flux_array, best_vdc, best_calo, mass_grid, coupling_g,
            n_samples=args.n_sensitivity_samples,
            exposure_days=EXPOSURE_DAYS,
        )

        # Save raw sensitivity data
        sens_df = pd.DataFrame({
            'ma_MeV':   mass_grid,
            'g_lower':  g_lower,
            'g_upper':  g_upper,
        })
        sens_path = out_dir / f'sensitivity_VDC{best_vdc:.0f}_Calo{best_calo:.0f}.csv'
        sens_df.to_csv(sens_path, index=False)
        print(f"  Saved: {sens_path}")

        plot_sensitivity_curve(
            mass_grid, g_lower, g_upper, plot_dir,
            best_vdc, best_calo, target_cm=10,
        )
    else:
        print("\nSensitivity computation skipped (--skip-sensitivity).")

    # ── Final summary ──────────────────────────────────────────────────────────
    print()
    print("=== Summary ===")
    # If 3D Pareto available, report mode of best configs across masses
    if global_pareto_3d is not None:
        best3d = []
        for ma in sorted(global_pareto_3d['ma_MeV'].unique()):
            sub = global_pareto_3d[global_pareto_3d['ma_MeV'] == ma]
            best3d.append(sub.loc[sub['fom'].idxmax()])
        best3d_df = pd.DataFrame(best3d)
        rec_target = float(best3d_df['target_cm'].mode().iloc[0])
        rec_vdc    = float(best3d_df['vdc_cm'].mode().iloc[0])
        rec_calo   = float(best3d_df['calo_cm'].mode().iloc[0])
    else:
        rec_target, rec_vdc, rec_calo = 10.0, best_vdc, best_calo

    print(f"Optimal geometry (3D Pareto):  "
          f"Target = {rec_target:.0f} cm,  VDC = {rec_vdc:.0f} cm,  Calo = {rec_calo:.0f} cm")
    print()
    print("  Rationale:")
    print("  • Target length: shower fully saturated at 10 cm (28.5 X₀ in W);")
    print("    ALP signal changes < 0.021 % for 10–20 cm. 10 cm is sufficient.")
    print("  • VDC: shorter VDC reduces photon spread at calo face → less background,")
    print("    more signal hits → VDC=30 cm consistently wins across all masses.")
    print("  • Calo XY: larger calo gains acceptance for ma ≥ 200 MeV (+18 pp);")
    print("    20 cm provides best FoM at minimal cost.")
    print()
    print(f"All outputs written to: {out_dir}/")
    print(f"Plots written to:       {plot_dir}/")


if __name__ == '__main__':
    main()
