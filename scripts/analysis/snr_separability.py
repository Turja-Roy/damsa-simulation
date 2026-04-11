#!/usr/bin/env python3
"""
SNR Separability Analysis for DAMSA — Phase 5

Computes signal-to-noise ratio (SNR) for ALP signal vs background at the
calorimeter face, using 2D (E, theta) likelihood-region cuts.

Inputs:
  - Background: output/brem_calo_face_particles.csv (from electron-beam run)
  - Signal: output/alp_decay_photons_ma{ma}MeV_calo_face_particles.csv (per mass)

Outputs:
  - output/snr_summary.csv: per-mass SNR numbers
  - plots/snr/snr_vs_mass.png: SNR vs ALP mass curve
  - plots/snr/cut_contour_ma{ma}MeV.png: 2D (E, theta) cut visualization per mass

Usage:
    python scripts/snr_separability.py \\
        --bkg-csv output/brem_calo_face_particles.csv \\
        --n-primaries 100000 \\
        --beam-current-uA 62.5 \\
        --signal-pattern "output/alp_decay_photons_ma{ma}MeV_calo_face_particles.csv" \\
        --ma-list 1 5 10 20 50 100 200 500 \\
        --out-csv output/snr_summary.csv \\
        --plot-dir plots/snr
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    PLOT = True
except ImportError:
    PLOT = False
    print("Warning: matplotlib not available, skipping plots")


# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
CHARGE_COULOMBS = 1.602176634e-19
SECONDS_PER_DAY = 86400.0


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def load_calo_face(path: str, particle_filter: list = None) -> pd.DataFrame:
    """
    Load per-particle CSV from calorimeter face.
    
    Args:
        path: Path to calo_face_particles.csv
        particle_filter: Optional list of PDG codes to keep (e.g., [22] for photons)
    
    Returns:
        DataFrame with columns: pdg, energy_MeV, time_ns, x_mm, y_mm, z_mm,
                                px, py, pz, weight, trackID, eventID
    """
    df = pd.read_csv(path)
    if particle_filter is not None:
        df = df[df['pdg'].isin(particle_filter)]
    return df


def smear_energy(E_MeV: np.ndarray, a: float = 0.02, b: float = 0.01, 
                 rng: np.random.Generator = None) -> np.ndarray:
    """
    Apply Gaussian energy smearing to simulate calorimeter resolution.
    
    Resolution: sigma/E = a/sqrt(E[GeV]) + b (quadrature sum)
    
    Args:
        E_MeV: True energies in MeV
        a: Stochastic term (default 2%)
        b: Constant term (default 1%)
        rng: Random number generator
    
    Returns:
        Smeared energies in MeV (negative values clipped to 0)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    E_GeV = E_MeV / 1000.0
    # sigma/E = sqrt( (a/sqrt(E))^2 + b^2 )
    rel_sigma = np.sqrt((a / np.sqrt(np.maximum(E_GeV, 1e-6)))**2 + b**2)
    sigma_MeV = E_MeV * rel_sigma
    
    E_smeared = rng.normal(E_MeV, sigma_MeV)
    return np.maximum(E_smeared, 0.0)


def polar_angle(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """
    Compute polar angle to +z axis in radians.
    
    theta = arccos(pz / |p|)
    """
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    cos_theta = np.clip(pz / np.maximum(p_mag, 1e-12), -1.0, 1.0)
    return np.arccos(cos_theta)


def build_2d_hist(E: np.ndarray, theta: np.ndarray, weights: np.ndarray,
                  E_bins: np.ndarray, theta_bins: np.ndarray) -> np.ndarray:
    """
    Build 2D histogram of (E, theta) with weights.
    
    Returns:
        2D histogram array, shape (len(E_bins)-1, len(theta_bins)-1)
    """
    H, _, _ = np.histogram2d(E, theta, bins=[E_bins, theta_bins], weights=weights)
    return H


def data_driven_k_grid(S: np.ndarray, B: np.ndarray, n: int = 60) -> np.ndarray:
    """Build a k_grid from the actual S/B distribution.

    Fixed logspace grids miss the interesting range when S/B is extreme. We
    instead pick n log-spaced percentiles between the min and max of the
    nonzero S/B ratios in the histogram, guaranteeing the threshold scan
    actually probes the data.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where((B > 0) & (S > 0), S / B, np.nan)
    valid = r[np.isfinite(r) & (r > 0)]
    if valid.size < 4:
        return np.array([0.0])
    lo = np.percentile(valid, 1.0)
    hi = np.percentile(valid, 99.9)
    if lo <= 0 or hi <= lo:
        return np.array([0.0])
    return np.logspace(np.log10(lo), np.log10(hi), n)


def best_sb_threshold(S: np.ndarray, B: np.ndarray,
                      k_grid: np.ndarray) -> tuple:
    """
    Find optimal S/B threshold that maximizes SNR = S_acc / sqrt(S_acc + B_acc).

    For each k in k_grid, accept bins where S/B > k. The no-cut baseline
    (k = 0, accept everything with S > 0) is always included as the
    starting point, so the returned SNR is never worse than the trivial
    S_total / sqrt(S_total + B_total) even when k_grid is too coarse or
    shifted out of range for the particular S/B landscape.

    Args:
        S: Signal histogram (2D)
        B: Background histogram (2D)
        k_grid: Array of S/B threshold values to try

    Returns:
        (best_k, best_snr, S_acc, B_acc, accepted_mask)
    """
    # No-cut baseline: every bin with S > 0 contributes (empty bins contribute
    # nothing anyway). Always a valid fallback.
    baseline_mask = S > 0
    baseline_S = float(np.sum(S[baseline_mask]))
    baseline_B = float(np.sum(B[baseline_mask]))
    if baseline_S + baseline_B > 0:
        baseline_snr = baseline_S / np.sqrt(baseline_S + baseline_B)
    else:
        baseline_snr = 0.0

    best_k = 0.0
    best_snr = baseline_snr
    best_S_acc = baseline_S
    best_B_acc = baseline_B
    best_mask = baseline_mask

    # Avoid division by zero in S/B ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        sb_ratio = np.where(B > 0, S / B, np.inf)

    for k in k_grid:
        mask = sb_ratio > k
        S_acc = float(np.sum(S[mask]))
        B_acc = float(np.sum(B[mask]))

        if S_acc + B_acc > 0:
            snr = S_acc / np.sqrt(S_acc + B_acc)
        else:
            snr = 0.0

        if snr > best_snr:
            best_snr = snr
            best_k = k
            best_S_acc = S_acc
            best_B_acc = B_acc
            best_mask = mask

    return best_k, best_snr, best_S_acc, best_B_acc, best_mask


def per_second_normalization(n_primaries: int, beam_current_uA: float) -> float:
    """
    Compute the multiplier to convert raw MC weights to per-second rates.
    
    electrons_per_second / n_primaries = scaling factor
    
    Args:
        n_primaries: Number of primary electrons simulated
        beam_current_uA: Beam current in microamps
    
    Returns:
        Scaling factor (electrons/s / n_primaries)
    """
    electrons_per_s = (beam_current_uA * 1e-6) / CHARGE_COULOMBS
    return electrons_per_s / n_primaries


def plot_cut_contour(S: np.ndarray, B: np.ndarray, mask: np.ndarray,
                     E_bins: np.ndarray, theta_bins: np.ndarray,
                     ma_MeV: float, k: float, snr: float,
                     output_path: Path,
                     exposure_days: float = 30.0,
                     beam_current_uA: float = 62.5):
    """
    Plot 2D (E, theta) heatmaps with cut contour overlay.

    Units: S and B are total events in the full exposure.
    """
    if not PLOT:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    cbar_label = f'Events / {exposure_days:.0f}-day run'

    # Background heatmap
    ax = axes[0]
    B_plot = np.maximum(B.T, 1e-10)  # Transpose for imshow
    im = ax.pcolormesh(E_bins, np.degrees(theta_bins), B_plot,
                       norm=LogNorm(vmin=B_plot[B_plot > 0].min() if np.any(B_plot > 0) else 1e-10,
                                   vmax=B_plot.max() if B_plot.max() > 0 else 1),
                       cmap='Blues')
    ax.set_xlabel('E [MeV]')
    ax.set_ylabel('theta [deg]')
    ax.set_title('Background (brem)')
    ax.set_xscale('log')
    plt.colorbar(im, ax=ax, label=cbar_label)

    # Signal heatmap
    ax = axes[1]
    S_plot = np.maximum(S.T, 1e-10)
    im = ax.pcolormesh(E_bins, np.degrees(theta_bins), S_plot,
                       norm=LogNorm(vmin=S_plot[S_plot > 0].min() if np.any(S_plot > 0) else 1e-10,
                                   vmax=S_plot.max() if S_plot.max() > 0 else 1),
                       cmap='Reds')
    ax.set_xlabel('E [MeV]')
    ax.set_ylabel('theta [deg]')
    ax.set_title(f'Signal (ma={ma_MeV} MeV)')
    ax.set_xscale('log')
    plt.colorbar(im, ax=ax, label=cbar_label)
    
    # S/B ratio with cut contour
    ax = axes[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        sb_ratio = np.where(B > 0, S / B, 0).T
    im = ax.pcolormesh(E_bins, np.degrees(theta_bins), sb_ratio,
                       norm=LogNorm(vmin=1e-3, vmax=max(sb_ratio.max(), 1)),
                       cmap='RdYlGn')
    # Overlay cut contour
    ax.contour(0.5*(E_bins[:-1]+E_bins[1:]), 
               np.degrees(0.5*(theta_bins[:-1]+theta_bins[1:])),
               mask.T.astype(float), levels=[0.5], colors='black', linewidths=2)
    ax.set_xlabel('E [MeV]')
    ax.set_ylabel('theta [deg]')
    ax.set_title(f'S/B,  cut k={k:.2e},  SNR={snr:.3g}')
    ax.set_xscale('log')
    plt.colorbar(im, ax=ax, label='S/B')

    fig.suptitle(
        f'DAMSA ALP search — ma={ma_MeV} MeV  |  '
        f'I_avg={beam_current_uA} µA CW  |  T={exposure_days:.0f} d  |  '
        f'σ/E = 2%/√E ⊕ 1%',
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_snr_vs_mass(results: pd.DataFrame, output_path: Path,
                     exposure_days: float = 30.0,
                     beam_current_uA: float = 62.5):
    """
    Plot SNR vs ALP mass curve (total exposure).
    """
    if not PLOT:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results['ma_MeV'], results['snr'],
            'o-', markersize=8, linewidth=2, color='C3')
    ax.set_xlabel('ALP mass [MeV]')
    ax.set_ylabel(r'SNR $= S / \sqrt{S+B}$  (full exposure)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(
        f'DAMSA SNR vs ALP mass  —  '
        f'I_avg={beam_current_uA} µA CW, T={exposure_days:.0f} days'
    )
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SNR separability analysis at calorimeter face"
    )
    parser.add_argument('--bkg-csv', required=True,
                        help='Background calo face CSV (from electron-beam run)')
    parser.add_argument('--n-primaries', type=int, required=True,
                        help='Number of primary electrons in background run')
    parser.add_argument('--beam-current-uA', type=float, default=62.5,
                        help='Beam current in microamps (default: 62.5)')
    parser.add_argument('--signal-pattern', required=True,
                        help='Signal CSV pattern with {ma} placeholder')
    parser.add_argument('--ma-list', nargs='+', type=int, 
                        default=[1, 5, 10, 20, 50, 100, 200, 500],
                        help='List of ALP masses in MeV')
    parser.add_argument('--calo-sigma-a', type=float, default=0.02,
                        help='Calorimeter resolution stochastic term (default: 0.02)')
    parser.add_argument('--calo-sigma-b', type=float, default=0.01,
                        help='Calorimeter resolution constant term (default: 0.01)')
    parser.add_argument('--calo-noise-cut-MeV', type=float, default=5.0,
                        help='Energy threshold cut in MeV (default: 5.0)')
    parser.add_argument('--exposure-days', type=float, default=30.0,
                        help='Exposure time in days. Signal CSV weights are '
                             'already integrated over this exposure (from '
                             'alp_signal_pipeline.py EXPOSURE_DAYS). Background '
                             'is scaled from per-second to exposure total.')
    parser.add_argument('--out-csv', default='output/snr_summary.csv',
                        help='Output summary CSV path')
    parser.add_argument('--plot-dir', default='plots/snr',
                        help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for energy smearing')
    
    args = parser.parse_args()
    
    # Print beam + exposure so every run is self-documenting. S and B are
    # both presented as TOTAL events integrated over EXPOSURE_DAYS at the
    # quoted average beam current.
    eps = (args.beam_current_uA * 1e-6) / CHARGE_COULOMBS
    exposure_s = args.exposure_days * SECONDS_PER_DAY
    print(f"Beam current (avg): {args.beam_current_uA} uA   ->  {eps:.3e} e/s")
    print(f"Exposure:           {args.exposure_days} days  ->  {exposure_s:.3e} s")
    print(f"N primaries (MC):   {args.n_primaries}")
    # Per-primary scale to "events in the full exposure": electrons_in_exposure
    # / n_primaries.
    electrons_in_exposure = eps * exposure_s
    norm_factor = electrons_in_exposure / args.n_primaries
    print(f"Per-primary -> exposure-events scale: {norm_factor:.3e}")

    rng = np.random.default_rng(args.seed)
    
    # Create output directories
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # ─── Load and process background ─────────────────────────────────────────
    print(f"\nLoading background: {args.bkg_csv}")
    bkg_df = load_calo_face(args.bkg_csv, particle_filter=[22, 2112])  # photons + neutrons
    print(f"  Loaded {len(bkg_df)} particles (photons + neutrons)")
    
    # Background MC weights are 1 per entry in electron-beam mode. Scale to
    # "events at the calo face over the full exposure": multiply by
    # (electrons_in_exposure / n_primaries) = norm_factor.
    bkg_weights = bkg_df['weight'].values * norm_factor
    
    # Smear energies
    bkg_E_smear = smear_energy(bkg_df['energy_MeV'].values, 
                               a=args.calo_sigma_a, b=args.calo_sigma_b, rng=rng)
    
    # Apply noise cut
    noise_mask = bkg_E_smear >= args.calo_noise_cut_MeV
    bkg_E_smear = bkg_E_smear[noise_mask]
    bkg_weights = bkg_weights[noise_mask]
    bkg_theta = polar_angle(bkg_df['px'].values[noise_mask],
                            bkg_df['py'].values[noise_mask],
                            bkg_df['pz'].values[noise_mask])
    
    print(f"  After smearing and {args.calo_noise_cut_MeV} MeV cut: {len(bkg_E_smear)} particles")
    
    # ─── Define binning ──────────────────────────────────────────────────────
    # Coarser grid (25x25 = 625 cells) so that the ~12k post-cut background
    # entries populate ~20 cells/cell on average and the S/B landscape is
    # not dominated by empty-background artifacts.
    E_bins = np.logspace(np.log10(args.calo_noise_cut_MeV), np.log10(10000), 26)
    theta_bins = np.linspace(0, np.pi/2, 26)
    
    # Build background histogram (once). Units: "events at calo face over the
    # full exposure".
    B_raw = build_2d_hist(bkg_E_smear, bkg_theta, bkg_weights, E_bins, theta_bins)
    print(f"  Background total (raw): {np.sum(B_raw):.3e} events / {args.exposure_days}-day run")

    # Regularize B:
    #   1. Light Gaussian smooth (sigma=1.5 bins) to fill MC-statistical zeros
    #      near populated regions. The true background is smooth on the scale
    #      of the calorimeter resolution, so adjacent cells correlate. Mode
    #      'reflect' conserves total integral.
    #   2. Local Poisson-1 floor: cells that remain EXACTLY zero after
    #      smoothing (i.e. far from any observed background MC event) get a
    #      floor equal to the weight of a single MC entry. This is the 1-sigma
    #      upper limit on the true rate in an unsampled bin, and it's applied
    #      *only* to empty cells — populated cells keep their measured value.
    B_smoothed = gaussian_filter(B_raw, sigma=1.5, mode='reflect')
    # Per-MC-event weight in the exposure-scaled histogram: one MC entry at
    # the calo face corresponds to `norm_factor` real events (w=1 in bkg CSV).
    b_floor_single_event = float(norm_factor)
    empty_after_smooth = (B_smoothed <= 0)
    B = np.where(empty_after_smooth, b_floor_single_event, B_smoothed)
    n_raw_nz  = int(np.sum(B_raw > 0))
    n_smth_nz = int(np.sum(B_smoothed > 0))
    n_floored = int(np.sum(empty_after_smooth))
    print(f"  Background regularized: {np.sum(B):.3e} events / {args.exposure_days}-day run")
    print(f"    nonzero cells: raw={n_raw_nz}/{B_raw.size}  smoothed={n_smth_nz}/{B_raw.size}"
          f"  floored_empty={n_floored}  (floor={b_floor_single_event:.3e}/cell)")
    
    # ─── Process each mass point ─────────────────────────────────────────────
    results = []
    
    for ma in args.ma_list:
        signal_path = args.signal_pattern.format(ma=ma)
        print(f"\nProcessing ma = {ma} MeV: {signal_path}")
        
        if not Path(signal_path).exists():
            print(f"  WARNING: File not found, skipping")
            continue
        
        # Load signal (photons only - secondaries are background-like)
        sig_df = load_calo_face(signal_path, particle_filter=[22])
        print(f"  Loaded {len(sig_df)} signal photons")
        
        if len(sig_df) == 0:
            print(f"  WARNING: No signal photons, skipping")
            continue
        
        # Signal weights from alp_signal_pipeline.py are already "events in
        # the full EXPOSURE_DAYS run" (despite the misleading
        # `weight_evts_per_day` column name — see pipeline line ~432). Use
        # them directly; no /SECONDS_PER_DAY conversion.
        sig_weights = sig_df['weight'].values
        
        # Smear energies
        sig_E_smear = smear_energy(sig_df['energy_MeV'].values,
                                   a=args.calo_sigma_a, b=args.calo_sigma_b, rng=rng)
        
        # Apply noise cut
        noise_mask = sig_E_smear >= args.calo_noise_cut_MeV
        sig_E_smear = sig_E_smear[noise_mask]
        sig_weights = sig_weights[noise_mask]
        sig_theta = polar_angle(sig_df['px'].values[noise_mask],
                                sig_df['py'].values[noise_mask],
                                sig_df['pz'].values[noise_mask])
        
        print(f"  After smearing and cut: {len(sig_E_smear)} photons")
        
        # Build signal histogram
        S = build_2d_hist(sig_E_smear, sig_theta, sig_weights, E_bins, theta_bins)
        print(f"  Signal total: {np.sum(S):.3e} events / {args.exposure_days}-day run")

        # Per-mass data-driven k grid (percentiles of actual S/B).
        k_grid = data_driven_k_grid(S, B, n=60)

        # Find best S/B threshold
        best_k, best_snr, S_acc, B_acc, best_mask = best_sb_threshold(S, B, k_grid)
        print(f"  Best threshold k = {best_k:.3e}")
        print(f"  S_accepted = {S_acc:.3e}   B_accepted = {B_acc:.3e}   (events / run)")
        print(f"  SNR (S/sqrt(S+B)) = {best_snr:.4g}")

        # Store results
        results.append({
            'ma_MeV': ma,
            'S_exposure': S_acc,
            'B_exposure': B_acc,
            'snr': best_snr,
            'threshold_k': best_k,
            'n_signal_rows': len(sig_df),
            'n_bkg_rows': len(bkg_df),
        })
        
        # Plot cut contour
        if PLOT and best_mask is not None:
            plot_path = plot_dir / f'cut_contour_ma{ma}MeV.png'
            plot_cut_contour(S, B, best_mask, E_bins, theta_bins,
                           ma, best_k, best_snr, plot_path,
                           exposure_days=args.exposure_days,
                           beam_current_uA=args.beam_current_uA)
            print(f"  Saved: {plot_path}")
    
    # ─── Save results ────────────────────────────────────────────────────────
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")
        
        # Plot SNR vs mass
        if PLOT:
            snr_plot_path = plot_dir / 'snr_vs_mass.png'
            plot_snr_vs_mass(results_df, snr_plot_path,
                             exposure_days=args.exposure_days,
                             beam_current_uA=args.beam_current_uA)
            print(f"Saved: {snr_plot_path}")
        
        print("\n=== Summary ===")
        print(results_df.to_string(index=False))
    else:
        print("\nNo results to save (no signal files found)")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
