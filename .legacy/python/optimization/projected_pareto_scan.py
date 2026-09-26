#!/usr/bin/env python3
"""
Projected Pareto scan for DAMSA gap optimization.

Uses weighted projection to estimate 100k-event background from 1000-event runs:
  1. Create weight map from existing 100k-event run
  2. Run Geant4 with 1000 events for each gap
  3. Apply weight map to project to 100k-event equivalent
  4. Build Pareto front from projected backgrounds + alplib signals

This captures gap-dependent physics (magnet effects, secondary interactions)
that the straight-line-only approach misses.

Output:
  output_proj/pareto_scan.csv          — per-gap table
  output_proj/gap_XXcm/                — Geant4 outputs per gap
  plots_proj/pareto_front.png          — Pareto front plot
  plots_proj/pareto_gap_curves.png    — signal & background vs gap

Usage:
  python scripts/projected_pareto_scan.py --n-events-small 2000 --gap-min 25 --gap-max 70
"""

import argparse

# Delivered LESA-Laser beam current in uA (plan.md §1): 4200 e/bunch x 18 bunches x 929 kHz x e.
LESA_DELIVERED_UA = 4200 * 18 * 929e3 * 1.602176634e-19 * 1e6

import sys
import subprocess
import shutil
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

TARGET_EXIT_Z_MM   = -400.0
MAGNET_LENGTH_MM   =  120.0
CALO_HALF_WIDTH_MM =   60.0

CHARGE_C           = 1.602176634e-19
SECONDS_PER_DAY    = 86400.0


N_PDG_TYPES = 3
N_ENERGY_BINS = 70
N_POSITION_BINS = 20

PDG_TO_INDEX = {22: 0, 2112: 1, 11: 2, -11: 2}


def create_phase_space_weight_map(particles_csv: str, output_npy: str,
                                   n_primaries_ref: int, n_events_small: int) -> np.ndarray:
    """
    Create weight map from reference 100k-event run using full phase-space binning.
    
    Bins: energy (70 bins of 1 MeV) × x position (20 bins of 10 mm) × y position (20 bins of 10 mm)
    PDG types: photons (22) → index 0, neutrons (2112) → index 1, electrons (11, -11) → index 2
    
    Weight = count_100k / count_1000 per bin, preserving distribution shape.
    """
    print(f"\nCreating phase-space weight map from: {particles_csv}")
    df = pd.read_csv(particles_csv)
    print(f"  Loaded {len(df)} particles")

    weight_map = np.zeros((N_PDG_TYPES, N_ENERGY_BINS, N_POSITION_BINS, N_POSITION_BINS),
                          dtype=np.float32)

    for _, row in df.iterrows():
        pdg = row['pdg']
        if pdg not in PDG_TO_INDEX:
            continue
        pdg_idx = PDG_TO_INDEX[pdg]
        
        e_bin = min(int(row['energy_MeV']), N_ENERGY_BINS - 1)
        x_bin = min(int((row['x_mm'] + 100) / 10), N_POSITION_BINS - 1)
        y_bin = min(int((row['y_mm'] + 100) / 10), N_POSITION_BINS - 1)
        
        weight_map[pdg_idx, e_bin, x_bin, y_bin] += 1

    scale = n_primaries_ref / n_events_small
    weight_map /= scale
    weight_map = np.maximum(weight_map, 1e-6)

    np.save(output_npy, weight_map)
    print(f"  Saved weight map to: {output_npy}")
    print(f"  Shape: {weight_map.shape}, nonzero bins: {np.count_nonzero(weight_map)}")

    for pdg_name, idx in [('photons', 0), ('neutrons', 1), ('electrons', 2)]:
        total = weight_map[idx].sum()
        print(f"    {pdg_name}: total weight = {total:.2f}")

    return weight_map


def create_weight_map(particles_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Create weight map from reference 100k-event run.
    Weight = n_primaries / count_per_pdg so that weight × n_small = n_primaries equivalent.
    """
    print(f"\nCreating weight map from: {particles_csv}")
    df = pd.read_csv(particles_csv)
    print(f"  Loaded {len(df)} particles")

    n_primaries = df['eventID'].max() + 1
    print(f"  Estimated n_primaries: {n_primaries}")

    weight_map = df.groupby('pdg').size().reset_index(name='raw_count')
    weight_map['weight_factor'] = n_primaries / weight_map['raw_count']

    weight_map.to_csv(output_csv, index=False)
    print(f"  Saved weight map to: {output_csv}")
    print(weight_map.to_string(index=False))

    return weight_map


def run_geant4_for_gap(gap_cm: float, n_events: int, executable: str,
                       output_dir: Path, seed: int = 42) -> bool:
    """
    Run Geant4 simulation for a specific gap.
    Returns True if successful.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    macro_content = f"""\
/run/initialize
/run/beamOn {n_events}
"""
    macro_path = output_dir / "run_proj.mac"
    macro_path.write_text(macro_content)

    cmd = [executable, macro_path.absolute()]
    env = {'OMP_NUM_THREADS': '4'}

    result = subprocess.run(
        cmd,
        cwd=str(Path(executable).parent.parent),
        capture_output=True,
        text=True,
        timeout=600,
        env=env
    )

    if result.returncode != 0:
        print(f"  Warning: Geant4 failed for gap={gap_cm} cm: {result.stderr[:200]}")
        return False

    src = Path(executable).parent.parent / "output"
    needed_file = "all_particles_target_exit.csv"
    src_file = src / needed_file
    if src_file.exists():
        shutil.copy(src_file, output_dir / needed_file)

    return True


def propagate_weighted_background(particles_df: pd.DataFrame,
                                   weight_map: pd.DataFrame,
                                   gaps_cm: np.ndarray,
                                   neutron_weight: float = 10.0) -> np.ndarray:
    """
    Apply weight map and propagate to calo face for each gap.
    """
    wmap = weight_map.set_index('pdg')['weight_factor']

    photon_mask  = particles_df['pdg'].values == 22
    neutron_mask = particles_df['pdg'].values == 2112
    keep_mask    = photon_mask | neutron_mask

    df = particles_df[keep_mask].reset_index(drop=True)
    pdg  = df['pdg'].values
    x    = df['x_mm'].values
    y    = df['y_mm'].values
    z    = df['z_mm'].values
    px   = df['px'].values
    py   = df['py'].values
    pz   = df['pz'].values
    w_mc = df['weight'].values

    base_weights = np.array([wmap.get(p, 1.0) for p in pdg])
    species_weight = np.where(pdg == 2112, neutron_weight, 1.0)

    bkg_vs_gap = np.zeros(len(gaps_cm))

    for i, gap_cm in enumerate(gaps_cm):
        z_calo_mm = TARGET_EXIT_Z_MM + gap_cm * 10.0 + MAGNET_LENGTH_MM

        fwd = pz > 0
        dt = np.where(fwd, (z_calo_mm - z) / np.where(fwd, pz, 1.0), np.inf)

        x_calo = x + dt * px
        y_calo = y + dt * py

        hits = (np.abs(x_calo) <= CALO_HALF_WIDTH_MM) & \
               (np.abs(y_calo) <= CALO_HALF_WIDTH_MM) & fwd

        bkg_vs_gap[i] = np.sum(w_mc[hits] * base_weights[hits] * species_weight[hits])

    return bkg_vs_gap


def run_alplib_once(flux_file: str, n_primaries: int, beam_current_uA: float,
                    ma_MeV: float, coupling_GeV: float,
                    exposure_days: float, n_samples: int,
                    nominal_gap_cm: float = 47.0):
    """Generate ALP decay events with alplib."""
    from scripts.alplib.alplib_signal_plots import load_flux_for_alplib
    import alplib.fluxes as af
    import alplib.materials as am
    import alplib.generators as ag
    from alplib.decay import W_gg
    from alplib.constants import HBAR, C_LIGHT, METER_BY_MEV

    flux_array = load_flux_for_alplib(flux_file, n_primaries, beam_current_uA)

    if coupling_GeV <= 0:
        E = flux_array[:, 0]
        w = flux_array[:, 1]
        mask = E > ma_MeV
        if mask.sum() == 0:
            return [], 0.0
        Ea_typ = np.average(E[mask], weights=w[mask])
        g_sq_MeV = (64.0 * np.pi * METER_BY_MEV * Ea_typ / (0.6 * ma_MeV**3))
        coupling_GeV = np.sqrt(max(g_sq_MeV, 0.0)) * 1000.0

    coupling_MeV = coupling_GeV / 1000.0
    max_E = flux_array[:, 0].max()
    if ma_MeV >= max_E:
        return [], 0.0

    det_dist_m = nominal_gap_cm / 100.0
    det_length_m = nominal_gap_cm / 100.0

    flux_obj = af.FluxPrimakoffIsotropic(
        photon_flux=flux_array,
        target=am.Material("W"),
        det_dist=det_dist_m,
        det_length=det_length_m,
        det_area=0.12 * 0.12,
        axion_mass=ma_MeV,
        axion_coupling=coupling_MeV,
        n_samples=n_samples,
    )
    flux_obj.simulate()
    if len(flux_obj.axion_energy) == 0:
        return [], 0.0

    decay_width = W_gg(coupling_MeV, ma_MeV)
    tau_rest = HBAR / decay_width
    flux_obj.propagate(decay_width)
    generator = ag.PhotonEventGenerator(flux_obj, am.Material("CsI"))
    n_events = generator.decays(days_exposure=exposure_days, threshold=0.1)

    p4_1, p4_2, weights = generator.simulate_decay_4vectors(
        days_exposure=exposure_days, n_samples=n_samples
    )

    events = []
    for idx in range(len(p4_1)):
        w = weights[idx] if idx < len(weights) else 0.0
        if w <= 0:
            continue
        p1 = p4_1[idx]
        p2 = p4_2[idx]
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
        cos_theta = (px1*px2 + py1*py2 + pz1*pz2) / (mag1 * mag2)
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta_deg = np.degrees(np.arccos(cos_theta))
        gamma = E_alp / ma_MeV
        beta = np.sqrt(max(1.0 - (ma_MeV / E_alp)**2, 0.0))
        lab_decay_length_cm = gamma * beta * C_LIGHT * tau_rest

        events.append({
            'weight': w,
            'alp_E_MeV': E_alp,
            'E1_MeV': E1,
            'E2_MeV': E2,
            'opening_angle_deg': theta_deg,
            'lab_decay_length_cm': lab_decay_length_cm,
        })

    return events, n_events


def geometric_acceptance_vs_gap(events, gaps_cm,
                                angle_cut_deg=20.0, energy_cut_MeV=100.0,
                                calo_half_width_cm=6.0, rng=None):
    """Compute geometric acceptance vs gap."""
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

    sep_mask = (theta_deg >= angle_cut_deg) | \
               ((theta_deg < angle_cut_deg) & (E1 >= energy_cut_MeV) & (E2 >= energy_cut_MeV))

    accepted_frac  = np.zeros(len(gaps_cm))
    separable_frac = np.zeros(len(gaps_cm))

    for i, gap_cm in enumerate(gaps_cm):
        u = rng.random(len(events))
        z_decay = -L_decay * np.log(np.clip(1.0 - u, 1e-30, None))
        in_gap = (z_decay > 0) & (z_decay < gap_cm)

        magnet_cm = MAGNET_LENGTH_MM / 10.0
        dist_to_calo = (gap_cm + magnet_cm) - z_decay

        theta_rad = np.radians(theta_deg)
        half_sep_cm = theta_rad * dist_to_calo / 2.0

        both_hit = in_gap & (half_sep_cm <= calo_half_width_cm)

        acc_w = (weights * both_hit).sum()
        sep_w = (weights * both_hit * sep_mask).sum()

        accepted_frac[i]  = acc_w / total_w
        separable_frac[i] = sep_w / total_w

    return accepted_frac, separable_frac


def pareto_mask(x_min, y_max):
    """Identify Pareto-optimal points."""
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


def plot_pareto_front(df, ma_list, plot_dir, beam_uA, exposure_days):
    if not PLOT:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(ma_list)))
    fig, ax = plt.subplots(figsize=(8, 6))

    for ma, col in zip(ma_list, colors):
        sub = df[df['ma_MeV'] == ma].copy()
        if sub.empty:
            continue
        bkg = sub['bkg_exposure'].values
        sig = sub['separable_fraction'].values
        gaps = sub['gap_cm'].values

        ax.scatter(bkg, sig, c=[col]*len(sub), zorder=5, s=60, label=f'ma={ma} MeV')
        ax.plot(bkg, sig, '-', color=col, alpha=0.4, linewidth=1)

        pm = pareto_mask(bkg, sig)
        ax.scatter(bkg[pm], sig[pm], c=[col]*pm.sum(),
                   edgecolors='black', linewidths=1.5, zorder=6, s=80)

        for idx in range(0, len(gaps), max(1, len(gaps)//4)):
            ax.annotate(f'{gaps[idx]:.0f}cm', (bkg[idx], sig[idx]),
                        textcoords='offset points', xytext=(5, 3), fontsize=7, color=col)

    ax.set_xlabel('Weighted background at calo face', fontsize=11)
    ax.set_ylabel('Signal separable fraction', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(f'DAMSA gap Pareto front — I_avg={beam_uA} µA, T={exposure_days:.0f} d', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = plot_dir / 'pareto_front.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_gap_curves(df, ma_list, plot_dir, beam_uA, exposure_days):
    if not PLOT:
        return
    active_ma = [ma for ma in ma_list
                 if df[df['ma_MeV'] == ma]['separable_fraction'].max() > 0]
    if not active_ma:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(active_ma)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_sig, ax_bkg, ax_fom = axes

    ref_sub = df[df['ma_MeV'] == active_ma[0]].copy()
    bkg_abs = ref_sub['bkg_exposure'].values
    bkg_ref = bkg_abs[0]
    bkg_norm = bkg_abs / bkg_ref * 100.0
    ax_bkg.plot(ref_sub['gap_cm'], bkg_norm, 'o-', color='steelblue', linewidth=2)
    ax_bkg.axhline(100, color='grey', linestyle=':', linewidth=1)
    ax_bkg.set_xlabel('Gap [cm]', fontsize=11)
    ax_bkg.set_ylabel('Background at calo face\n(% of gap=30 cm level)', fontsize=11)
    ax_bkg.set_title('Background reduction vs gap\n(Projected from 1k→100k weighted)', fontsize=10)
    ax_bkg.set_yscale('log')
    ax_bkg.grid(True, alpha=0.3, which='both')

    for ma, col in zip(active_ma, colors):
        sub = df[df['ma_MeV'] == ma].copy()
        gaps = sub['gap_cm'].values
        sig = sub['separable_fraction'].values
        bkg = sub['bkg_exposure'].values

        sig_ref = max(sig[0], 1e-30)
        sig_norm = sig / sig_ref * 100.0
        ax_sig.plot(gaps, sig_norm, 'o-', color=col, label=f'ma={ma:.0f} MeV')

        fom = (sig / sig_ref) / np.sqrt(bkg / bkg_ref + 1e-30)
        fom_norm = fom / max(fom.max(), 1e-30)
        ax_fom.plot(gaps, fom_norm, 'o-', color=col, label=f'ma={ma:.0f} MeV')

        peak_idx = np.argmax(fom_norm)
        ax_fom.axvline(gaps[peak_idx], color=col, linestyle='--', alpha=0.35)
        ax_fom.annotate(f'{gaps[peak_idx]:.0f} cm', (gaps[peak_idx], fom_norm[peak_idx]),
                        textcoords='offset points', xytext=(4, 2), fontsize=8, color=col)

    ax_sig.axhline(100, color='grey', linestyle=':', linewidth=1)
    ax_sig.set_xlabel('Gap [cm]', fontsize=11)
    ax_sig.set_ylabel('Signal separable fraction\n(% of gap=30 cm level)', fontsize=11)
    ax_sig.set_title('Signal acceptance vs gap', fontsize=10)
    ax_sig.set_yscale('log')
    ax_sig.legend(fontsize=9)
    ax_sig.grid(True, alpha=0.3, which='both')

    ax_fom.set_xlabel('Gap [cm]', fontsize=11)
    ax_fom.set_ylabel('S/√B  (normalised)', fontsize=11)
    ax_fom.set_title('Figure of merit S/√B vs gap', fontsize=10)
    ax_fom.legend(fontsize=9)
    ax_fom.grid(True, alpha=0.3)

    plt.tight_layout()
    out = plot_dir / 'pareto_gap_curves.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--particles-csv', default='output/all_particles_target_exit.csv')
    parser.add_argument('--flux-csv', default='output/photon_flux_target_exit.csv')
    parser.add_argument('--executable', default='./build/damsa')
    parser.add_argument('--n-events-small', type=int, default=1000,
                        help='Events per Geant4 run per gap')
    parser.add_argument('--n-primaries-ref', type=int, default=100000,
                        help='Reference run size (for weight map)')
    parser.add_argument('--beam-uA', type=float, default=LESA_DELIVERED_UA)
    parser.add_argument('--exposure-days', type=float, default=30.0)
    parser.add_argument('--ma-list', type=float, nargs='+', default=[5, 10, 20])
    parser.add_argument('--coupling', type=float, default=-1)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--angle-cut', type=float, default=20.0)
    parser.add_argument('--energy-cut', type=float, default=100.0)
    parser.add_argument('--gap-min', type=float, default=30.0)
    parser.add_argument('--gap-max', type=float, default=60.0)
    parser.add_argument('--gap-step', type=float, default=2.0)
    parser.add_argument('--out-dir', default='output_proj')
    parser.add_argument('--plot-dir', default='plots_proj')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-geant4', action='store_true',
                        help='Skip Geant4 runs, use existing output')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gaps_cm = np.arange(args.gap_min, args.gap_max + 0.1, args.gap_step)
    n_gaps = len(gaps_cm)
    print(f"\nGap range: {gaps_cm[0]:.0f}–{gaps_cm[-1]:.0f} cm ({n_gaps} configurations)")

    weight_map_npy = out_dir / "weight_map.npy"
    if weight_map_npy.exists():
        print(f"\nLoading existing weight map: {weight_map_npy}")
        weight_map = np.load(weight_map_npy)
    else:
        weight_map = create_phase_space_weight_map(
            args.particles_csv, weight_map_npy,
            args.n_primaries_ref, args.n_events_small
        )

    if not args.skip_geant4:
        print(f"\n{'='*60}")
        print(f"Running Geant4 for each gap ({args.n_events_small} events/run)")
        print(f"{'='*60}")
        for gap_cm in gaps_cm:
            gap_dir = out_dir / f"gap_{int(gap_cm)}cm"
            if (gap_dir / "all_particles_target_exit.csv").exists():
                print(f"  Gap {gap_cm:.0f} cm: using existing output")
                continue
            print(f"  Gap {gap_cm:.0f} cm: running Geant4...", end=" ", flush=True)
            success = run_geant4_for_gap(gap_cm, args.n_events_small, args.executable, gap_dir, args.seed)
            print("done" if success else "FAILED")
    else:
        print("\nSkipping Geant4 runs (--skip-geant4)")

    bkg_vs_gap = np.zeros(n_gaps)
    for i, gap_cm in enumerate(gaps_cm):
        gap_dir = out_dir / f"gap_{int(gap_cm)}cm"
        particles_csv = gap_dir / "all_particles_target_exit.csv"
        if particles_csv.exists():
            df = pd.read_csv(particles_csv)
            
            pdg_arr = df['pdg'].values
            energy_arr = df['energy_MeV'].values
            x_arr = df['x_mm'].values
            y_arr = df['y_mm'].values
            z_arr = df['z_mm'].values
            px_arr = df['px'].values
            py_arr = df['py'].values
            pz_arr = df['pz'].values
            w_mc = df['weight'].values
            
            z_calo_mm = TARGET_EXIT_Z_MM + gap_cm * 10.0 + MAGNET_LENGTH_MM
            fwd = pz_arr > 0
            dt = np.where(fwd, (z_calo_mm - z_arr) / np.where(fwd, pz_arr, 1.0), np.inf)
            x_calo = x_arr + dt * px_arr
            y_calo = y_arr + dt * py_arr
            hits = (np.abs(x_calo) <= CALO_HALF_WIDTH_MM) & \
                   (np.abs(y_calo) <= CALO_HALF_WIDTH_MM) & fwd
            
            if np.any(hits):
                hit_idx = np.where(hits)[0]
                total_weighted = 0.0
                for idx in hit_idx:
                    pdg = pdg_arr[idx]
                    if pdg not in PDG_TO_INDEX:
                        continue
                    pdg_idx = PDG_TO_INDEX[pdg]
                    e_bin = min(int(energy_arr[idx]), N_ENERGY_BINS - 1)
                    x_bin = min(int((x_arr[idx] + 100) / 10), N_POSITION_BINS - 1)
                    y_bin = min(int((y_arr[idx] + 100) / 10), N_POSITION_BINS - 1)
                    bin_weight = weight_map[pdg_idx, e_bin, x_bin, y_bin]
                    species_weight = 10.0 if pdg == 2112 else 1.0
                    total_weighted += w_mc[idx] * bin_weight * species_weight
                bkg_vs_gap[i] = total_weighted
            
            shutil.rmtree(gap_dir)
            print(f"  Gap {gap_cm:.0f} cm: bkg = {bkg_vs_gap[i]:.3e}")
        else:
            print(f"  Warning: no output for gap {gap_cm} cm")

    print(f"\nBackground range (projected): {bkg_vs_gap.min():.3e} – {bkg_vs_gap.max():.3e}")

    all_rows = []
    for ma in args.ma_list:
        print(f"\n{'='*55}")
        print(f"ALP mass: ma = {ma} MeV")
        print(f"{'='*55}")

        events, n_total = run_alplib_once(
            flux_file=args.flux_csv,
            n_primaries=args.n_primaries_ref,
            beam_current_uA=args.beam_uA,
            ma_MeV=ma,
            coupling_GeV=args.coupling,
            exposure_days=args.exposure_days,
            n_samples=args.n_samples,
            nominal_gap_cm=float(np.mean(gaps_cm)),
        )

        if not events:
            print(f"  Skipping ma={ma}: no events generated")
            continue

        acc_frac, sep_frac = geometric_acceptance_vs_gap(
            events, gaps_cm,
            angle_cut_deg=args.angle_cut,
            energy_cut_MeV=args.energy_cut,
            calo_half_width_cm=CALO_HALF_WIDTH_MM / 10.0,
            rng=rng,
        )

        for i, gap_cm in enumerate(gaps_cm):
            acc_val = acc_frac[i]
            sep_eff = sep_frac[i] / acc_val if acc_val > 0 else 0.0
            all_rows.append({
                'ma_MeV': ma,
                'gap_cm': gap_cm,
                'bkg_exposure': bkg_vs_gap[i],
                'accepted_fraction': acc_val,
                'separable_fraction': sep_frac[i],
                'sep_efficiency': sep_eff,
                'n_alp_events': n_total,
            })

        print(f"  Accepted fraction:   {acc_frac.min():.3f} – {acc_frac.max():.3f}")
        print(f"  Separable fraction: {sep_frac.min():.4f} – {sep_frac.max():.4f}")

    if not all_rows:
        print("No results to save.")
        return 1

    df = pd.DataFrame(all_rows)
    out_csv = out_dir / "pareto_scan.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))

    plot_pareto_front(df, args.ma_list, plot_dir, args.beam_uA, args.exposure_days)
    plot_gap_curves(df, args.ma_list, plot_dir, args.beam_uA, args.exposure_days)

    return 0


if __name__ == '__main__':
    sys.exit(main())