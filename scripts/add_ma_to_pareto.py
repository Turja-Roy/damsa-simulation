#!/usr/bin/env python3
"""
Add ma=50 to existing Pareto scan without re-running Geant4.
Uses existing background data from output_proj/pareto_scan.csv and
computes signal via alplib for ma=50.
"""

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

TARGET_EXIT_Z_MM = -400.0
MAGNET_LENGTH_MM = 120.0
CALO_HALF_WIDTH_MM = 60.0


def run_alplib_once(flux_file: str, n_primaries: int, beam_current_uA: float,
                    ma_MeV: float, coupling_GeV: float,
                    exposure_days: float, n_samples: int,
                    nominal_gap_cm: float = 47.0):
    """Generate ALP decay events with alplib."""
    from scripts.alplib_signal_plots import load_flux_for_alplib
    import alplib.fluxes as af
    import alplib.materials as am
    import alplib.generators as ag
    from alplib.decay import W_gg
    from alplib.constants import HBAR, C_LIGHT, METER_BY_MEV

    print(f"  Running alplib for ma={ma_MeV} MeV...")
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
        print(f"    ma={ma_MeV} MeV >= max photon E={max_E:.1f} MeV — no production")
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

    print(f"    Generated {len(events)} events")
    return events, n_events


def geometric_acceptance_vs_gap(events, gaps_cm,
                                angle_cut_deg=20.0, energy_cut_MeV=100.0,
                                calo_half_width_cm=6.0, rng=None):
    """Compute geometric acceptance vs gap."""
    if rng is None:
        rng = np.random.default_rng(42)

    weights = np.array([e['weight'] for e in events])
    theta_deg = np.array([e['opening_angle_deg'] for e in events])
    E1 = np.array([e['E1_MeV'] for e in events])
    E2 = np.array([e['E2_MeV'] for e in events])
    L_decay = np.array([e['lab_decay_length_cm'] for e in events])

    total_w = weights.sum()
    if total_w <= 0:
        return np.zeros(len(gaps_cm)), np.zeros(len(gaps_cm))

    sep_mask = (theta_deg >= angle_cut_deg) | \
               ((theta_deg < angle_cut_deg) & (E1 >= energy_cut_MeV) & (E2 >= energy_cut_MeV))

    accepted_frac = np.zeros(len(gaps_cm))
    separable_frac = np.zeros(len(gaps_cm))

    for i, gap_cm in enumerate(gaps_cm):
        u = rng.random(len(events))
        z_decay = -L_decay * np.log(np.clip(1.0 - u, 1e-30, None))
        in_gap = z_decay < gap_cm
        magnet_cm = MAGNET_LENGTH_MM / 10.0
        dist_to_calo = (gap_cm + magnet_cm) - z_decay
        theta_rad = np.radians(theta_deg)
        half_sep_cm = theta_rad * dist_to_calo / 2.0
        both_hit = in_gap & (half_sep_cm <= calo_half_width_cm)
        acc_w = (weights * both_hit).sum()
        sep_w = (weights * both_hit * sep_mask).sum()
        accepted_frac[i] = acc_w / total_w
        separable_frac[i] = sep_w / total_w

    return accepted_frac, separable_frac


def pareto_mask(x_min, y_max):
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Add ma=50 to existing Pareto scan')
    parser.add_argument('--csv', default='output_proj/pareto_scan.csv')
    parser.add_argument('--flux-csv', default='output/photon_flux_target_exit.csv')
    parser.add_argument('--n-primaries', type=int, default=100000)
    parser.add_argument('--beam-uA', type=float, default=62.5)
    parser.add_argument('--exposure-days', type=float, default=30.0)
    parser.add_argument('--ma-list', type=float, nargs='+', default=[50])
    parser.add_argument('--coupling', type=float, default=-1)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--angle-cut', type=float, default=20.0)
    parser.add_argument('--energy-cut', type=float, default=100.0)
    parser.add_argument('--plot-dir', default='plots_proj')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"\nLoading existing CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    existing_mas = df['ma_MeV'].unique()
    gaps_cm = df[df['ma_MeV'] == existing_mas[0]]['gap_cm'].values
    n_gaps = len(gaps_cm)
    print(f"  Found {len(existing_mas)} masses: {existing_mas}")
    print(f"  Gaps: {gaps_cm[0]:.0f}–{gaps_cm[-1]:.0f} cm ({n_gaps} configurations)")

    rng = np.random.default_rng(args.seed)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    for ma in args.ma_list:
        print(f"\n{'='*55}")
        print(f"ALP mass: ma = {ma} MeV")
        print(f"{'='*55}")

        events, n_total = run_alplib_once(
            flux_file=args.flux_csv,
            n_primaries=args.n_primaries,
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
            bkg = df[(df['ma_MeV'] == existing_mas[0]) & (df['gap_cm'] == gap_cm)]['bkg_exposure'].values[0]
            new_rows.append({
                'ma_MeV': ma,
                'gap_cm': gap_cm,
                'bkg_exposure': bkg,
                'accepted_fraction': acc_frac[i],
                'separable_fraction': sep_frac[i],
                'n_alp_events': n_total,
            })

        print(f"  Separable fraction: {sep_frac.min():.4f} – {sep_frac.max():.4f}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(args.csv, index=False)
        print(f"\nUpdated: {args.csv}")
        print(df_new.to_string(index=False))

        all_mas = sorted(df['ma_MeV'].unique())
        plot_pareto_front(df, all_mas, plot_dir, args.beam_uA, args.exposure_days)
    else:
        print("No new rows to add.")


if __name__ == '__main__':
    sys.exit(main())