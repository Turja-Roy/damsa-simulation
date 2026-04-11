#!/usr/bin/env python3
"""
Combine per-target-length Pareto CSVs into a global 3D Pareto front.

After running joint_pareto_scan.py for each target length, this script:
  1. Loads all <label>_pareto.csv files
  2. Tags each row with its target length (parsed from the label, e.g. 'Tz14' → 14 cm)
  3. Computes a global Pareto front across all (target, VDC, calo) combinations
  4. Saves the combined result and produces summary plots

Usage:
    python scripts/combine_pareto.py output/joint_pareto/Tz*_pareto.csv
    python scripts/combine_pareto.py output/joint_pareto/Tz*_pareto.csv \\
        --output output/global_pareto.csv \\
        --plot-dir plots/global_pareto
"""

import argparse
import re
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


def parse_target_length(label_or_path: str) -> float:
    """Extract target length in cm from a label like 'Tz14' or file path."""
    match = re.search(r'Tz(\d+(?:\.\d+)?)', label_or_path)
    if match:
        return float(match.group(1))
    raise ValueError(f"Cannot parse target length from '{label_or_path}'. "
                     f"Expected pattern 'Tz<N>' in filename.")


def is_pareto_optimal(objectives: np.ndarray, minimize: list) -> np.ndarray:
    """Non-dominated sort. objectives shape (N, k); minimize is list of bool."""
    N  = len(objectives)
    obj = objectives.copy()
    for j, mn in enumerate(minimize):
        if not mn:
            obj[:, j] = -obj[:, j]

    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        dominated = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False

    return is_pareto


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pareto_files', nargs='+',
                        help='Per-target-length pareto CSV files (e.g. Tz*_pareto.csv)')
    parser.add_argument('--output',   default='output/global_pareto.csv')
    parser.add_argument('--plot-dir', default='plots/global_pareto')
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Load and tag ──────────────────────────────────────────────────────────
    frames = []
    for fpath in sorted(args.pareto_files):
        try:
            target_cm = parse_target_length(fpath)
        except ValueError as e:
            print(f"WARNING: {e} — skipping")
            continue
        df = pd.read_csv(fpath)
        df['target_cm'] = target_cm
        frames.append(df)
        print(f"Loaded {fpath}: {len(df)} rows, target={target_cm:.0f} cm")

    if not frames:
        print("No valid files loaded.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows total")

    # ── Global Pareto per mass ────────────────────────────────────────────────
    # Objectives: minimize bkg, maximize separable_fraction
    global_pareto_rows = []
    for ma in sorted(combined['ma_MeV'].unique()):
        sub  = combined[combined['ma_MeV'] == ma].reset_index(drop=True)
        obj  = sub[['bkg_exposure', 'separable_fraction']].values
        mask = is_pareto_optimal(obj, minimize=[True, False])
        sub_pareto = sub[mask].copy()
        sub_pareto['global_pareto'] = True
        global_pareto_rows.append(sub_pareto)

    global_df = pd.concat(global_pareto_rows, ignore_index=True)

    # Add FoM if not already present
    if 'fom' not in global_df.columns:
        global_df['fom'] = global_df['separable_fraction'] / \
                           np.sqrt(global_df['bkg_exposure'] + 1e-30)

    global_df.to_csv(args.output, index=False)
    print(f"\nGlobal Pareto saved: {args.output}  ({len(global_df)} rows)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Best configuration per mass (highest FoM on global Pareto) ===")
    print(f"{'ma [MeV]':>10} {'target [cm]':>12} {'VDC [cm]':>9} "
          f"{'calo [cm]':>10} {'sep_frac':>10} {'bkg':>12} {'FoM':>12}")
    print("-" * 77)
    for ma in sorted(global_df['ma_MeV'].unique()):
        sub  = global_df[global_df['ma_MeV'] == ma]
        best = sub.loc[sub['fom'].idxmax()]
        print(f"{ma:10.0f} {best['target_cm']:12.0f} {best['vdc_cm']:9.0f} "
              f"{best['calo_cm']:10.0f} {best['separable_fraction']:10.4f} "
              f"{best['bkg_exposure']:12.3e} {best['fom']:12.4e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not PLOT:
        return

    masses  = sorted(global_df['ma_MeV'].unique())
    targets = sorted(global_df['target_cm'].unique())
    colors  = plt.cm.tab10(np.linspace(0, 1, len(masses)))

    # Global Pareto front coloured by mass
    fig, ax = plt.subplots(figsize=(9, 6))
    for ma, col in zip(masses, colors):
        sub = global_df[global_df['ma_MeV'] == ma]
        ax.scatter(sub['bkg_exposure'], sub['separable_fraction'],
                   c=[col] * len(sub), label=f'ma={ma:.0f} MeV',
                   edgecolors='black', linewidths=0.7, s=60, zorder=5)
        for _, row in sub.iterrows():
            ax.annotate(
                f"T{row['target_cm']:.0f}/V{row['vdc_cm']:.0f}/C{row['calo_cm']:.0f}",
                (row['bkg_exposure'], row['separable_fraction']),
                textcoords='offset points', xytext=(4, 2),
                fontsize=6, color=col,
            )

    ax.set_xlabel('Weighted background (full exposure)', fontsize=11)
    ax.set_ylabel('Signal separable fraction', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(
        'Global Pareto front — all (target, VDC, calo) combinations\n'
        'Labels: T=target cm / V=VDC cm / C=calo cm',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = plot_dir / 'global_pareto_front.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # Per-mass: signal fraction vs target length (bar chart)
    if len(targets) > 1:
        fig, axes = plt.subplots(1, len(masses), figsize=(3 * len(masses), 4),
                                 sharey=False)
        if len(masses) == 1:
            axes = [axes]

        for ax, ma, col in zip(axes, masses, colors):
            sub = global_df[global_df['ma_MeV'] == ma]
            best_per_target = sub.groupby('target_cm')['fom'].idxmax()
            bt = sub.loc[best_per_target].sort_values('target_cm')

            ax.bar(bt['target_cm'].astype(str), bt['separable_fraction'],
                   color=col, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Target length [cm]', fontsize=9)
            ax.set_ylabel('sep. efficiency', fontsize=9)
            ax.set_title(f'ma={ma:.0f} MeV', fontsize=9)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Best separability efficiency vs target length (per mass)',
                     fontsize=10)
        plt.tight_layout()
        out2 = plot_dir / 'sep_eff_vs_target.png'
        plt.savefig(out2, dpi=130)
        plt.close()
        print(f"Saved: {out2}")


if __name__ == '__main__':
    main()
