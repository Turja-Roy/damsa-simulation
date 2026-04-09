#!/usr/bin/env python3
"""Subsample an alplib decay-photon CSV to N nonzero-weight rows.

Uniform random sampling without replacement over the nonzero-weight subset,
with each kept row's weight multiplied by (n_nonzero / n_kept) so the total
weight (events/day) is preserved in expectation. This is the Horvitz-Thompson
estimator for a uniform sample.

Usage:
    python scripts/subsample_alp_csv.py <in.csv> <out.csv> <n_target> [seed]
"""
import sys
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    n_target = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) >= 5 else 12345

    df = pd.read_csv(in_path)
    n_total = len(df)
    nz = df[df["weight_evts_per_day"] > 0].reset_index(drop=True)
    n_nz = len(nz)

    if n_nz <= n_target:
        out = nz.copy()
        factor = 1.0
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_nz, size=n_target, replace=False)
        out = nz.iloc[idx].copy()
        factor = n_nz / n_target
        out["weight_evts_per_day"] = out["weight_evts_per_day"] * factor

    w_in = nz["weight_evts_per_day"].sum()
    w_out = out["weight_evts_per_day"].sum()
    out.to_csv(out_path, index=False)
    print(
        f"{in_path}: total={n_total} nonzero={n_nz} kept={len(out)} "
        f"reweight={factor:.4g} w_in={w_in:.6e} w_out={w_out:.6e}"
    )


if __name__ == "__main__":
    main()
