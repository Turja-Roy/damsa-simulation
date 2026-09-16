#!/usr/bin/env python3
"""Geant4-viewer-style renders of ALP signal: a -> gamma gamma decays.

Reads alp_decay_photons_ma<M>MeV.csv (from alp_signal_pipeline.py
export_decay_4vectors) and produces, per mass:
  <out>/alp_ma<M>MeV_g4view3d.png   3D render, beam +z to the right
  <out>/alp_ma<M>MeV_g4side.png     x-z side view with component spans

Physics used for the display:
  - ALPs are produced forward (theta_ALP = 0, alplib convention) at the
    target centre, z = -450 mm — same vertex damsa_alp_inject uses.
  - Per-row ALP 4-vector = sum of the two photon 4-vectors; lab decay
    length L = (p_a/m_a) * hbar*c / Gamma, Gamma = g^2 m^3 / 64pi, with g
    re-derived exactly as pick_safe_coupling() did for the export
    (needs the same brems flux CSV).
  - Decay vertex sampled from the exponential decay law truncated to
    [production, calo face] — i.e. "given this ALP decays before the calo,
    where?". Row selection is weighted by the alplib event weight.
Photon color: red/orange = projects into calo aperture, gray = misses.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi0_vertices_g4view import (Geometry, draw_geometry_3d,
                                 checkpoint_labels_3d, BG, FG,
                                 C_G1_GEOM, C_G2_GEOM, C_MISS, C_BEAM)

HBARC_MEV_M = 197.3269804e-15   # hbar*c [MeV*m]
C_ALP = "#66ff99"               # ALP flight path


def safe_coupling_GeV(ma_MeV, flux_csv, target_L_m=5.0):
    """Reproduce alp_signal_pipeline.pick_safe_coupling for the export."""
    dat = np.loadtxt(flux_csv, delimiter=",", comments="#")
    E, w = dat[:, 0], dat[:, 1]
    m = E > ma_MeV
    if m.sum() == 0:
        return 1e-4
    Ea_typ = np.average(E[m], weights=w[m])
    g_sq_MeV = 64.0 * np.pi * HBARC_MEV_M * Ea_typ / (target_L_m * ma_MeV**4)
    return np.sqrt(g_sq_MeV) * 1000.0


def load_pairs(csv_path, ma_MeV, g_GeV, n_draw, rng, prod_z, calo_z):
    """Weighted subsample; returns DataFrame with decay vertex + kinematics."""
    df = pd.read_csv(csv_path)
    df = df[df.weight_evts_per_day > 0].reset_index(drop=True)

    # ALP 4-vector (alplib fires forward: transverse components cancel)
    Ea = df.E1_MeV + df.E2_MeV
    pz = df.pz1 + df.pz2
    pa = np.sqrt(np.maximum((df.px1 + df.px2)**2 + (df.py1 + df.py2)**2
                            + pz**2, 1e-30))
    gamma_mev = (g_GeV / 1000.0)**2 * ma_MeV**3 / (64.0 * np.pi)
    L_mm = (pa / ma_MeV) * (HBARC_MEV_M / gamma_mev) * 1000.0  # lab decay length

    # Row weight = alplib decay_axion_weight — surv & decay probability are
    # ALREADY folded in; do not re-apply a decay factor here.
    span = calo_z - prod_z
    w = df.weight_evts_per_day.to_numpy()
    if w.sum() <= 0:
        return None, 0.0
    L_wmean = float(np.average(L_mm, weights=w))

    idx = rng.choice(len(df), size=min(n_draw, len(df)), replace=False,
                     p=w / w.sum())
    s = df.iloc[idx].copy()
    Ls = L_mm.iloc[idx].to_numpy()
    if "decay_z_m" in s.columns:
        # pipeline-sampled vertex (metres from target centre) — use directly
        s["decay_z"] = prod_z + s["decay_z_m"] * 1000.0
    else:
        # legacy 9-column CSV: sample truncated exponential here
        u = rng.uniform(size=len(s))
        d = -Ls * np.log1p(-u * (1.0 - np.exp(-span / Ls)))
        s["decay_z"] = prod_z + d
    s["L_mm"] = Ls
    for g in (1, 2):
        E = s[f"E{g}_MeV"]
        for c in ("px", "py", "pz"):
            s[f"{c}{g}u"] = s[f"{c}{g}"] / E    # ~unit direction (massless)
    # opening angle
    dot = (s.px1u * s.px2u + s.py1u * s.py2u + s.pz1u * s.pz2u)
    s["open_deg"] = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    return s, L_wmean


def photon_endpoint(zv, pxu, pyu, pzu, geom, stub=250.0):
    if pzu > 1e-9:
        t = (geom.calo0 - zv) / pzu
        if 0 < t < 3000.0:
            x, y = t * pxu, t * pyu
            ok = abs(x) <= geom.calo_hxy and abs(y) <= geom.calo_hxy
            return x, y, geom.calo0, ok
    return stub * pxu, stub * pyu, zv + stub * pzu, False


def render(sub, geom, ma, g_GeV, L_mean_mm, out3d, outside):
    prod_z = geom.tgt0 + 50.0
    frac_vdc = float(((sub.decay_z >= geom.vdc0)
                      & (sub.decay_z <= geom.vdc1)).mean())
    stats = (f"$m_a$ = {ma:g} MeV,  $g_{{a\\gamma\\gamma}}$ = {g_GeV:.2e}"
             f" GeV$^{{-1}}$,  mean $L_{{lab}}$ = {L_mean_mm/1000:.2f} m,"
             f"  {frac_vdc*100:.0f}% of drawn decays in VDC,"
             f"  median $\\theta_{{\\gamma\\gamma}}$ ="
             f" {sub.open_deg.median():.1f}$^\\circ$")

    # ── 3D ──
    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    ax.set_axis_off()
    draw_geometry_3d(ax, geom)
    checkpoint_labels_3d(ax, geom, y_lab=205.0)

    segs, cols, lws = [], [], []
    for _, d in sub.iterrows():
        zv = d.decay_z
        segs.append([(prod_z, 0, 0), (zv, 0, 0)])      # ALP flight
        cols.append(C_ALP + "cc")
        lws.append(1.0)
        for g, cg in ((1, C_G1_GEOM), (2, C_G2_GEOM)):
            x, y, z, ok = photon_endpoint(zv, d[f"px{g}u"], d[f"py{g}u"],
                                          d[f"pz{g}u"], geom)
            c = cg if ok else C_MISS
            segs.append([(zv, 0, 0), (z, x, y)])
            cols.append(c + ("30" if c == C_MISS else "ff"))
            lws.append(0.5 if c == C_MISS else 1.0)
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=lws))
    ax.scatter(sub.decay_z, np.zeros(len(sub)), np.zeros(len(sub)),
               s=6, c=C_ALP, depthshade=False)

    zlo, zhi = geom.tgt0 - 140, geom.calo1 + 40
    r = 200.0
    ax.set_xlim(zlo, zhi); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_box_aspect((zhi - zlo, 2 * r, 2 * r), zoom=1.35)
    ax.view_init(elev=18, azim=-83)
    fig.text(0.02, 0.965, r"ALP signal: $a\to\gamma\gamma$  (beam"
             r" $+z\ \rightarrow$)   " + stats, color=FG, fontsize=11)
    fig.text(0.02, 0.045,
             f"{len(sub)} weighted-sampled decays.  green = ALP flight from"
             " target centre + decay vertex;  red/orange ="
             " $\\gamma_1$/$\\gamma_2$ into calo aperture;  gray = misses."
             "  Vertex z ~ truncated exponential (decay-before-calo).",
             color=FG, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out3d, dpi=170, facecolor=BG)
    plt.close(fig)

    # ── side view ──
    fig, ax = plt.subplots(figsize=(14, 5.6), facecolor=BG)
    ax.set_facecolor(BG)

    def rect(z0, z1, h, color, alpha):
        ax.add_patch(Rectangle((z0, -h), z1 - z0, 2 * h, facecolor=color,
                               edgecolor=color, alpha=alpha, lw=1.0))

    rect(geom.tgt0, geom.tgt1, geom.tgt_hxy, "#8c8c8c", 0.75)
    rout = geom.vdc_rin + geom.vdc_wall
    rect(geom.vdc0, geom.vdc1, rout, "#e6d800", 0.10)
    for r0, r1 in ((geom.vdc_rin, rout), (-rout, -geom.vdc_rin)):
        ax.fill_between([geom.vdc0, geom.vdc1], r0, r1, color="#e6d800",
                        alpha=0.6)
    rect(geom.mag0, geom.mag1, geom.mag_hxy, "#888888", 0.15)
    for zt in geom.tracker_z():
        rect(zt - 1, zt + 1, geom.trk_hxy, "#00e6e6", 0.8)
    ax.plot([geom.calo0, geom.calo0], [-geom.calo_hxy, geom.calo_hxy],
            color="#00ff55", lw=1.4)
    rect(geom.calo0, geom.calo1, geom.calo_hxy, "#e600e6", 0.30)

    for _, d in sub.iterrows():
        zv = d.decay_z
        ax.plot([prod_z, zv], [0, 0], color=C_ALP, lw=0.8, alpha=0.8)
        for g, cg in ((1, C_G1_GEOM), (2, C_G2_GEOM)):
            x, y, z, ok = photon_endpoint(zv, d[f"px{g}u"], d[f"py{g}u"],
                                          d[f"pz{g}u"], geom)
            c = cg if ok else C_MISS
            ax.plot([zv, z], [0, x], color=c, lw=0.4 if c == C_MISS else 0.9,
                    alpha=0.15 if c == C_MISS else 0.95, zorder=2)
    ax.scatter(sub.decay_z, np.zeros(len(sub)), s=9, c=C_ALP, zorder=3)

    y_span = 235
    for name, z0, z1, col in geom.spans():
        for z in (z0, z1):
            ax.axvline(z, color=col, lw=0.7, ls=":", alpha=0.8)
        ax.annotate("", xy=(z1, y_span), xytext=(z0, y_span),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
        ax.text(0.5 * (z0 + z1), y_span + 8,
                f"{name}  ({(z1 - z0) / 10:.0f} cm)", color=col, fontsize=9,
                ha="center")
    ax.annotate("", xy=(geom.tgt0, 0), xytext=(geom.tgt0 - 130, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_BEAM, lw=1.6))
    ax.text(geom.tgt0 - 130, 12, "e$^-$ 8 GeV", color=C_BEAM, fontsize=10)

    proxies = [
        Line2D([], [], color=C_ALP, marker="o", ls="-", ms=5,
               label=r"ALP flight + decay vertex"),
        Line2D([], [], color=C_G1_GEOM, lw=1.2, label=r"$\gamma_1$ into calo"),
        Line2D([], [], color=C_G2_GEOM, lw=1.2, label=r"$\gamma_2$ into calo"),
        Line2D([], [], color=C_MISS, lw=1.2, alpha=0.6,
               label=r"$\gamma$ misses calo (3D test, may miss in $y$)"),
    ]
    leg = ax.legend(handles=proxies, loc="lower left", fontsize=8,
                    facecolor=BG, edgecolor=FG, framealpha=0.7)
    for t in leg.get_texts():
        t.set_color(FG)

    ax.set_xlim(geom.tgt0 - 160, geom.calo1 + 40)
    ax.set_ylim(-y_span - 35, y_span + 45)
    ax.set_xlabel("z [mm]  (beam direction →)", color=FG)
    ax.set_ylabel("x [mm]", color=FG)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(FG)
    ax.set_title(r"ALP $a\to\gamma\gamma$ — side view (x–z).  " + stats,
                 color=FG, fontsize=10)
    fig.tight_layout()
    fig.savefig(outside, dpi=170, facecolor=BG)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="*",
                   default=sorted(glob.glob("output/alp_decay_photons_ma*MeV.csv")),
                   help="ALP decay CSVs (default: all in output/)")
    p.add_argument("--flux", default="output/alplib_brems_flux.csv",
                   help="brems flux CSV used for the export (coupling rederivation)")
    p.add_argument("--out-dir", default="plots/png/alp")
    p.add_argument("--n-draw", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-len-mm", type=float, default=100.0)
    p.add_argument("--vdc-len-mm", type=float, default=300.0)
    p.add_argument("--calo-xy-mm", type=float, default=120.0)
    args = p.parse_args()

    geom = Geometry(args.target_len_mm, 50.0, args.vdc_len_mm, args.calo_xy_mm)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prod_z = geom.tgt0 + args.target_len_mm / 2.0

    for path in args.csvs:
        m = re.search(r"ma([\d.]+)MeV", os.path.basename(path))
        if not m:
            print(f"[skip] {path}: cannot parse mass", file=sys.stderr)
            continue
        ma = float(m.group(1))
        g = safe_coupling_GeV(ma, args.flux)
        sub, L_mean = load_pairs(path, ma, g, args.n_draw, rng,
                                 prod_z, geom.calo0)
        if sub is None:
            print(f"[skip] {path}: zero total display weight", file=sys.stderr)
            continue
        o3 = os.path.join(args.out_dir, f"alp_ma{ma:g}MeV_g4view3d.png")
        osd = os.path.join(args.out_dir, f"alp_ma{ma:g}MeV_g4side.png")
        render(sub, geom, ma, g, L_mean, o3, osd)
        print(f"[ok] ma={ma:g} MeV  g={g:.2e}/GeV  meanL={L_mean/1000:.2f} m"
              f"  median open={sub.open_deg.median():.1f} deg -> {o3}")


if __name__ == "__main__":
    main()
