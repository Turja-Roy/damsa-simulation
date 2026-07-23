#!/usr/bin/env python3
"""Geant4-viewer-style renders of pi0 -> gamma gamma decay vertices and tracks.

Reads pi0_decays.csv (from DamsaPi0Collector::WriteCSV) and produces, per input:
  <out>/<prefix>_decay_vertices_g4view3d.png   3D render, beam +z to the right
  <out>/<prefix>_decay_vertices_g4side.png     x-z side view with component spans

Geometry defaults mirror construction.cpp (build starts at z = -50 cm):
  target [-500,-400] mm, VDC cylinder, magnet+trackers, monolithic CsI ECAL.
Override with the CLI flags if the run used a scanned geometry.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# ── Geant4 vis colors (construction.cpp G4VisAttributes) ────────────────────
C_TARGET  = "#8c8c8c"   # tungsten gray
C_VDC     = "#e6d800"   # chamber yellow
C_TRACKER = "#00e6e6"   # silicon cyan
C_ECAL    = "#e600e6"   # CsI magenta
C_SCORE   = "#00ff55"   # calo-entrance scoring green
C_BEAM    = "#4488ff"
C_G1_GEOM, C_G1_CALO = "#ff3333", "#990000"
C_G2_GEOM, C_G2_CALO = "#ff9933", "#b35900"
C_MISS = "#777777"
BG = "black"
FG = "white"


class Geometry:
    """z positions in mm, world frame. Mirrors construction.cpp defaults."""

    def __init__(self, target_len=100.0, target_xy=50.0, vdc_len=300.0,
                 calo_xy=120.0, build_start=-500.0):
        self.tgt0 = build_start
        self.tgt1 = build_start + target_len
        self.tgt_hxy = target_xy / 2.0
        self.vdc0, self.vdc1 = self.tgt1, self.tgt1 + vdc_len
        self.vdc_rin, self.vdc_wall = 100.0, 5.0
        self.mag0, self.mag1 = self.vdc1, self.vdc1 + 120.0
        self.mag_hxy = 50.0            # hollow half-XY
        self.n_trackers, self.trk_hxy, self.trk_thick = 6, 49.0, 2.0
        self.calo0 = self.mag1
        self.calo1 = self.mag1 + 440.0  # 44 layers x 1 cm
        self.calo_hxy = calo_xy / 2.0

    def tracker_z(self):
        total = self.mag1 - self.mag0
        gap = (total - self.n_trackers * self.trk_thick) / (self.n_trackers + 1)
        return [self.mag0 + (i + 1) * gap + i * self.trk_thick + self.trk_thick / 2.0
                for i in range(self.n_trackers)]

    def spans(self):
        return [("W target", self.tgt0, self.tgt1, C_TARGET),
                ("Vacuum decay chamber", self.vdc0, self.vdc1, C_VDC),
                ("Magnet + trackers", self.mag0, self.mag1, C_TRACKER),
                ("CsI ECAL", self.calo0, self.calo1, C_ECAL)]


# ── 3D primitives (data axes: X=z_beam, Y=x_lab, Z=y_lab → +z points right) ──

def box_faces(z0, z1, hx, hy):
    v = np.array([[z, sx * hx, sy * hy]
                  for z in (z0, z1) for sx in (-1, 1) for sy in (-1, 1)])
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [v[list(i)] for i in idx]


def add_box(ax, z0, z1, hx, hy, color, alpha, lw=0.6):
    pc = Poly3DCollection(box_faces(z0, z1, hx, hy), facecolors=color,
                          edgecolors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(pc)


def add_tube(ax, z0, z1, r, color, alpha):
    th = np.linspace(0, 2 * np.pi, 40)
    zz = np.array([z0, z1])
    TH, ZZ = np.meshgrid(th, zz)
    ax.plot_surface(ZZ, r * np.cos(TH), r * np.sin(TH), color=color,
                    alpha=alpha, linewidth=0, shade=False)
    for z in (z0, z1):  # rim silhouettes
        ax.plot(np.full_like(th, z), r * np.cos(th), r * np.sin(th),
                color=color, lw=0.8, alpha=min(1.0, alpha * 3))


def track_end(vx, vy, vz, px, py, pz, geom, default_len=250.0):
    """Extend geom-accepted photons to the calo face; others a fixed stub."""
    if pz > 1e-9:
        t = (geom.calo0 - vz) / pz
        if 0 < t < 2000.0:
            return vx + t * px, vy + t * py, geom.calo0
    return vx + default_len * px, vy + default_len * py, vz + default_len * pz


def gamma_color(geom_ok, at_calo, c_geom, c_calo):
    if geom_ok:
        return c_calo if at_calo else c_geom
    return C_MISS


# ── renders ──────────────────────────────────────────────────────────────────

def draw_geometry_3d(ax, geom):
    add_box(ax, geom.tgt0, geom.tgt1, geom.tgt_hxy, geom.tgt_hxy, C_TARGET, 0.55)
    add_tube(ax, geom.vdc0, geom.vdc1, geom.vdc_rin + geom.vdc_wall, C_VDC, 0.14)
    add_box(ax, geom.mag0, geom.mag1, geom.mag_hxy, geom.mag_hxy, "#666666", 0.06)
    for zt in geom.tracker_z():
        add_box(ax, zt - geom.trk_thick / 2, zt + geom.trk_thick / 2,
                geom.trk_hxy, geom.trk_hxy, C_TRACKER, 0.28)
    add_box(ax, geom.calo0, geom.calo0 + 0.5, geom.calo_hxy, geom.calo_hxy,
            C_SCORE, 0.5)
    add_box(ax, geom.calo0, geom.calo1, geom.calo_hxy, geom.calo_hxy, C_ECAL, 0.20)
    # beam axis
    ax.plot([geom.tgt0 - 120, geom.tgt0], [0, 0], [0, 0],
            color=C_BEAM, lw=1.6)
    ax.text(geom.tgt0 - 120, 0, 18, "e$^-$ 8 GeV", color=C_BEAM, fontsize=9)


def checkpoint_labels_3d(ax, geom, y_lab):
    for name, z0, z1, col in geom.spans():
        zm = 0.5 * (z0 + z1)
        ax.text(zm, 0, y_lab, f"{name}\n[{z0:.0f}, {z1:.0f}] mm",
                color=col, fontsize=7.5, ha="center", va="bottom")
        for z in (z0, z1):
            ax.plot([z, z], [0, 0], [y_lab - 18, y_lab - 4],
                    color=col, lw=0.8, ls=":")


def render_3d(df, geom, out_png, max_tracks=1500):
    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    ax.set_axis_off()

    draw_geometry_3d(ax, geom)
    checkpoint_labels_3d(ax, geom, y_lab=205.0)

    segs, cols, lws = [], [], []
    sub = df.head(max_tracks)
    for _, d in sub.iterrows():
        for g in (1, 2):
            ex, ey, ez = track_end(d.vx_mm, d.vy_mm, d.vz_mm,
                                   d[f"px{g}"], d[f"py{g}"], d[f"pz{g}"], geom)
            c = gamma_color(d[f"gamma{g}GeomAccept"], d[f"gamma{g}AtCalo"],
                            C_G1_GEOM if g == 1 else C_G2_GEOM,
                            C_G1_CALO if g == 1 else C_G2_CALO)
            # (X=z, Y=x, Z=y)
            segs.append([(d.vz_mm, d.vx_mm, d.vy_mm), (ez, ex, ey)])
            cols.append(c + ("22" if c == C_MISS else "ff"))
            lws.append(0.4 if c == C_MISS else 1.1)
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=lws))
    ax.scatter(df.vz_mm, df.vx_mm, df.vy_mm, s=5, c="#55bbff",
               depthshade=False, zorder=5)

    zlo, zhi = geom.tgt0 - 140, geom.calo1 + 40
    r = 200.0
    ax.set_xlim(zlo, zhi)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((zhi - zlo, 2 * r, 2 * r), zoom=1.35)
    ax.view_init(elev=18, azim=-83)

    fig.text(0.02, 0.965, r"$\pi^0\!\to\!\gamma\gamma$ decay vertices + photon"
             " directions  (beam $+z\\ \\rightarrow$)", color=FG, fontsize=12)
    fig.text(0.02, 0.045,
             f"{len(df)} decays, first {min(len(sub), len(df))} drawn as tracks."
             "  blue dot = vertex;  red/orange = $\\gamma_1$/$\\gamma_2$ toward"
             " calo aperture (dark = crossed calo plane);  gray = misses calo."
             "  Accepted tracks extended to calo face.",
             color=FG, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, facecolor=BG)
    plt.close(fig)


def render_side(df, geom, out_png, max_tracks=1500):
    fig, ax = plt.subplots(figsize=(14, 5.6), facecolor=BG)
    ax.set_facecolor(BG)

    # geometry cross-sections (x-z plane)
    def rect(z0, z1, h, color, alpha, label=None):
        ax.add_patch(Rectangle((z0, -h), z1 - z0, 2 * h, facecolor=color,
                               edgecolor=color, alpha=alpha, lw=1.0,
                               label=label))

    rect(geom.tgt0, geom.tgt1, geom.tgt_hxy, C_TARGET, 0.75)
    rout = geom.vdc_rin + geom.vdc_wall
    rect(geom.vdc0, geom.vdc1, rout, C_VDC, 0.10)
    for r0, r1 in ((geom.vdc_rin, rout), (-rout, -geom.vdc_rin)):
        ax.fill_between([geom.vdc0, geom.vdc1], r0, r1, color=C_VDC, alpha=0.6)
    for z in (geom.vdc0, geom.vdc1):  # steel end caps
        ax.fill_between([z - 5 if z == geom.vdc1 else z,
                         z if z == geom.vdc1 else z + 5],
                        -geom.vdc_rin, geom.vdc_rin, color=C_VDC, alpha=0.6)
    rect(geom.mag0, geom.mag1, geom.mag_hxy, "#888888", 0.15)
    for zt in geom.tracker_z():
        rect(zt - 1, zt + 1, geom.trk_hxy, C_TRACKER, 0.8)
    ax.plot([geom.calo0, geom.calo0], [-geom.calo_hxy, geom.calo_hxy],
            color=C_SCORE, lw=1.4)
    rect(geom.calo0, geom.calo1, geom.calo_hxy, C_ECAL, 0.30)

    # beam
    ax.annotate("", xy=(geom.tgt0, 0), xytext=(geom.tgt0 - 130, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_BEAM, lw=1.6))
    ax.text(geom.tgt0 - 130, 12, "e$^-$ 8 GeV", color=C_BEAM, fontsize=10)

    # tracks
    sub = df.head(max_tracks)
    for _, d in sub.iterrows():
        for g in (1, 2):
            ex, _, ez = track_end(d.vx_mm, d.vy_mm, d.vz_mm,
                                  d[f"px{g}"], d[f"py{g}"], d[f"pz{g}"], geom)
            c = gamma_color(d[f"gamma{g}GeomAccept"], d[f"gamma{g}AtCalo"],
                            C_G1_GEOM if g == 1 else C_G2_GEOM,
                            C_G1_CALO if g == 1 else C_G2_CALO)
            ax.plot([d.vz_mm, ez], [d.vx_mm, ex], color=c,
                    lw=0.4 if c == C_MISS else 0.9,
                    alpha=0.13 if c == C_MISS else 1.0, zorder=2)
    ax.scatter(df.vz_mm, df.vx_mm, s=4, c="#55bbff", zorder=3,
               label="decay vertex")

    # track-color legend (proxy artists)
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([], [], color="#55bbff", marker="o", ls="none", ms=4,
               label=r"$\pi^0$ decay vertex"),
        Line2D([], [], color=C_G1_GEOM, lw=1.2,
               label=r"$\gamma_1$ toward calo (dark = crossed plane)"),
        Line2D([], [], color=C_G2_GEOM, lw=1.2,
               label=r"$\gamma_2$ toward calo"),
        Line2D([], [], color=C_MISS, lw=1.2, alpha=0.6,
               label=r"$\gamma$ misses calo aperture (3D test:"
                     " may miss in $y$, out of this plane)"),
    ]
    leg = ax.legend(handles=proxies, loc="lower left", fontsize=8,
                    facecolor=BG, edgecolor=FG, framealpha=0.7)
    for t in leg.get_texts():
        t.set_color(FG)

    # checkpoint spans + boundary lines
    y_span = 235
    for name, z0, z1, col in geom.spans():
        for z in (z0, z1):
            ax.axvline(z, color=col, lw=0.7, ls=":", alpha=0.8)
        ax.annotate("", xy=(z1, y_span), xytext=(z0, y_span),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
        ax.text(0.5 * (z0 + z1), y_span + 8,
                f"{name}  ({(z1 - z0) / 10:.0f} cm)", color=col,
                fontsize=9, ha="center")
        ax.text(z0, -y_span - 22, f"{z0:.0f}", color=col, fontsize=7.5,
                ha="center")
    ax.text(geom.calo1, -y_span - 22, f"{geom.calo1:.0f}", color=C_ECAL,
            fontsize=7.5, ha="center")

    ax.set_xlim(geom.tgt0 - 160, geom.calo1 + 40)
    ax.set_ylim(-y_span - 35, y_span + 45)
    ax.set_xlabel("z [mm]  (beam direction →)", color=FG)
    ax.set_ylabel("x [mm]", color=FG)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(FG)
    ax.set_title(r"$\pi^0\!\to\!\gamma\gamma$ decay vertices — side view"
                 " (x–z, lab frame)", color=FG)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, facecolor=BG)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", help="pi0_decays.csv file(s)")
    p.add_argument("--out-dir", default="plots/png/pi0")
    p.add_argument("--target-len-mm", type=float, default=100.0)
    p.add_argument("--target-xy-mm", type=float, default=50.0)
    p.add_argument("--vdc-len-mm", type=float, default=300.0)
    p.add_argument("--calo-xy-mm", type=float, default=120.0)
    p.add_argument("--build-start-mm", type=float, default=-500.0,
                   help="z of target front face (construction.cpp: -50 cm)")
    p.add_argument("--max-tracks", type=int, default=1500)
    args = p.parse_args()

    geom = Geometry(args.target_len_mm, args.target_xy_mm, args.vdc_len_mm,
                    args.calo_xy_mm, args.build_start_mm)
    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.csvs:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[skip] {path}: no decays", file=sys.stderr)
            continue
        prefix = os.path.basename(path).replace("pi0_decays.csv", "").rstrip("_")
        prefix = prefix + "_" if prefix else ""
        p3 = os.path.join(args.out_dir, f"{prefix}decay_vertices_g4view3d.png")
        ps = os.path.join(args.out_dir, f"{prefix}decay_vertices_g4side.png")
        render_3d(df, geom, p3, args.max_tracks)
        render_side(df, geom, ps, args.max_tracks)
        print(f"[ok] {path}: {len(df)} decays -> {p3}, {ps}")


if __name__ == "__main__":
    main()
