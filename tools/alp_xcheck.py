#!/usr/bin/env python3
"""Emit the same grid as tools/alp_xcheck.cpp, computed by the real alplib.

    ./build/alp_xcheck > cpp.txt
    python3 tools/alp_xcheck.py > py.txt
    diff cpp.txt py.txt

alplib is imported unmodified; this only reads from it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from alplib.photon_xs import AbsCrossSection
from alplib.prod_xs import primakoff_sigma
from alplib.decay import W_gg
from alplib.fluxes import FluxPrimakoffIsotropic
from alplib.materials import Material
from alplib.constants import METER_BY_MEV, S_PER_DAY
from alplib.cross_section_mc import LorentzVector, Vector3, lorentz_boost, Decay2Body

W = Material("W")
abs_xs = AbsCrossSection(W)

ENERGIES = [1.0e-3, 1.5e-3, 0.01, 0.1, 0.511, 1.0, 1.5, 2.5, 10.0, 100.0,
            1000.0, 4000.0, 7999.5, 8000.0, 1.0e4, 1.0e5]
MASSES = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
COUPLINGS = [1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

g17 = lambda x: np.format_float_positional(
    float(x), precision=17, unique=True, fractional=False, trim='-')


def p(*a):
    print(*a)


p("# abs_xs_rows %d" % len(abs_xs.pe_data))

for e in ENERGIES:
    p("abs_xs_mev", g17(e), g17(abs_xs.sigma_mev(e)))
for e in ENERGIES:
    p("abs_xs_cm2", g17(e), g17(abs_xs.sigma_cm2(e)))

for g in COUPLINGS:
    for m in MASSES:
        p("w_gg", g17(g), g17(m), g17(W_gg(g, m)))

R0 = 2.2e-10 / METER_BY_MEV
p("r0", g17(R0))
for g in COUPLINGS:
    for m in MASSES:
        for e in ENERGIES:
            p("primakoff", g17(g), g17(m), g17(e),
              g17(primakoff_sigma(e, g, m, W.z[0])))

pe = np.array([i * 200.0 - 100.0 for i in range(1, 41)])
pr = np.array([1e12 / (i * i) for i in range(1, 41)])
photon_flux = np.stack([pe, pr], axis=1)

for m in MASSES:
    for g in COUPLINGS:
        f = FluxPrimakoffIsotropic(
            photon_flux=photon_flux, target=W,
            det_dist=0.35, det_length=0.42, det_area=0.04,
            axion_mass=m, axion_coupling=g)
        f.simulate()
        f.propagate(is_isotropic=False)
        n = len(f.axion_energy)
        sf = np.sum(f.axion_flux)
        sd = np.sum(f.decay_axion_weight)
        ss = np.sum(f.scatter_axion_weight)
        d30 = np.sum(30.0 * S_PER_DAY * f.decay_axion_weight
                     * np.heaviside(np.array(f.axion_energy) - 0.0, 1.0))
        p("flux", g17(g), g17(m), "n=%d" % n,
          "sumflux=" + g17(sf), "sumdecay=" + g17(sd),
          "sumscat=" + g17(ss), "decays30=" + g17(d30))

        # alplib already stores these as float32 (fluxes.py:64); the C++ side
        # casts its doubles down to match. This is the exact comparison.
        g9 = lambda x: np.format_float_positional(
            float(x), precision=9, unique=True, fractional=False, trim='-')
        for i in range(n):
            p("w32", g17(g), g17(m), str(i),
              g9(f.decay_axion_weight[i]), g9(f.scatter_axion_weight[i]))


# ── Boost and two-body decay, deterministic ─────────────────────────────────
BOOST_TEST = [(100.0, 0.0, 0.0, 99.9),
              (500.0, 10.0, -20.0, 480.0),
              (8000.0, 0.0, 0.0, 7999.0),
              (50.0, -5.0, 3.0, -49.0)]
VELS = [(0.0, 0.0, 0.5), (0.1, -0.2, 0.3), (0.0, 0.0, 0.0), (-0.6, 0.0, 0.0)]

for p4 in BOOST_TEST:
    for v in VELS:
        o = lorentz_boost(LorentzVector(*p4), Vector3(*v))
        p("boost", g17(p4[0]), g17(p4[1]), g17(p4[2]), g17(p4[3]),
          g17(v[0]), g17(v[1]), g17(v[2]),
          g17(o.p0), g17(o.p1), g17(o.p2), g17(o.p3))

PHIS = [0.0, 0.7, 2.0, 4.5, 6.0]
THETAS = [0.1, 0.9, 1.5708, 2.4, 3.0]


def decay_at(parent, mp, phi, theta):
    """cross_section_mc.py:523 Decay2Body.decay(), with the sphere draw fixed."""
    p_cm = mp / 2.0
    e_cm = p_cm
    p1 = LorentzVector(e_cm, p_cm * np.cos(phi) * np.sin(theta),
                       p_cm * np.sin(phi) * np.sin(theta), p_cm * np.cos(theta))
    p2 = LorentzVector(e_cm, -p_cm * np.cos(phi) * np.sin(theta),
                       -p_cm * np.sin(phi) * np.sin(theta), -p_cm * np.cos(theta))
    v_in = Vector3(-parent.p1 / parent.p0, -parent.p2 / parent.p0, -parent.p3 / parent.p0)
    return lorentz_boost(p1, v_in), lorentz_boost(p2, v_in)


for m in [1.0, 10.0, 100.0, 500.0]:
    for ea in [600.0, 2000.0, 8000.0]:
        if ea <= m:
            continue
        pa = np.sqrt(ea**2 - m**2)
        alp = LorentzVector(ea, 0.0, 0.0, pa)
        for ph in PHIS:
            for th in THETAS:
                g1, g2 = decay_at(alp, m, ph, th)
                p("decay", g17(m), g17(ea), g17(ph), g17(th),
                  g17(g1.p0), g17(g1.p1), g17(g1.p2), g17(g1.p3),
                  g17(g2.p0), g17(g2.p1), g17(g2.p2), g17(g2.p3))
