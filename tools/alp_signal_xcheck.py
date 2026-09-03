#!/usr/bin/env python3
"""Cross-check damsa_alp_signal against alp_signal_pipeline.py.

    ./build/damsa_alp_signal --flux F --alplib alplib --auto-coupling \
        --angle-samples N --no-4vec --no-sensitivity --xcheck > cpp.txt
    python3 tools/alp_signal_xcheck.py --flux F --angle-samples N > py.txt

Three quantities are deterministic even though the decays are sampled, and must
agree exactly:
    coupling      pick_safe_coupling is a closed-form function of the flux
    theta_kin     exact kinematic expectation, weighted by the ALP weights
    total_weight  every sample carries weight/n and there are exactly n of them
The sampled quantities (theta_MC, accept) are compared statistically instead --
the C++ and numpy RNG streams differ, so per-event agreement is impossible.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline"))

import numpy as np
import alp_signal_pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument("--flux", required=True)
ap.add_argument("--angle-samples", type=int, default=5)
ap.add_argument("--masses", default="1,5,10,20,50,100,200,500")
ap.add_argument("--vdc-length", type=float, default=None)
a = ap.parse_args()

if a.vdc_length is not None:
    P.VDC_M = a.vdc_length
    P.DET_DIST_M = P.TARGET_HALF_M + P.VDC_M + P.MAGNET_M

flux = P.load_geant4_brems_flux(a.flux, 0)
masses = [float(m) for m in a.masses.split(",")]

print("%-10s %-14s %-16s %-16s %-14s %-12s" %
      ("ma_MeV", "coupling", "theta_kin_mrad", "total_weight", "theta_MC_mrad", "accept"))
for ma in masses:
    g = P.pick_safe_coupling(ma, flux, target_decay_length_m=5.0)
    fo, gen = P.run_alplib(flux, ma, g, n_samples=500)

    Ea = np.asarray(fo.axion_energy)
    w = np.asarray(fo.decay_axion_weight, dtype=np.float64)
    tk = (1000.0 * np.average([P.expected_mean_theta_per_alp(e, ma) for e in Ea],
                              weights=w)) if w.sum() > 0 else 0.0
    total = float(P.EXPOSURE_DAYS * 86400.0 * w.sum())

    p41, p42, wg = gen.simulate_decay_4vectors(days_exposure=P.EXPOSURE_DAYS,
                                               n_samples=a.angle_samples)
    wg = np.asarray(wg)
    th = np.array([P.compute_opening_angles.__globals__['np'].arccos(
        np.clip(np.dot([p1.p1, p1.p2, p1.p3], [p2.p1, p2.p2, p2.p3]) /
                (np.linalg.norm([p1.p1, p1.p2, p1.p3]) *
                 np.linalg.norm([p2.p1, p2.p2, p2.p3])), -1.0, 1.0))
        for p1, p2 in zip(p41, p42)])
    vz = P.sample_decay_vertex_z(p41, p42, ma, g)
    acc = P.transverse_acceptance_mask(p41, p42, det_dist_m=P.DET_DIST_M, vertex_z_m=vz)

    mean = 1000.0 * np.average(th, weights=wg) if wg.sum() > 0 else 0.0
    frac = float(wg[acc].sum() / wg.sum()) if wg.sum() > 0 else 0.0
    print("%-10.1f %-14.8e %-16.10f %-16.10e %-14.4f %-12.6e" %
          (ma, g, tk, total, mean, frac))
