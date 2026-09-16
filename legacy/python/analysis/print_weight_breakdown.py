#!/usr/bin/env python3
"""
Per-event weight breakdown diagnostic.

Answers the postdoc's request: print (E_ALP, E_γ1, E_γ2, weight) for a few
events, AND decompose the weight into its physical factors so we can see WHICH
factor drives the signal up or down.

    weight = N_ALP_produced × P_survival × P_decay_in_detector × (exposure)

mapping to alplib (alplib/generators.py:394, FluxPrimakoffIsotropic.propagate):

    decay_axion_weight = g**2 * wgt * surv_prob * decay_prob
                         └─────┬────┘  └──┬───┘   └───┬────┘
                         N_ALP_produced  P_surv   P_decay
      g**2 * wgt = (photon flux) × (Primakoff σ) × (coupling²)
      surv_prob  = exp(-det_dist   / (γβ c τ))
      decay_prob = 1 - exp(-det_length / (γβ c τ))

Final per-event event count = days_exposure × S_PER_DAY × decay_axion_weight,
restricted to E_ALP > threshold.

Beam normalization: DAMSA paper (fermilab-pub-26-0039-ppd) assumes
10^4 electrons/pulse × 1 kHz = 1e7 e/s. The Geant4 flux file was produced at
62.5 µA (~3.9e14 e/s), so we rescale the flux by ELECTRONS_PER_S_PAPER / file norm.

Usage:
    python scripts/analysis/print_weight_breakdown.py \
        --flux output/alplib_brems_flux.csv --ma 100 --g 1e-4 --n 10
"""
import sys
import argparse
import numpy as np
from pathlib import Path

# Reuse the production pipeline (constants, flux loader, alplib runner).
PIPE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(PIPE_DIR))
import alp_signal_pipeline as pipe  # noqa: E402
from alplib.constants import METER_BY_MEV, S_PER_DAY  # noqa: E402

# Beam spec from the DAMSA paper: 1e4 e/pulse × 1 kHz.
ELECTRONS_PER_S_PAPER = 1.0e4 * 1.0e3   # = 1e7 e/s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flux", default="output/alplib_brems_flux.csv",
                    help="Geant4 brems flux CSV (photons/s, file beam norm)")
    ap.add_argument("--ma", type=float, default=100.0, help="ALP mass [MeV]")
    ap.add_argument("--g", type=float, default=1.0e-4,
                    help="ALP-photon coupling [GeV^-1]")
    ap.add_argument("--n", type=int, default=10, help="events to print")
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="alplib MC samples per flux bin")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="calo energy threshold on E_ALP [MeV]")
    ap.add_argument("--paper-beam", action="store_true", default=True,
                    help="rescale flux to paper beam (1e7 e/s) [default on]")
    ap.add_argument("--file-beam", dest="paper_beam", action="store_false",
                    help="keep the flux file's own beam norm (62.5 µA)")
    args = ap.parse_args()

    flux = pipe.load_geant4_brems_flux(args.flux, n_primaries=None)

    # Rescale flux to the paper's beam intensity. Flux file ∝ ELECTRONS_PER_S
    # (62.5 µA). Weight is linear in flux, so a scalar rescale is exact.
    beam_scale = 1.0
    if args.paper_beam:
        beam_scale = ELECTRONS_PER_S_PAPER / pipe.ELECTRONS_PER_S
        flux = flux.copy()
        flux[:, 1] *= beam_scale
        print(f"[beam] rescaled file ({pipe.ELECTRONS_PER_S:.2e} e/s) "
              f"→ paper ({ELECTRONS_PER_S_PAPER:.2e} e/s), factor {beam_scale:.2e}")
    else:
        print(f"[beam] using file norm ({pipe.ELECTRONS_PER_S:.2e} e/s)")

    flux_obj, gen = pipe.run_alplib(flux, args.ma, args.g, n_samples=args.n_samples)

    Ea = np.asarray(flux_obj.axion_energy)
    if len(Ea) == 0:
        sys.exit(f"No ALP events (ma={args.ma} > max photon energy?)")

    g_MeV = args.g / 1000.0                       # alplib coupling units
    wgt   = np.asarray(flux_obj.axion_flux)       # flux × Primakoff σ (pre-coupling)
    ma    = args.ma

    # Recompute the three physical factors per event (mirror of propagate()).
    p_a   = np.sqrt(Ea**2 - ma**2)
    v_a   = p_a / Ea
    boost = Ea / ma
    tau   = 64.0 * np.pi / (g_MeV**2 * ma**3) * boost          # boosted lifetime [MeV^-1]
    decay_length_m = METER_BY_MEV * v_a * tau                  # γβ c τ in metres

    n_alp     = g_MeV**2 * wgt                                  # ALPs produced / s
    surv_prob = np.exp(-pipe.DET_DIST_M / decay_length_m)
    decay_prob= 1.0 - np.exp(-pipe.DET_LENGTH_M / decay_length_m)
    w_per_s   = n_alp * surv_prob * decay_prob                  # = decay_axion_weight
    w_exposure= w_per_s * pipe.EXPOSURE_DAYS * S_PER_DAY        # events over exposure

    # Decay photons (for E_γ1, E_γ2) at the chosen exposure.
    p41, p42, w_evt = gen.simulate_decay_4vectors(
        days_exposure=pipe.EXPOSURE_DAYS, n_samples=1)

    print()
    print(f"ma={ma:.0f} MeV  g={args.g:.2e} GeV^-1  "
          f"det_dist={pipe.DET_DIST_M:.3f} m  det_length={pipe.DET_LENGTH_M:.3f} m  "
          f"exposure={pipe.EXPOSURE_DAYS:.0f} d")
    print(f"median boosted decay length = {np.median(decay_length_m):.3e} m "
          f"(detector at {pipe.DET_DIST_M:.2f} m)")
    print()
    hdr = (f"{'E_ALP':>9} {'E_g1':>9} {'E_g2':>9} | "
           f"{'N_ALP/s':>11} {'P_surv':>9} {'P_decay':>9} | "
           f"{'w/s':>11} {'w_exposure':>11}")
    print(hdr)
    print("-" * len(hdr))
    n_print = min(args.n, len(p41))
    for i in range(n_print):
        e_g1, e_g2 = p41[i].energy(), p42[i].energy()
        e_alp = e_g1 + e_g2
        # match the printed pair back to the closest axion-energy index for factors
        j = int(np.argmin(np.abs(Ea - e_alp)))
        print(f"{e_alp:9.1f} {e_g1:9.1f} {e_g2:9.1f} | "
              f"{n_alp[j]:11.3e} {surv_prob[j]:9.3e} {decay_prob[j]:9.3e} | "
              f"{w_per_s[j]:11.3e} {w_exposure[j]:11.3e}")

    # Totals + threshold cut.
    mask = Ea > args.threshold
    total = w_exposure[mask].sum()
    print()
    print(f"Σ weight (E_ALP>{args.threshold:.0f} MeV, {pipe.EXPOSURE_DAYS:.0f} d) "
          f"= {total:.3e} events")
    print(f"  ⟨N_ALP/s⟩={np.mean(n_alp):.3e}  "
          f"⟨P_surv⟩={np.mean(surv_prob):.3e}  "
          f"⟨P_decay⟩={np.mean(decay_prob):.3e}")
    # Flag the dominant suppressor.
    factors = {"P_surv": np.mean(surv_prob), "P_decay": np.mean(decay_prob)}
    killer = min(factors, key=factors.get)
    print(f"  → smallest geometric factor: {killer} = {factors[killer]:.3e}")


if __name__ == "__main__":
    main()
