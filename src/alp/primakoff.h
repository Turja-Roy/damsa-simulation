#ifndef DAMSA_ALP_PRIMAKOFF_H
#define DAMSA_ALP_PRIMAKOFF_H

// Primakoff production cross section and the a -> gamma gamma width.
// Ports alplib/prod_xs.py:99 (primakoff_sigma) and alplib/decay.py:10 (W_gg).

#include "constants.h"

#include <cmath>

namespace damsa::alp {

// decay.py:10 — W_gg(g_agamma, ma) = g^2 ma^3 / (64 pi), g in MeV^-1.
inline double WGammaGamma(double g_agamma, double ma)
{
    // Association matches Python's `g**2 * ma**3 / (64*pi)`; writing it as
    // g*g*ma*ma*ma reassociates and lands 1 ULP away.
    return std::pow(g_agamma, 2) * std::pow(ma, 3) / (64.0 * M_PI);
}

// prod_xs.py:99 default screening parameter r0 = 2.2e-10 / METER_BY_MEV.
// NB: FluxPrimakoffIsotropic calls primakoff_sigma without an r0 argument, so
// this default is used and the material's own atomic_radius (2.1 A for W in
// mat_params.json) is NOT used. Reproduced as-is.
inline constexpr double kPrimakoffR0 = 2.2e-10 / kMeterByMeV;

// Inverse-Primakoff total cross section (Creswick et al), screened.
//   prefactor = (g z)^2 / (2*137)
//   eta2      = r0^2 eg^2
//   sigma     = theta(eg - ma) * prefactor * ((2 eta2 + 1)/(4 eta2) * ln(1 + 4 eta2) - 1)
//
// numpy's heaviside(x, 0.0) is 0 at x == 0, so eg == ma gives exactly zero.
inline double PrimakoffSigma(double eg, double g, double ma, double z,
                             double r0 = kPrimakoffR0)
{
    if (!(eg > ma)) return 0.0;          // covers eg < ma and eg == ma
    const double gz     = g * z;
    const double prefac = gz * gz / (2.0 * 137.0);
    const double eta2   = r0 * r0 * eg * eg;
    if (eta2 <= 0.0) return 0.0;
    return prefac * (((2.0 * eta2 + 1.0) / (4.0 * eta2)) * std::log(1.0 + 4.0 * eta2) - 1.0);
}

}  // namespace damsa::alp

#endif
