#ifndef DAMSA_ALP_CONSTANTS_H
#define DAMSA_ALP_CONSTANTS_H

// Constants transcribed from alplib/constants.py.
//
// alplib is NOT modified by this port; it stays on disk as the reference the
// cross-check (tools/alp_xcheck.cpp) diffs against. Values are reproduced
// LITERALLY, including where alplib uses rounded inputs — "correcting" them
// here would silently change published numbers.
//
// alplib base units: MeV, cm, s.

#include <cmath>

namespace damsa::alp {

// alplib writes ALPHA = 1/137, not the CODATA 7.2973525693e-3.
inline constexpr double kAlpha = 1.0 / 137.0;

// constants.py: METER_BY_MEV = 6.58212e-22 * 2.998e8   [MeV*m]
// Note 2.998e8, not the exact c = 2.99792458e8.
inline constexpr double kMeterByMeV = 6.58212e-22 * 2.998e8;

// MEV2_CM2 = (METER_BY_MEV * 100)**2
inline constexpr double kMeV2Cm2 = (kMeterByMeV * 100.0) * (kMeterByMeV * 100.0);

inline constexpr double kSecPerDay     = 3600.0 * 24.0;
inline constexpr double kChargeCoulombs = 1.602176634e-19;
inline constexpr double kCLight        = 2.99792458e10;  // cm/s
inline constexpr double kHbar          = 6.58212e-22;    // MeV*s

// Tungsten, from alplib/data/mat_params.json ("W").
inline constexpr double kTungstenZ       = 74.0;
inline constexpr double kTungstenN       = 108.0;
inline constexpr double kTungstenDensity = 19.3;   // g/cm^3

}  // namespace damsa::alp

#endif
