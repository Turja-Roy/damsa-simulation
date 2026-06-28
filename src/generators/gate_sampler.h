#ifndef GATE_SAMPLER_H
#define GATE_SAMPLER_H

// Samples how many electrons arrive in one calorimeter readout gate for the
// active LESA beam mode (Level B time structure — see plan.md §3.2-3.3).
//
//   n_bunches_in_gate = min( floor(gate / bunch_spacing), bunches_per_kick )
//   occupancy per bunch ~ Poisson(bunch_charge)   [dark current: Poisson(0.07)]
//                       or = round(bunch_charge)   [fixed, laser modes]
//   n_electrons = sum over bunches
//
// Thread-safe: G4Poisson draws from the thread-local RNG engine, so this is
// safe to call from MT worker threads.

#include <algorithm>
#include <cmath>

#include "G4Poisson.hh"
#include "damsa_config.h"

namespace DamsaConfig {

inline int SampleGateOccupancy(const BeamSpec& s, double gate_s, bool poisson) {
    int nb = static_cast<int>(std::floor(gate_s / s.bunchSpacing_s));
    nb = std::min(nb, s.bunchesPerKick);
    if (nb < 1) nb = 1;   // always at least one bunch in the gate

    long total = 0;
    for (int b = 0; b < nb; ++b) {
        total += poisson ? static_cast<long>(G4Poisson(s.bunchCharge_e))
                         : std::lround(s.bunchCharge_e);
    }
    return static_cast<int>(total);
}

}  // namespace DamsaConfig

#endif  // GATE_SAMPLER_H
