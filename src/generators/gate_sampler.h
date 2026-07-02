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
#include <vector>

#include "G4Poisson.hh"
#include "damsa_config.h"

namespace DamsaConfig {

// Per-bunch occupancies for one gate. Index b = bunch number; the electron
// arrival time within the gate is b * bunchSpacing_s (used by the generator to
// give each vertex its physical time offset).
inline std::vector<int> SampleGateBunchOccupancies(const BeamSpec& s, double gate_s,
                                                   bool poisson) {
    int nb = static_cast<int>(std::floor(gate_s / s.bunchSpacing_s));
    nb = std::min(nb, s.bunchesPerKick);
    if (nb < 1) nb = 1;   // always at least one bunch in the gate

    std::vector<int> occ(nb, 0);
    for (int b = 0; b < nb; ++b) {
        occ[b] = poisson ? static_cast<int>(G4Poisson(s.bunchCharge_e))
                         : static_cast<int>(std::lround(s.bunchCharge_e));
    }
    return occ;
}

inline int SampleGateOccupancy(const BeamSpec& s, double gate_s, bool poisson) {
    const auto occ = SampleGateBunchOccupancies(s, gate_s, poisson);
    long total = 0;
    for (int n : occ) total += n;
    return static_cast<int>(total);
}

}  // namespace DamsaConfig

#endif  // GATE_SAMPLER_H
