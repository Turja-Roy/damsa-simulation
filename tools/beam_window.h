#ifndef DAMSA_BEAM_WINDOW_H
#define DAMSA_BEAM_WINDOW_H

// LESA beam time structure for the offline overlay tools (damsa_pileup,
// damsa_calo_reco): one coincidence window = the bunches inside it, each with a
// Poisson number of library electrons. Random-2-photons plan §3.2.

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>

namespace {

struct BeamSpec { double bunchCharge, kickerRate; int bunchesPerKick; double spacing_s; };

// Matches DamsaConfig::BeamSpecFor in src/config/damsa_config.h.
const std::map<std::string, BeamSpec> kBeamModes = {
    {"dark",        {0.07,     929e3, 100, 5.4e-9 }},
    {"lesa",        {4200.0,   929e3, 18,  26.9e-9}},
    {"xleap",       {167000.0, 929e3, 1,   1.08e-6}},
    {"interleaved", {6.2e8,    100.0, 1,   10e-3  }},
};

// Bunches inside one coincidence window of width W. The window is not
// phase-locked to the beam: the triggering bunch sits at a uniform position u
// inside it, so the count is 1 + floor(u/s) + floor((W-u)/s), capped at the
// bunches in one kick. Below one spacing this is exactly 1 -- all remaining
// accidentals are same-bunch (plan §3.2).
template <class RNG>
int SampleBunchesInWindow(const BeamSpec& s, double window_s, RNG& rng)
{
    std::uniform_real_distribution<double> uni(0.0, window_s);
    const double u = uni(rng);
    const int nb = 1 + static_cast<int>(std::floor(u / s.spacing_s))
                     + static_cast<int>(std::floor((window_s - u) / s.spacing_s));
    return std::min(nb, s.bunchesPerKick);
}

// A sum of n iid Poisson(q) is Poisson(n*q), so one draw per window suffices.
template <class RNG>
long long SampleOccupancy(int nBunches, double bunchCharge, bool poisson, RNG& rng)
{
    const double mean = nBunches * bunchCharge;
    if (!poisson) return static_cast<long long>(std::llround(mean));
    std::poisson_distribution<long long> pd(mean);
    return pd(rng);
}

}  // namespace

#endif
