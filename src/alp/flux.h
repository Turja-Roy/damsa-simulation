#ifndef DAMSA_ALP_FLUX_H
#define DAMSA_ALP_FLUX_H

// Primakoff ALP production and propagation.
// Ports alplib/fluxes.py:111 (FluxPrimakoffIsotropic) and its base-class
// propagate() at fluxes.py:42.

#include "abs_xs.h"
#include "constants.h"
#include "primakoff.h"

#include <cmath>
#include <limits>
#include <vector>

namespace damsa::alp {

struct FluxPrimakoffIsotropic {
    // Geometry in metres, energies in MeV, coupling in MeV^-1.
    double ma          = 0.1;
    double gagamma     = 1e-3;
    double targetZ     = kTungstenZ;
    double detDist_m   = 4.0;
    double detLength_m = 0.2;
    double detArea_m2  = 0.04;

    // Filled by Simulate(), consumed by Propagate().
    std::vector<double> axionEnergy;
    std::vector<double> axionFlux;          // N_ALP produced per second
    std::vector<double> decayAxionWeight;
    std::vector<double> scatterAxionWeight;

    // fluxes.py:135 simulate_single / :146 simulate.
    // photonEnergy/photonRate: the brems spectrum, rate in photons/s.
    // Rows with E < ma are dropped entirely (not zero-weighted), so the output
    // arrays are shorter than the input — matching alplib.
    void Simulate(const std::vector<double>& photonEnergy,
                  const std::vector<double>& photonRate,
                  const AbsCrossSection& targetPhotonXs)
    {
        axionEnergy.clear();
        axionFlux.clear();
        decayAxionWeight.clear();
        scatterAxionWeight.clear();

        for (std::size_t i = 0; i < photonEnergy.size(); ++i) {
            const double eg = photonEnergy[i];
            if (eg < ma) continue;                       // fluxes.py:137
            const double xs = PrimakoffSigma(eg, gagamma, ma, targetZ);
            const double br = xs / targetPhotonXs.SigmaMeV(eg);
            axionEnergy.push_back(eg);
            axionFlux.push_back(photonRate[i] * br);
        }
    }

    // fluxes.py:42 propagate(decay_width, rescale_factor).
    //
    // isIsotropic mirrors alplib's flag, but alp_signal_pipeline.py calls
    // propagate(is_isotropic=False): the det_area/(4 pi d^2) factor models an
    // isotropic source, whereas DAMSA ALPs inherit the forward collimation of
    // 8 GeV bremsstrahlung, and it would double-count the transverse acceptance
    // applied downstream. Default false to match the pipeline.
    void Propagate(bool isIsotropic = false, double newCoupling = -1.0)
    {
        double rescale = 1.0;
        double width   = WGammaGamma(gagamma, ma);
        if (newCoupling > 0.0) {
            const double r = newCoupling / gagamma;
            rescale = r * r;
            width   = WGammaGamma(newCoupling, ma);
        }

        const std::size_t n = axionEnergy.size();
        decayAxionWeight.assign(n, 0.0);
        scatterAxionWeight.assign(n, 0.0);

        for (std::size_t i = 0; i < n; ++i) {
            const double ea = axionEnergy[i];
            const double pa = std::sqrt(ea * ea - ma * ma);
            const double va = pa / ea;
            const double boost = ea / ma;
            const double tau = (width > 0.0) ? boost / width
                                             : std::numeric_limits<double>::infinity();

            const double survProb  = std::exp(-detDist_m / kMeterByMeV / va / tau);
            const double decayProb = 1.0 - std::exp(-detLength_m / kMeterByMeV / va / tau);

            decayAxionWeight[i]   = rescale * axionFlux[i] * survProb * decayProb;
            scatterAxionWeight[i] = rescale * axionFlux[i] * survProb;
        }

        if (isIsotropic) {
            const double geom = detArea_m2 / (4.0 * M_PI * detDist_m * detDist_m);
            for (auto& w : decayAxionWeight)   w *= geom;
            for (auto& w : scatterAxionWeight) w *= geom;
        }
    }

    // generators.py:88 PhotonEventGenerator.decays().
    // heaviside(E - threshold, 1.0) keeps the row when E == threshold.
    double Decays(double daysExposure, double threshold) const
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < axionEnergy.size(); ++i)
            if (axionEnergy[i] >= threshold)
                sum += daysExposure * kSecPerDay * decayAxionWeight[i];
        return sum;
    }
};

}  // namespace damsa::alp

#endif
