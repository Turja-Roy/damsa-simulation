#ifndef DAMSA_ALP_PIPELINE_H
#define DAMSA_ALP_PIPELINE_H

// The DAMSA ALP signal pipeline: brems flux -> Primakoff production ->
// a -> gamma gamma decay -> 4-vector export, opening angles, sensitivity.
//
// Ports scripts/pipeline/alp_signal_pipeline.py. The alplib physics it calls
// lives in the sibling headers here (see src/alp/flux.h etc.), all of which are
// cross-checked against alplib by tools/alp_xcheck.
//
// Geometry note carried over from the Python: alplib's propagate() applies
// survival over det_dist then decay within det_length, so for DAMSA
//   det_dist   = target centre -> target exit   (decays inside W are absorbed)
//   det_length = target exit  -> calo face      (the fiducial decay window)

#include "constants.h"
#include "decay2body.h"
#include "flux.h"
#include "primakoff.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace damsa::alp {

struct Geometry {
    double beamEnergy_MeV = 8000.0;
    double targetLength_m = 0.10;    // 10 cm tungsten dump
    double vdc_m          = 0.30;    // vacuum decay chamber (primary scan parameter)
    double magnet_m       = 0.12;    // magnet + tracker, fixed hardware
    double detHalfX_m     = 0.06;    // calorimeter face half-side
    double detHalfY_m     = 0.06;
    double exposureDays   = 30.0;

    double targetHalf_m() const { return targetLength_m / 2.0; }
    // Target centre -> calo face; the transverse projection plane.
    double detDist_m()    const { return targetHalf_m() + vdc_m + magnet_m; }
    double decayZMin_m()  const { return targetHalf_m(); }
    double decayZMax_m()  const { return targetHalf_m() + vdc_m + magnet_m; }
    double detArea_m2()   const { return 4.0 * detHalfX_m * detHalfY_m; }
};

// alp_signal_pipeline.py:182. Choose g so the mean boosted decay length is
// ~targetDecayLength_m, keeping surv_prob near 1 so heavy ALPs still reach the
// detector (otherwise the exported weights are all zero).
//   g^2 = 64 pi hbar c Ea / (L ma^4)   [MeV^-2],  returned in GeV^-1
inline double PickSafeCoupling(double ma_MeV,
                               const std::vector<double>& energy,
                               const std::vector<double>& rate,
                               double targetDecayLength_m = 5.0)
{
    double sw = 0.0, swx = 0.0;
    for (std::size_t i = 0; i < energy.size(); ++i)
        if (energy[i] > ma_MeV) { sw += rate[i]; swx += rate[i] * energy[i]; }
    if (sw == 0.0) return 1e-4;

    const double eaTyp = swx / sw;
    const double g2 = 64.0 * M_PI * kMeterByMeV * eaTyp
                    / (targetDecayLength_m * std::pow(ma_MeV, 4));
    return std::sqrt(g2) * 1000.0;      // MeV^-1 -> GeV^-1
}

// alp_signal_pipeline.py:205. Exact <theta_open> for one ALP, averaged
// uniformly over rest-frame cos(theta*) in [-1, 1]:
//   cos theta_open(u) = 1 - 2 ma^2 / (Ea^2 - pa^2 u^2)
// Trapezoidal over a uniform grid, as np.trapezoid does.
inline double ExpectedMeanThetaPerAlp(double Ea, double ma, int n = 200)
{
    if (Ea <= ma) return 0.0;
    const double pa2 = Ea * Ea - ma * ma;

    // np.linspace(-1, 1, n): start + i*step, with the last point set exactly
    // to stop. Reproducing the spacing matters -- acos is steep where the
    // denominator approaches ma^2, so a 1-ULP shift in u is visible.
    const double step = 2.0 / double(n - 1);
    std::vector<double> u(n), theta(n);
    for (int i = 0; i < n; ++i) u[i] = -1.0 + i * step;
    u[n - 1] = 1.0;

    for (int i = 0; i < n; ++i) {
        double denom = Ea * Ea - pa2 * u[i] * u[i];
        denom = std::max(denom, ma * ma);
        double c = 1.0 - 2.0 * ma * ma / denom;
        c = std::clamp(c, -1.0, 1.0);
        theta[i] = std::acos(c);
    }

    // np.trapezoid uses the actual per-interval widths, which are not all
    // bit-identical after the linspace rounding above.
    double integral = 0.0;
    for (int i = 0; i + 1 < n; ++i)
        integral += (u[i + 1] - u[i]) * 0.5 * (theta[i] + theta[i + 1]);
    return integral / 2.0;      // divide by the u-range
}

// alp_signal_pipeline.py:271. Decay-vertex z, sampled from the exponential
// decay law truncated to [z0, z1]. The alplib weight already carries
// P(decay in window); this is the CONDITIONAL vertex distribution given that
// decay, so importance weighting stays valid and weights are unchanged.
inline double SampleDecayVertexZ(double Ea, double ma, double coupling_GeV,
                                 double z0, double z1, double u)
{
    const double gMeV = coupling_GeV / 1000.0;
    const double width = std::pow(gMeV, 2) * std::pow(ma, 3) / (64.0 * M_PI);
    const double pa = std::sqrt(std::max(Ea * Ea - ma * ma, 1e-30));
    double L = (pa / ma) * (kMeterByMeV / width);
    L = std::max(L, 1e-12);

    const double a = std::exp(-z0 / L);
    const double b = std::exp(-z1 / L);
    return -L * std::log(a - u * (a - b));
}

// alp_signal_pipeline.py:334. Straight-line propagate both photons from
// (0, 0, vertexZ) to the calo face; require both inside the aperture.
// Photons are neutral, so ignoring the magnet is exact for them.
inline bool TransverseAccept(const FourVector& p1, const FourVector& p2,
                             double detDist_m, double halfX, double halfY,
                             double vertexZ_m)
{
    if (p1.pz <= 0.0 || p2.pz <= 0.0) return false;
    const double dz = detDist_m - vertexZ_m;
    if (dz <= 0.0) return false;

    const double x1 = (p1.px / p1.pz) * dz, y1 = (p1.py / p1.pz) * dz;
    const double x2 = (p2.px / p2.pz) * dz, y2 = (p2.py / p2.pz) * dz;
    return std::abs(x1) <= halfX && std::abs(y1) <= halfY
        && std::abs(x2) <= halfX && std::abs(y2) <= halfY;
}

inline double OpeningAngle(const FourVector& p1, const FourVector& p2)
{
    const double m1 = std::sqrt(p1.px * p1.px + p1.py * p1.py + p1.pz * p1.pz);
    const double m2 = std::sqrt(p2.px * p2.px + p2.py * p2.py + p2.pz * p2.pz);
    if (m1 < 1e-12 || m2 < 1e-12) return 0.0;
    const double c = (p1.px * p2.px + p1.py * p2.py + p1.pz * p2.pz) / (m1 * m2);
    return std::acos(std::clamp(c, -1.0, 1.0));
}

// One exported gamma-gamma pair.
struct DecayPair {
    FourVector g1, g2;
    double weight   = 0.0;   // events per exposure window
    double vertexZ_m = 0.0;
    double Ea       = 0.0;
};

// generators.py:94 simulate_decay_4vectors, with the vertex sampling from
// alp_signal_pipeline.py folded in.
//
// Row order matches the Python: flux index major, decay sample minor.
// nDecaySamples multiplies the row count; note alplib's own `n_samples`
// constructor argument is stored but never used by FluxPrimakoffIsotropic,
// so the flux always has exactly one entry per input spectrum bin.
template <class RNG>
inline std::vector<DecayPair> SimulateDecay4Vectors(
    const FluxPrimakoffIsotropic& flux, const Geometry& geo,
    double coupling_GeV, int nDecaySamples, RNG& rng)
{
    std::vector<DecayPair> out;
    out.reserve(flux.axionEnergy.size() * nDecaySamples);

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double perSample = geo.exposureDays * kSecPerDay / double(nDecaySamples);

    for (std::size_t i = 0; i < flux.axionEnergy.size(); ++i) {
        const double ea = flux.axionEnergy[i];
        const double pa = std::sqrt(std::max(ea * ea - flux.ma * flux.ma, 0.0));
        // theta_ALP = 0: alplib assumes forward flux when no ALP angles were
        // simulated, which is right for collimated bremsstrahlung.
        const FourVector alp{ea, 0.0, 0.0, pa};
        const double w = perSample * flux.decayAxionWeight[i];

        for (int s = 0; s < nDecaySamples; ++s) {
            DecayPair d;
            Decay2BodyMassless(alp, flux.ma, rng, d.g1, d.g2);
            d.weight = w;
            d.Ea = ea;
            d.vertexZ_m = SampleDecayVertexZ(ea, flux.ma, coupling_GeV,
                                             geo.decayZMin_m(), geo.decayZMax_m(),
                                             uni(rng));
            out.push_back(d);
        }
    }
    return out;
}

// Build a configured flux object for one (mass, coupling).
inline FluxPrimakoffIsotropic MakeFlux(const Geometry& geo, double ma_MeV,
                                       double coupling_GeV)
{
    FluxPrimakoffIsotropic f;
    f.ma          = ma_MeV;
    f.gagamma     = coupling_GeV / 1000.0;         // GeV^-1 -> MeV^-1
    f.targetZ     = kTungstenZ;
    f.detDist_m   = geo.decayZMin_m();
    f.detLength_m = geo.decayZMax_m() - geo.decayZMin_m();
    f.detArea_m2  = geo.detArea_m2();
    return f;
}

struct AngleResult {
    double coupling_GeV  = 0.0;
    double meanMrad      = 0.0;   // all decays
    double meanInMrad    = 0.0;   // both photons on the calo face
    double thetaKinMrad  = 0.0;   // exact kinematic expectation
    double acceptFrac    = 0.0;
    std::size_t nPairs   = 0;
};

// alp_signal_pipeline.py:384 compute_opening_angles.
inline AngleResult ComputeOpeningAngles(const std::vector<DecayPair>& pairs,
                                        const FluxPrimakoffIsotropic& flux,
                                        const Geometry& geo, double coupling_GeV)
{
    AngleResult r;
    r.coupling_GeV = coupling_GeV;
    r.nPairs = pairs.size();

    // Kinematic expectation over the same ensemble, weighted by ALP weight.
    double wk = 0.0, swk = 0.0;
    for (std::size_t i = 0; i < flux.axionEnergy.size(); ++i) {
        const double w = flux.decayAxionWeight[i];
        if (w <= 0.0) continue;
        wk  += w;
        swk += w * ExpectedMeanThetaPerAlp(flux.axionEnergy[i], flux.ma);
    }
    r.thetaKinMrad = (wk > 0.0) ? 1000.0 * swk / wk : 0.0;

    double sw = 0.0, swt = 0.0, swIn = 0.0, swtIn = 0.0;
    for (const auto& d : pairs) {
        const double th = OpeningAngle(d.g1, d.g2);
        sw  += d.weight;
        swt += d.weight * th;
        if (TransverseAccept(d.g1, d.g2, geo.detDist_m(),
                             geo.detHalfX_m, geo.detHalfY_m, d.vertexZ_m)) {
            swIn  += d.weight;
            swtIn += d.weight * th;
        }
    }
    if (sw > 0.0) {
        r.meanMrad   = 1000.0 * swt / sw;
        r.acceptFrac = swIn / sw;
    }
    if (swIn > 0.0) r.meanInMrad = 1000.0 * swtIn / swIn;
    return r;
}

// alp_signal_pipeline.py:302 signal_events: both photons on the calo face AND
// each above the calorimeter threshold (a photon below it is missing energy,
// so the pair is not reconstructable).
inline double SignalEvents(const std::vector<DecayPair>& pairs,
                           const Geometry& geo, double threshold_MeV)
{
    double sum = 0.0;
    for (const auto& d : pairs)
        if (d.g1.e >= threshold_MeV && d.g2.e >= threshold_MeV
            && TransverseAccept(d.g1, d.g2, geo.detDist_m(),
                                geo.detHalfX_m, geo.detHalfY_m, d.vertexZ_m))
            sum += d.weight;
    return sum;
}

// alp_signal_pipeline.py:84 bethe_heitler_spectrum, the no-Geant4 fallback.
// dN/dk ~ (1/k)[4/3 - (4/3)(k/E0) + (k/E0)^2] per radiation length.
inline void BetheHeitlerSpectrum(double E0_MeV, double electronsPerS, int nBins,
                                 std::vector<double>& energy,
                                 std::vector<double>& rate)
{
    const double kMin = 1.0, kMax = 0.999 * E0_MeV;
    const double dk = (kMax - kMin) / nBins;
    energy.clear(); rate.clear();
    for (int i = 0; i < nBins; ++i) {
        const double lo = kMin + i * dk;
        const double k = lo + 0.5 * dk;
        const double x = k / E0_MeV;
        const double dNdk = (4.0 / 3.0 - 4.0 * x / 3.0 + x * x) / k;
        energy.push_back(k);
        rate.push_back(dNdk * dk * electronsPerS);
    }
}

}  // namespace damsa::alp

#endif
