#ifndef DAMSA_ALP_SCAN_H
#define DAMSA_ALP_SCAN_H

// Shared helpers for the geometry-scan tools (damsa_joint_scan,
// damsa_combine_pareto, damsa_target_projection, damsa_report).

#include "damsa_io.h"
#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace damsa::alp {

// ── Shared geometry-scan machinery ──────────────────────────────────────────
// Used by damsa_joint_scan (VDC x calo at one target length) and
// damsa_target_projection (the same grid across target lengths).

inline constexpr double kTargetExitZ_mm  = -400.0;   // target rear face
inline constexpr double kMagnetLength_mm =  120.0;   // fixed magnet + tracker
inline constexpr double kCharge_C        = 1.602176634e-19;

// fast_pareto_scan.py:113 auto_coupling. NOTE ma^3 here, where
// alp_signal_pipeline.py's pick_safe_coupling uses ma^4 -- different targets,
// do not conflate them.
inline double AutoCoupling(const std::vector<double>& e, const std::vector<double>& w,
                           double ma, double targetDecayLength_m = 0.6)
{
    double sw = 0, swx = 0;
    for (std::size_t i = 0; i < e.size(); ++i)
        if (e[i] > ma) { sw += w[i]; swx += w[i] * e[i]; }
    if (sw == 0.0) return 1e-4;
    const double eaTyp = swx / sw;
    const double g2 = 64.0 * M_PI * kMeterByMeV * eaTyp
                    / (targetDecayLength_m * std::pow(ma, 3));
    return std::sqrt(std::max(g2, 0.0)) * 1000.0;
}

// joint_pareto_scan.py:59. Straight-line propagation of target-exit photons and
// neutrons to the calo face, for every (VDC, calo) pair.
inline std::vector<std::vector<double>> BackgroundGrid(
    const std::vector<io::ParticleRow>& parts,
    const std::vector<double>& vdc, const std::vector<double>& calo,
    double norm, double neutronWeight = 10.0)
{
    std::vector<std::vector<double>> g(vdc.size(), std::vector<double>(calo.size(), 0.0));
    for (std::size_t iv = 0; iv < vdc.size(); ++iv) {
        const double zCalo = kTargetExitZ_mm + vdc[iv] * 10.0 + kMagnetLength_mm;
        for (const auto& p : parts) {
            if ((p.pdg != 22 && p.pdg != 2112) || p.pz <= 0) continue;
            const double dt = (zCalo - p.z_mm) / p.pz;
            const double xc = p.x_mm + dt * p.px;
            const double yc = p.y_mm + dt * p.py;
            const double sw = (p.pdg == 2112) ? neutronWeight : 1.0;
            for (std::size_t ic = 0; ic < calo.size(); ++ic) {
                const double hw = calo[ic] / 2.0 * 10.0;   // half-width [mm]
                if (std::abs(xc) <= hw && std::abs(yc) <= hw)
                    g[iv][ic] += p.weight * sw * norm;
            }
        }
    }
    return g;
}

// Per-decay quantities the acceptance grid needs.
struct ScanEvent {
    double weight, E1, E2, thetaDeg, decayLen_cm;
};

// joint_pareto_scan.py calls flux_obj.propagate(decay_width), but
// FluxPrimakoffIsotropic.propagate's first parameter is new_coupling, not a
// width (fluxes.py:151). So the decay WIDTH is used as a COUPLING: the rescale
// becomes (width/g)^2 and the width handed to the base propagate is
// W_gg(width, ma) -- around 1e-40 rather than the intended 1e-20.
//
// That is not a constant factor, so it distorts the weight distribution over
// ALP energy and therefore the accepted/separable fractions these scans report.
// It does NOT affect the sampled decay positions, which use tau = HBAR/width
// with the correct width.
//
// Reproduced by default so the port can be diffed against the Python;
// fixWeights uses the intended physics instead.
inline void PropagateForScan(FluxPrimakoffIsotropic& f, double g_MeV, bool fixWeights)
{
    if (fixWeights) { f.Propagate(false); return; }
    f.Propagate(false, WGammaGamma(g_MeV, f.ma));
}

// joint_pareto_scan.py generate_events.
template <class RNG>
inline std::vector<ScanEvent> GenerateScanEvents(
    const io::BremsFlux& flux, const AbsCrossSection& absXs,
    double ma, double gMeV, int nSamples,
    double detDist_m, double detArea_m2, double exposureDays,
    RNG& rng, bool fixWeights, double& nTotalOut)
{
    std::vector<ScanEvent> events;
    nTotalOut = 0.0;

    FluxPrimakoffIsotropic f;
    f.ma = ma; f.gagamma = gMeV; f.targetZ = kTungstenZ;
    f.detDist_m = detDist_m; f.detLength_m = detDist_m; f.detArea_m2 = detArea_m2;
    f.Simulate(flux.energy_MeV, flux.rate_per_s, absXs);
    if (f.axionEnergy.empty()) return events;

    PropagateForScan(f, gMeV, fixWeights);

    const double tauRest = kHbar / WGammaGamma(gMeV, ma);     // s
    nTotalOut = f.Decays(exposureDays, 0.1);

    const double perSample = exposureDays * kSecPerDay / double(nSamples);
    for (std::size_t i = 0; i < f.axionEnergy.size(); ++i) {
        const double ea = f.axionEnergy[i];
        const double pa = std::sqrt(std::max(ea * ea - ma * ma, 0.0));
        const FourVector alp{ea, 0.0, 0.0, pa};
        const double w = perSample * f.decayAxionWeight[i];
        if (w <= 0) continue;
        for (int s = 0; s < nSamples; ++s) {
            FourVector g1, g2;
            Decay2BodyMassless(alp, ma, rng, g1, g2);
            const double eAlp = g1.e + g2.e;
            if (g1.e <= 0 || g2.e <= 0 || eAlp <= ma) continue;
            const double gamma = eAlp / ma;
            const double beta = std::sqrt(std::max(1.0 - std::pow(ma / eAlp, 2), 0.0));
            events.push_back({w, g1.e, g2.e,
                              OpeningAngle(g1, g2) * 180.0 / M_PI,
                              gamma * beta * kCLight * tauRest});
        }
    }
    return events;
}

// joint_pareto_scan.py:102 geometric_acceptance_grid. Decay positions are
// redrawn once per VDC value and reused across calo sizes, as the Python does.
template <class RNG>
inline void AcceptanceGrid(const std::vector<ScanEvent>& events,
                           const std::vector<double>& vdc,
                           const std::vector<double>& calo,
                           double angleCut, double energyCut, RNG& rng,
                           std::vector<std::vector<double>>& acc,
                           std::vector<std::vector<double>>& sep)
{
    acc.assign(vdc.size(), std::vector<double>(calo.size(), 0.0));
    sep.assign(vdc.size(), std::vector<double>(calo.size(), 0.0));

    double totalW = 0; for (const auto& e : events) totalW += e.weight;
    if (totalW <= 0) return;

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double magnet_cm = kMagnetLength_mm / 10.0;

    for (std::size_t iv = 0; iv < vdc.size(); ++iv) {
        std::vector<double> halfSep(events.size());
        std::vector<char>   inGap(events.size());
        for (std::size_t k = 0; k < events.size(); ++k) {
            const double u = uni(rng);
            const double z = -events[k].decayLen_cm * std::log(std::max(1.0 - u, 1e-30));
            inGap[k] = (z > 0) && (z < vdc[iv]);
            halfSep[k] = (events[k].thetaDeg * M_PI / 180.0)
                       * ((vdc[iv] + magnet_cm) - z) / 2.0;
        }
        for (std::size_t ic = 0; ic < calo.size(); ++ic) {
            const double hw = calo[ic] / 2.0;
            double wa = 0, ws = 0;
            for (std::size_t k = 0; k < events.size(); ++k) {
                if (!inGap[k] || halfSep[k] > hw) continue;
                wa += events[k].weight;
                const auto& e = events[k];
                if (e.thetaDeg >= angleCut || (e.E1 >= energyCut && e.E2 >= energyCut))
                    ws += e.weight;
            }
            acc[iv][ic] = wa / totalW;
            sep[iv][ic] = ws / totalW;
        }
    }
}

// joint_pareto_scan.py:153 is_pareto_optimal.
// objectives[i] is one row; minimize[j] says whether column j is minimised.
// O(N^2), which is fine: these grids are a few thousand rows at most.
// ponytail: O(N^2) Pareto scan; switch to a sorted sweep if a grid ever gets big
inline std::vector<bool> IsParetoOptimal(const std::vector<std::vector<double>>& objectives,
                                         const std::vector<bool>& minimize)
{
    const std::size_t n = objectives.size();
    std::vector<bool> keep(n, true);
    if (n == 0) return keep;
    const std::size_t m = minimize.size();

    // Flip maximised columns so only minimisation needs handling.
    std::vector<std::vector<double>> obj = objectives;
    for (auto& row : obj)
        for (std::size_t j = 0; j < m && j < row.size(); ++j)
            if (!minimize[j]) row[j] = -row[j];

    for (std::size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        for (std::size_t k = 0; k < n; ++k) {
            if (k == i) continue;
            bool allLE = true, anyLT = false;
            for (std::size_t j = 0; j < m; ++j) {
                if (obj[k][j] > obj[i][j]) { allLE = false; break; }
                if (obj[k][j] < obj[i][j]) anyLT = true;
            }
            if (allLE && anyLT) { keep[i] = false; break; }
        }
    }
    return keep;
}

// Minimal row-oriented CSV writer for the scan result tables. The columns are
// fixed per tool, so a header plus ordered values is all that is needed.
class CsvOut {
public:
    CsvOut(const std::string& path, const std::string& header) : fOut(path)
    {
        if (!fOut.is_open()) throw std::runtime_error("cannot write " + path);
        fOut << header << "\n";
        fOut << std::scientific;
    }
    template <class... T>
    void Row(T... vals) { ((fOut << vals << ","), ...); fOut.seekp(-1, std::ios_base::cur); fOut << "\n"; }
    void Raw(const std::string& line) { fOut << line << "\n"; }
    void Close() { fOut.close(); }
private:
    std::ofstream fOut;
};

}  // namespace damsa::alp

#endif
