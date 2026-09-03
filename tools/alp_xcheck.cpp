// Emit src/alp/ values on a fixed grid, for diffing against alplib.
//
//   ./build/alp_xcheck > cpp.txt
//   python3 tools/alp_xcheck.py > py.txt
//   diff cpp.txt py.txt
//
// Closed-form only (no MC), so the two must agree to round-off. Printed at 17
// significant digits: anything less would hide a real discrepancy.

#include "abs_xs.h"
#include "constants.h"
#include "decay2body.h"
#include "flux.h"
#include "primakoff.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace damsa::alp;

int main(int argc, char** argv)
{
    const std::string alplib = (argc > 1) ? argv[1] : "alplib";
    const AbsCrossSection absXs(alplib + "/data/photon_absorption/photon_abs_W.txt");

    std::printf("# abs_xs_rows %zu\n", absXs.size());

    // Energies spanning the brems range, plus points near the table edges.
    const std::vector<double> energies = {
        1.0e-3, 1.5e-3, 0.01, 0.1, 0.511, 1.0, 1.5, 2.5, 10.0, 100.0,
        1000.0, 4000.0, 7999.5, 8000.0, 1.0e4, 1.0e5};
    const std::vector<double> masses    = {0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0};
    const std::vector<double> couplings = {1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3};

    for (double e : energies)
        std::printf("abs_xs_mev %.17g %.17g\n", e, absXs.SigmaMeV(e));
    for (double e : energies)
        std::printf("abs_xs_cm2 %.17g %.17g\n", e, absXs.SigmaCm2(e));

    for (double g : couplings)
        for (double m : masses)
            std::printf("w_gg %.17g %.17g %.17g\n", g, m, WGammaGamma(g, m));

    std::printf("r0 %.17g\n", kPrimakoffR0);
    for (double g : couplings)
        for (double m : masses)
            for (double e : energies)
                std::printf("primakoff %.17g %.17g %.17g %.17g\n",
                            g, m, e, PrimakoffSigma(e, g, m, kTungstenZ));

    // Full simulate + propagate on a small synthetic spectrum.
    std::vector<double> pe, pr;
    for (int i = 1; i <= 40; ++i) {
        pe.push_back(i * 200.0 - 100.0);     // 100 .. 7900 MeV
        pr.push_back(1e12 / (i * i));
    }

    for (double m : masses) {
        for (double g : couplings) {
            FluxPrimakoffIsotropic f;
            f.ma = m; f.gagamma = g;
            f.detDist_m = 0.35; f.detLength_m = 0.42; f.detArea_m2 = 0.04;
            f.Simulate(pe, pr, absXs);
            f.Propagate(false);

            // axionFlux is plain float64 on both sides, so sum it directly.
            // The weights are float32 in alplib (fluxes.py:64), so sum the
            // float32-cast values here: comparing double sums against float32
            // sums would only re-measure a quantization we already check
            // element by element below (w32). What is left is summation order.
            double sd = 0, ss = 0, sf = 0;
            for (std::size_t i = 0; i < f.axionEnergy.size(); ++i) {
                sf += f.axionFlux[i];
                sd += static_cast<double>(static_cast<float>(f.decayAxionWeight[i]));
                ss += static_cast<double>(static_cast<float>(f.scatterAxionWeight[i]));
            }
            double d30 = 0;
            for (std::size_t i = 0; i < f.axionEnergy.size(); ++i)
                d30 += 30.0 * kSecPerDay
                     * static_cast<double>(static_cast<float>(f.decayAxionWeight[i]));
            std::printf("flux %.17g %.17g n=%zu sumflux=%.17g sumdecay=%.17g sumscat=%.17g decays30=%.17g\n",
                        g, m, f.axionEnergy.size(), sf, sd, ss, d30);

            // Per-element weights cast to float32. alplib stores its weights as
            // float32 (fluxes.py:64), so this is the comparison that can be
            // exact; the double sums above cannot be, since anything below the
            // float32 floor (~1.2e-38) is zero on the Python side.
            for (std::size_t i = 0; i < f.axionEnergy.size(); ++i)
                std::printf("w32 %.17g %.17g %zu %.9g %.9g\n", g, m, i,
                            static_cast<double>(static_cast<float>(f.decayAxionWeight[i])),
                            static_cast<double>(static_cast<float>(f.scatterAxionWeight[i])));
        }
    }
    // ── Boost and two-body decay, deterministic ─────────────────────────────
    const FourVector boostTest[] = {
        {100.0,  0.0,   0.0,   99.9},
        {500.0, 10.0, -20.0,  480.0},
        {8000.0, 0.0,   0.0, 7999.0},
        {50.0,  -5.0,   3.0,  -49.0},
    };
    const double vels[][3] = {
        { 0.0,  0.0,  0.5},
        { 0.1, -0.2,  0.3},
        { 0.0,  0.0,  0.0},        // beta == 0: alplib returns p unchanged
        {-0.6,  0.0,  0.0},
    };
    for (const auto& p4 : boostTest)
        for (const auto& v : vels) {
            const FourVector o = LorentzBoost(p4, v[0], v[1], v[2]);
            std::printf("boost %.17g %.17g %.17g %.17g  %.17g %.17g %.17g  %.17g %.17g %.17g %.17g\n",
                        p4.e, p4.px, p4.py, p4.pz, v[0], v[1], v[2], o.e, o.px, o.py, o.pz);
        }

    // Forward-going ALP, as generators.py:120 builds it (theta = 0).
    const double phis[]   = {0.0, 0.7, 2.0, 4.5, 6.0};
    const double thetas[] = {0.1, 0.9, 1.5708, 2.4, 3.0};
    for (double m : {1.0, 10.0, 100.0, 500.0})
        for (double ea : {600.0, 2000.0, 8000.0}) {
            if (ea <= m) continue;
            const double pa = std::sqrt(ea * ea - m * m);
            const FourVector alp{ea, 0.0, 0.0, pa};
            for (double ph : phis)
                for (double th : thetas) {
                    FourVector g1, g2;
                    Decay2BodyMasslessAt(alp, m, ph, th, g1, g2);
                    std::printf("decay %.17g %.17g %.17g %.17g  %.17g %.17g %.17g %.17g  %.17g %.17g %.17g %.17g\n",
                                m, ea, ph, th, g1.e, g1.px, g1.py, g1.pz,
                                g2.e, g2.px, g2.py, g2.pz);
                }
        }
    return 0;
}
