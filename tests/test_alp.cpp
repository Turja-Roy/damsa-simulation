// Self-check for src/alp/. Complements tools/alp_xcheck.cpp: that proves we
// agree with alplib, this proves the physics identities hold -- a shared error
// would pass the former and fail this.

#include "abs_xs.h"
#include "constants.h"
#include "decay2body.h"
#include "flux.h"
#include "primakoff.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

using namespace damsa::alp;

static bool Close(double a, double b, double rtol)
{
    const double d = std::max(std::abs(a), std::abs(b));
    return d == 0.0 ? std::abs(a - b) < 1e-300 : std::abs(a - b) / d < rtol;
}

static void TestDecayInvariants()
{
    // Tolerance is set by alplib's gamma = 1/sqrt(1 - beta^2), which cancels as
    // beta -> 1: at ma = 1 MeV / E = 8 GeV only ~8 digits survive. This bound is
    // the measured worst case, so tightening it catches a regression while the
    // value itself records an inherited limitation, not a porting error.
    constexpr double kBoostTol = 1e-8;

    std::mt19937_64 rng(12345);
    double worstE = 0, worstP = 0, worstM2 = 0;
    for (double ma : {1.0, 10.0, 100.0, 500.0}) {
        for (double ea : {600.0, 2000.0, 8000.0}) {
            const double pa = std::sqrt(ea * ea - ma * ma);
            const FourVector alp{ea, 0.0, 0.0, pa};
            for (int i = 0; i < 200; ++i) {
                FourVector g1, g2;
                Decay2BodyMassless(alp, ma, rng, g1, g2);

                // Energy-momentum conservation.
                worstE = std::max(worstE, std::abs((g1.e + g2.e) - ea) / ea);
                worstP = std::max(worstP, std::abs((g1.pz + g2.pz) - pa) / pa);
                assert(Close(g1.e + g2.e, ea, kBoostTol));
                assert(Close(g1.pz + g2.pz, pa, kBoostTol));
                // Transverse momenta cancel exactly: p2 is built as -p1 in the CM.
                assert(std::abs(g1.px + g2.px) < 1e-9);
                assert(std::abs(g1.py + g2.py) < 1e-9);

                // Daughters are massless.
                for (const auto& g : {g1, g2}) {
                    const double m2 = g.e * g.e - g.px * g.px - g.py * g.py - g.pz * g.pz;
                    worstM2 = std::max(worstM2, std::abs(m2) / (g.e * g.e));
                    assert(std::abs(m2) / (g.e * g.e) < 1e-11);
                }

                // Invariant mass of the pair reconstructs the ALP mass.
                const double E = g1.e + g2.e;
                const double px = g1.px + g2.px, py = g1.py + g2.py, pz = g1.pz + g2.pz;
                const double mgg = std::sqrt(E * E - px * px - py * py - pz * pz);
                assert(Close(mgg, ma, 1e-6));   // sqrt halves the digits lost above

                // Forward boost: both photons go downstream for these energies.
                assert(g1.e > 0 && g2.e > 0);
            }
        }
    }
    std::printf("  decay invariants: worst rel err  E %.2e  pz %.2e  m^2 %.2e\n",
                worstE, worstP, worstM2);
    std::printf("                    (bounded by alplib's gamma cancellation, not the port)\n");
}

static void TestDecayIsotropy()
{
    // In the ALP rest frame the decay is isotropic, so <cos(theta*)> == 0.
    std::mt19937_64 rng(999);
    const double ma = 100.0, ea = 100.0;   // at rest: gamma = 1
    const FourVector alp{ea, 0.0, 0.0, 0.0};
    double sum = 0.0;
    const int n = 20000;
    for (int i = 0; i < n; ++i) {
        FourVector g1, g2;
        Decay2BodyMassless(alp, ma, rng, g1, g2);
        sum += g1.pz / std::sqrt(g1.px * g1.px + g1.py * g1.py + g1.pz * g1.pz);
    }
    const double mean = sum / n;
    assert(std::abs(mean) < 5.0 / std::sqrt(double(n)));   // 5 sigma
    std::printf("  decay isotropy: <cos t*> = %+.5f over %d draws\n", mean, n);
}

static void TestPrimakoffThreshold()
{
    // heaviside(eg - ma, 0.0): zero below AND at threshold.
    assert(PrimakoffSigma(50.0, 1e-3, 100.0, kTungstenZ) == 0.0);
    assert(PrimakoffSigma(100.0, 1e-3, 100.0, kTungstenZ) == 0.0);
    assert(PrimakoffSigma(100.1, 1e-3, 100.0, kTungstenZ) > 0.0);

    // sigma scales as g^2.
    const double s1 = PrimakoffSigma(1000.0, 1e-6, 10.0, kTungstenZ);
    const double s2 = PrimakoffSigma(1000.0, 2e-6, 10.0, kTungstenZ);
    assert(Close(s2 / s1, 4.0, 1e-12));
    std::printf("  primakoff: threshold closed at eg == ma, sigma ~ g^2\n");
}

static void TestWidthScaling()
{
    // Gamma = g^2 ma^3 / 64pi.
    assert(Close(WGammaGamma(2e-3, 10.0) / WGammaGamma(1e-3, 10.0), 4.0, 1e-12));
    assert(Close(WGammaGamma(1e-3, 20.0) / WGammaGamma(1e-3, 10.0), 8.0, 1e-12));
    std::printf("  W_gg: scales as g^2 and ma^3\n");
}

static void TestFluxMonotonicity(const std::string& alplib)
{
    const AbsCrossSection absXs(alplib + "/data/photon_absorption/photon_abs_W.txt");
    assert(absXs.size() == 94);   // 105 lines, duplicate energies at edges

    std::vector<double> pe, pr;
    for (int i = 1; i <= 40; ++i) { pe.push_back(i * 200.0); pr.push_back(1e12 / (i * i)); }

    // Production is ~ g^2, so before propagation the flux must scale that way.
    FluxPrimakoffIsotropic a, b;
    a.ma = b.ma = 10.0;
    a.gagamma = 1e-6; b.gagamma = 2e-6;
    a.Simulate(pe, pr, absXs); b.Simulate(pe, pr, absXs);
    double sa = 0, sb = 0;
    for (std::size_t i = 0; i < a.axionFlux.size(); ++i) { sa += a.axionFlux[i]; sb += b.axionFlux[i]; }
    assert(Close(sb / sa, 4.0, 1e-9));

    // Rows below threshold are dropped, not zero-weighted.
    FluxPrimakoffIsotropic hi;
    hi.ma = 3000.0; hi.gagamma = 1e-6;
    hi.Simulate(pe, pr, absXs);
    std::size_t expected = 0;
    for (double e : pe) if (e >= hi.ma) ++expected;
    assert(hi.axionEnergy.size() == expected);
    std::printf("  flux: N_ALP ~ g^2, %zu/%zu rows survive ma = 3 GeV\n", expected, pe.size());
}

int main(int argc, char** argv)
{
    const std::string alplib = (argc > 1) ? argv[1] : "alplib";
    std::printf("src/alp self-check\n");
    TestPrimakoffThreshold();
    TestWidthScaling();
    TestDecayInvariants();
    TestDecayIsotropy();
    TestFluxMonotonicity(alplib);
    std::printf("all passed\n");
    return 0;
}
