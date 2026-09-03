// ALP signal pipeline. Replaces scripts/pipeline/alp_signal_pipeline.py.
//
//   ./build/damsa_alp_signal --flux output/alplib_brems_flux.csv --auto-coupling
//
// Reads the Geant4 bremsstrahlung flux, runs Primakoff production and a -> gg
// decay, and writes per-mass decay 4-vectors for Geant4 re-injection plus the
// opening-angle table and sensitivity band.
//
// The flux file carries its own beam normalization in its header; the rates are
// already photons/s at that current and are never rescaled here.

#include "damsa_io.h"
#include "pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace damsa::alp;
namespace io = damsa::io;

namespace {

struct Args {
    std::string fluxPath;
    std::string alplibDir = "alplib";
    std::string outDir    = "output";
    std::vector<double> masses = {1, 5, 10, 20, 50, 100, 200, 500};
    double coupling      = 1e-3;    // GeV^-1
    bool   autoCoupling  = false;
    bool   analytic      = false;
    bool   no4vec        = false;
    bool   noSensitivity = false;
    double threshold_MeV = 5.0;
    int    nDecaySamples = 500;
    int    nAngleSamples = 300;
    int    nSensSamples  = 20;
    std::uint64_t seed   = 1234;
    bool   xcheck        = false;
    Geometry geo;
};

void Usage(const char* p)
{
    std::printf(
        "Usage: %s [--flux <alplib_brems_flux.csv>] [options]\n\n"
        "  --flux PATH            Geant4 brems flux CSV (omit with --analytic)\n"
        "  --analytic             Use the Bethe-Heitler spectrum instead\n"
        "  --alplib DIR           alplib checkout, for the absorption table (default alplib)\n"
        "  --outdir DIR           Output directory (default output)\n"
        "  --mass MEV             Single ALP mass (default: scan 1..500)\n"
        "  --coupling GEV_INV     g_agamma in GeV^-1 (default 1e-3)\n"
        "  --auto-coupling        Pick g per mass so surv_prob ~ 1\n"
        "  --threshold-mev MEV    Per-photon calorimeter threshold (default 5)\n"
        "  --vdc-length M         VDC length in metres (default %.2f)\n"
        "  --decay-samples N      Decays per flux bin for export (default 500)\n"
        "  --angle-samples N      Decays per flux bin for angles (default 300)\n"
        "  --sens-samples N       Decays per flux bin for sensitivity (default 20)\n"
        "  --exposure-days D      Exposure (default 30)\n"
        "  --seed N               RNG seed (default 1234)\n"
        "  --no-4vec              Skip the 4-vector export\n"
        "  --no-sensitivity       Skip the sensitivity scan\n"
        "  --xcheck               Machine-readable table for tools/alp_signal_xcheck.py\n",
        p, Geometry{}.vdc_m);
}

bool Parse(int argc, char** argv, Args& a)
{
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (s == "--flux")           a.fluxPath = next();
        else if (s == "--alplib")         a.alplibDir = next();
        else if (s == "--outdir")         a.outDir = next();
        else if (s == "--analytic")       a.analytic = true;
        else if (s == "--mass")           a.masses = {std::stod(next())};
        else if (s == "--coupling")       a.coupling = std::stod(next());
        else if (s == "--auto-coupling")  a.autoCoupling = true;
        else if (s == "--threshold-mev")  a.threshold_MeV = std::stod(next());
        else if (s == "--vdc-length")     a.geo.vdc_m = std::stod(next());
        else if (s == "--decay-samples")  a.nDecaySamples = std::stoi(next());
        else if (s == "--angle-samples")  a.nAngleSamples = std::stoi(next());
        else if (s == "--sens-samples")   a.nSensSamples = std::stoi(next());
        else if (s == "--exposure-days")  a.geo.exposureDays = std::stod(next());
        else if (s == "--seed")           a.seed = std::stoull(next());
        else if (s == "--no-4vec")        a.no4vec = true;
        else if (s == "--no-sensitivity") a.noSensitivity = true;
        else if (s == "--xcheck")         a.xcheck = true;
        else if (s == "-h" || s == "--help") { Usage(argv[0]); return false; }
        else { std::fprintf(stderr, "Unknown argument: %s\n", s.c_str()); return false; }
    }
    if (a.fluxPath.empty() && !a.analytic) {
        std::fprintf(stderr, "Error: --flux or --analytic is required.\n");
        Usage(argv[0]);
        return false;
    }
    return true;
}

// The 60-point log grid from alp_signal_pipeline.py: np.logspace(-8, -2, 60).
std::vector<double> CouplingGrid()
{
    std::vector<double> g;
    for (int i = 0; i < 60; ++i)
        g.push_back(std::pow(10.0, -8.0 + 6.0 * i / 59.0));
    return g;
}

}  // namespace

int main(int argc, char** argv)
{
    Args a;
    if (!Parse(argc, argv, a)) return 1;

    // ── Photon flux ─────────────────────────────────────────────────────────
    std::vector<double> energy, rate;
    if (a.analytic) {
        const double current = 4200.0 * 18 * 929e3 * kChargeCoulombs;
        BetheHeitlerSpectrum(a.geo.beamEnergy_MeV, current / kChargeCoulombs, 500,
                             energy, rate);
        std::printf("[flux] analytic Bethe-Heitler, %zu bins\n", energy.size());
    } else {
        const auto f = io::ReadBremsFlux(a.fluxPath);
        energy = f.energy_MeV;
        rate   = f.rate_per_s;
        std::printf("[flux] %zu bins from %s\n", f.size(), a.fluxPath.c_str());
        if (!f.beamMode.empty())
            std::printf("[flux] beam mode: %s (delivered current %.3e A)\n",
                        f.beamMode.c_str(), f.beamCurrent_A);
        else
            std::printf("[flux] WARNING: no beam-mode header; assuming rates are "
                        "already normalized\n");
        std::printf("[flux] energy range: %.1f - %.1f MeV, total %.3e photons/s\n",
                    energy.front(), energy.back(), f.totalRate());
    }

    const AbsCrossSection absXs(a.alplibDir + "/data/photon_absorption/photon_abs_W.txt");

    if (!a.xcheck) {
    std::printf("\n=== calculation ranges ===\n");
    std::printf("  decay window:  z = %.0f - %.0f cm from target centre "
                "(target exit -> calo face)\n",
                a.geo.decayZMin_m() * 100, a.geo.decayZMax_m() * 100);
    std::printf("  calo face:     %.0f x %.0f cm at %.1f cm\n",
                2 * a.geo.detHalfX_m * 100, 2 * a.geo.detHalfY_m * 100,
                a.geo.detDist_m() * 100);
    std::printf("  exposure:      %.0f days,  per-photon threshold %.1f MeV\n",
                a.geo.exposureDays, a.threshold_MeV);
    }

    std::mt19937_64 rng(a.seed);

    // ── Opening angles ──────────────────────────────────────────────────────
    if (a.xcheck) {
        std::printf("%-10s %-14s %-16s %-16s %-14s %-12s\n",
                    "ma_MeV", "coupling", "theta_kin_mrad", "total_weight",
                    "theta_MC_mrad", "accept");
    } else {
        std::printf("\n=== opening angle scan ===\n");
        std::printf("%10s %12s %13s %14s %8s %13s %11s\n",
                    "ma (MeV)", "g (GeV^-1)", "th_MC (mrad)", "th_kin (mrad)",
                    "MC/kin", "th_in (mrad)", "accept");
        std::printf("%s\n", std::string(88, '-').c_str());
    }

    std::vector<double> usedCoupling(a.masses.size(), 0.0);
    for (std::size_t i = 0; i < a.masses.size(); ++i) {
        const double ma = a.masses[i];
        const double g = a.autoCoupling ? PickSafeCoupling(ma, energy, rate)
                                        : a.coupling;
        usedCoupling[i] = g;

        auto flux = MakeFlux(a.geo, ma, g);
        flux.Simulate(energy, rate, absXs);
        flux.Propagate(false);
        if (flux.axionEnergy.empty()) {
            std::printf("%10.1f %12.2e  (no ALP production: ma above the flux)\n", ma, g);
            continue;
        }

        const auto pairs = SimulateDecay4Vectors(flux, a.geo, g, a.nAngleSamples, rng);
        const auto r = ComputeOpeningAngles(pairs, flux, a.geo, g);

        if (a.xcheck) {
            // total_weight is deterministic: every sample carries weight/n and
            // there are exactly n of them, so it does not depend on the draws.
            double total = 0.0;
            for (double w : flux.decayAxionWeight) total += w;
            total *= a.geo.exposureDays * kSecPerDay;
            std::printf("%-10.1f %-14.8e %-16.10f %-16.10e %-14.4f %-12.6e\n",
                        ma, g, r.thetaKinMrad, total, r.meanMrad, r.acceptFrac);
        } else {
            const double ratio = (r.thetaKinMrad > 0) ? r.meanMrad / r.thetaKinMrad : 0.0;
            std::printf("%10.1f %12.2e %13.2f %14.2f %8.3f %13.2f %11.3e\n",
                        ma, g, r.meanMrad, r.thetaKinMrad, ratio, r.meanInMrad, r.acceptFrac);
        }
    }

    // ── 4-vector export ─────────────────────────────────────────────────────
    if (!a.no4vec) {
        std::printf("\n=== exporting decay 4-vectors -> %s/ ===\n", a.outDir.c_str());
        for (std::size_t i = 0; i < a.masses.size(); ++i) {
            const double ma = a.masses[i];
            const double g = usedCoupling[i];

            auto flux = MakeFlux(a.geo, ma, g);
            flux.Simulate(energy, rate, absXs);
            flux.Propagate(false);
            if (flux.axionEnergy.empty()) {
                std::printf("  [ma=%.0f MeV] no ALP events generated\n", ma);
                continue;
            }

            char name[256];
            std::snprintf(name, sizeof(name), "%s/alp_decay_photons_ma%.0fMeV.root",
                          a.outDir.c_str(), ma);

            const auto t0 = std::chrono::steady_clock::now();
            io::NTupleWriter<io::AlpDecayRow> w(name);
            double total = 0.0;
            std::size_t n = 0;

            // Streamed per flux bin: at 7999 bins x 500 samples this is 4M rows,
            // which is exactly what made the old CSVs ~1 GB each.
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            const double perSample = a.geo.exposureDays * kSecPerDay / double(a.nDecaySamples);
            for (std::size_t k = 0; k < flux.axionEnergy.size(); ++k) {
                const double ea = flux.axionEnergy[k];
                const double pa = std::sqrt(std::max(ea * ea - ma * ma, 0.0));
                const FourVector alp{ea, 0.0, 0.0, pa};
                const double wt = perSample * flux.decayAxionWeight[k];
                for (int s = 0; s < a.nDecaySamples; ++s) {
                    FourVector g1, g2;
                    Decay2BodyMassless(alp, ma, rng, g1, g2);
                    io::AlpDecayRow row;
                    row.E1 = g1.e; row.px1 = g1.px; row.py1 = g1.py; row.pz1 = g1.pz;
                    row.E2 = g2.e; row.px2 = g2.px; row.py2 = g2.py; row.pz2 = g2.pz;
                    row.weight = wt;
                    row.decayZ_m = SampleDecayVertexZ(ea, ma, g, a.geo.decayZMin_m(),
                                                      a.geo.decayZMax_m(), uni(rng));
                    w.Fill(row);
                    total += wt;
                    ++n;
                }
            }
            w.Finish();
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  [ma=%.0f MeV, g=%.2e GeV^-1] %zu pairs -> %s  "
                        "(total weight %.3e over %.0f days, %.1fs)\n",
                        ma, g, n, name, total, a.geo.exposureDays, secs);
        }
    }

    // ── Sensitivity ─────────────────────────────────────────────────────────
    if (!a.noSensitivity) {
        std::printf("\n=== sensitivity scan ===\n");
        const auto grid = CouplingGrid();
        const std::string outCsv = a.outDir + "/alp_sensitivity.csv";
        io::EnsureParentDir(outCsv);
        std::ofstream out(outCsv);
        out << "ma_MeV,g_lower_GeVinv,g_upper_GeVinv,n_sig_max\n";

        for (double ma : a.masses) {
            double gLo = std::nan(""), gHi = std::nan(""), nMax = 0.0;
            for (double g : grid) {
                auto flux = MakeFlux(a.geo, ma, g);
                flux.Simulate(energy, rate, absXs);
                flux.Propagate(false);
                if (flux.axionEnergy.empty()) continue;
                const auto pairs = SimulateDecay4Vectors(flux, a.geo, g, a.nSensSamples, rng);
                const double n = SignalEvents(pairs, a.geo, a.threshold_MeV);
                if (n > 2.3) {                    // 90% CL, one-sided
                    if (std::isnan(gLo)) gLo = g;
                    gHi = g;
                    nMax = std::max(nMax, n);
                }
            }
            if (std::isnan(gLo))
                std::printf("  ma=%6.1f MeV -> not excluded in the coupling range\n", ma);
            else
                std::printf("  ma=%6.1f MeV -> excluded band [%.2e, %.2e] GeV^-1 "
                            "(max N_sig=%.1f)\n", ma, gLo, gHi, nMax);
            out << ma << "," << gLo << "," << gHi << "," << nMax << "\n";
        }
        std::printf("  wrote %s\n", outCsv.c_str());
    }

    std::printf("\nDone.\n");
    return 0;
}
