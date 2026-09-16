// Internal consistency checks for the ALP signal chain, with diagnostic plots.
// Replaces scripts/analysis/print_weight_breakdown.py,
// scripts/analysis/validate_opening_angles.py and
// scripts/analysis/angle_validation/angle_diagnostics.py.
//
//   ./build/damsa_validate --flux output/alplib_brems_flux.csv --alplib alplib
//
// These are INTERNAL checks: not a comparison against an external analytic
// weighting of the photon spectrum (which is ambiguous), but a verification that
// the MC two-body decay agrees with the kinematics of the SAME ALP ensemble.
//
// Complements tests/test_alp.cpp, which covers the exact conservation
// identities; this covers the approximations and the weight decomposition.

#include "damsa_io.h"
#include "pipeline.h"
#include "plotting.h"

#include <TCanvas.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

using namespace damsa::alp;
namespace io = damsa::io;

int main(int argc, char** argv)
{
    std::string fluxPath = "output/alplib_brems_flux.csv";
    std::string alplibDir = "alplib";
    std::string outDir = "plots/validation";
    std::vector<double> masses = {1, 5, 10, 20, 50, 100, 200, 500};
    int nEvents = 10, nSamples = 200;
    double coupling = -1;
    std::uint64_t seed = 1234;
    Geometry geo;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--flux")     fluxPath = nx();
        else if (s == "--alplib")   alplibDir = nx();
        else if (s == "--out-dir")  outDir = nx();
        else if (s == "--nevents")  nEvents = std::stoi(nx());
        else if (s == "--nsamples") nSamples = std::stoi(nx());
        else if (s == "--coupling") coupling = std::stod(nx());
        else if (s == "--seed")     seed = std::stoull(nx());
        else if (s == "--ma-list") {
            masses.clear(); std::string v = nx(), tok; std::istringstream ss(v);
            while (std::getline(ss, tok, ',')) masses.push_back(std::stod(tok));
        }
        else { std::fprintf(stderr, "Unknown argument: %s\n", s.c_str()); return 1; }
    }

    const auto flux = io::ReadBremsFlux(fluxPath);
    const AbsCrossSection absXs(alplibDir + "/data/photon_absorption/photon_abs_W.txt");
    std::filesystem::create_directories(outDir);
    SetPublicationStyle();
    std::mt19937_64 rng(seed);
    int failures = 0;

    // ── 1. Weight decomposition ─────────────────────────────────────────────
    // weight = N_ALP_produced x P_survival x P_decay_in_window, and this prints
    // each factor so it is visible WHICH one drives the yield at a given mass.
    std::printf("=== 1. Weight breakdown (first %d flux bins per mass) ===\n", nEvents);
    std::printf("%8s %10s %12s %12s %12s %12s\n",
                "ma[MeV]", "Ea[MeV]", "N_ALP", "P_surv", "P_decay", "weight");
    std::printf("%s\n", std::string(70, '-').c_str());

    for (double ma : {masses.front(), masses[masses.size() / 2], masses.back()}) {
        const double g = PickSafeCoupling(ma, flux.energy_MeV, flux.rate_per_s) / 1000.0;
        auto f = MakeFlux(geo, ma, g * 1000.0);
        f.Simulate(flux.energy_MeV, flux.rate_per_s, absXs);
        if (f.axionEnergy.empty()) continue;
        f.Propagate(false);

        const double width = WGammaGamma(g, ma);
        for (int k = 0; k < nEvents && k < int(f.axionEnergy.size()); ++k) {
            const std::size_t i = f.axionEnergy.size() * k / nEvents;   // spread over the spectrum
            const double ea = f.axionEnergy[i];
            const double pa = std::sqrt(ea * ea - ma * ma);
            const double va = pa / ea, tau = (ea / ma) / width;
            const double surv  = std::exp(-f.detDist_m / kMeterByMeV / va / tau);
            const double decay = 1.0 - std::exp(-f.detLength_m / kMeterByMeV / va / tau);
            std::printf("%8.0f %10.1f %12.4e %12.4e %12.4e %12.4e\n",
                        ma, ea, f.axionFlux[i], surv, decay, f.decayAxionWeight[i]);
            // The identity the decomposition asserts.
            const double recon = f.axionFlux[i] * surv * decay;
            if (std::abs(recon - f.decayAxionWeight[i]) > 1e-9 * std::max(recon, 1e-300)) {
                std::printf("  MISMATCH: N*Psurv*Pdecay = %.6e != weight\n", recon);
                ++failures;
            }
        }
        std::printf("\n");
    }

    // ── 2. Opening angle: MC vs kinematics ──────────────────────────────────
    // The UR limit approaches pi*ma/<Ea>, NOT 2*ma/<Ea> -- the latter is the
    // MINIMUM opening angle, not the mean. Both are printed so the distinction
    // stays visible.
    std::printf("=== 2. Opening angle, MC vs kinematics ===\n");
    std::printf("%8s %12s %12s %12s %10s %10s\n",
                "ma[MeV]", "th_MC[mrad]", "th_kin[mrad]", "th_min[mrad]", "MC/kin", "verdict");
    std::printf("%s\n", std::string(70, '-').c_str());

    std::vector<double> gMass, gRatio;
    for (double ma : masses) {
        const double g = (coupling > 0 ? coupling
                                       : PickSafeCoupling(ma, flux.energy_MeV, flux.rate_per_s));
        auto f = MakeFlux(geo, ma, g);
        f.Simulate(flux.energy_MeV, flux.rate_per_s, absXs);
        if (f.axionEnergy.empty()) continue;
        f.Propagate(false);

        const auto pairs = SimulateDecay4Vectors(f, geo, g, nSamples, rng);
        const auto r = ComputeOpeningAngles(pairs, f, geo, g);

        // Weighted <Ea> over the same ensemble, for the UR formulas.
        double sw = 0, swe = 0;
        for (std::size_t i = 0; i < f.axionEnergy.size(); ++i) {
            sw += f.decayAxionWeight[i];
            swe += f.decayAxionWeight[i] * f.axionEnergy[i];
        }
        const double eaMean = sw > 0 ? swe / sw : 0.0;
        const double thMin = eaMean > 0 ? 1000.0 * 2.0 * ma / eaMean : 0.0;

        const double ratio = r.thetaKinMrad > 0 ? r.meanMrad / r.thetaKinMrad : 0.0;
        const bool ok = std::abs(ratio - 1.0) < 0.05;
        if (!ok) ++failures;
        std::printf("%8.0f %12.2f %12.2f %12.2f %10.3f %10s\n",
                    ma, r.meanMrad, r.thetaKinMrad, thMin, ratio, ok ? "ok" : "OFF >5%");
        gMass.push_back(ma); gRatio.push_back(ratio);
    }

    {
        auto* c = new TCanvas("c_ratio", "", 800, 600);
        c->SetLogx();
        auto* g = new TGraph(gMass.size(), gMass.data(), gRatio.data());
        g->SetTitle("MC opening angle vs exact kinematics;m_{a} [MeV];#theta_{MC} / #theta_{kin}");
        g->SetMarkerStyle(20); g->SetMarkerColor(kAzure + 2);
        g->SetLineColor(kAzure + 2); g->SetLineWidth(2);
        g->Draw("ALP");
        g->GetYaxis()->SetRangeUser(0.9, 1.1);
        SaveCanvas(c, outDir + "/opening_angle_ratio");
        delete c;
    }

    // ── 3. Signal scaling with coupling ─────────────────────────────────────
    // N_ALP ~ g^2 from production; the decay probability adds another g^2 while
    // the ALP still reaches the detector, so the yield goes as g^4 at small g
    // and then turns over once it decays before arriving.
    std::printf("\n=== 3. Coupling scaling at ma = %.0f MeV ===\n", masses[masses.size() / 2]);
    std::printf("%12s %14s %10s\n", "g [GeV^-1]", "N_events", "ratio/g^4");
    const double maS = masses[masses.size() / 2];
    double prevG = 0, prevN = 0;
    std::vector<double> sg, sn;
    for (double e = -7.0; e <= -3.0 + 1e-9; e += 0.5) {
        const double g = std::pow(10.0, e);
        auto f = MakeFlux(geo, maS, g);
        f.Simulate(flux.energy_MeV, flux.rate_per_s, absXs);
        if (f.axionEnergy.empty()) continue;
        f.Propagate(false);
        const double n = f.Decays(geo.exposureDays, 0.0);
        double scal = 0;
        if (prevN > 0 && n > 0) scal = (n / prevN) / std::pow(g / prevG, 4);
        std::printf("%12.2e %14.4e %10s\n", g, n,
                    prevN > 0 ? Form("%.3f", scal) : "-");
        prevG = g; prevN = n;
        if (n > 0) { sg.push_back(g); sn.push_back(n); }
    }
    if (sg.size() > 1) {
        auto* c = new TCanvas("c_scal", "", 800, 600);
        c->SetLogx(); c->SetLogy();
        auto* g = new TGraph(sg.size(), sg.data(), sn.data());
        g->SetTitle(Form("Signal yield vs coupling, m_{a}=%.0f MeV;"
                         "g_{a#gamma#gamma} [GeV^{-1}];N events", maS));
        g->SetMarkerStyle(20); g->SetLineWidth(2); g->SetLineColor(kRed + 1);
        g->SetMarkerColor(kRed + 1);
        g->Draw("ALP");
        SaveCanvas(c, outDir + "/yield_vs_coupling");
        delete c;
    }

    // ── 4. Invariant mass reconstruction ────────────────────────────────────
    std::printf("\n=== 4. Reconstructed m_{#gamma#gamma} ===\n");
    std::printf("%8s %14s %12s %10s\n", "ma[MeV]", "mean m_gg", "rel. dev.", "verdict");
    auto* cM = new TCanvas("c_mgg", "", 800, 600);
    auto* legM = new TLegend(0.62, 0.62, 0.88, 0.88);
    int mi = 0;
    for (double ma : {masses.front(), masses[masses.size() / 2], masses.back()}) {
        const double g = PickSafeCoupling(ma, flux.energy_MeV, flux.rate_per_s);
        auto f = MakeFlux(geo, ma, g);
        f.Simulate(flux.energy_MeV, flux.rate_per_s, absXs);
        if (f.axionEnergy.empty()) continue;
        f.Propagate(false);
        const auto pairs = SimulateDecay4Vectors(f, geo, g, 20, rng);

        auto* h = new TH1D(Form("h_mgg_%d", mi), "", 100, 0, ma * 2);
        double s = 0, sw2 = 0;
        for (const auto& d : pairs) {
            const double E = d.g1.e + d.g2.e;
            const double px = d.g1.px + d.g2.px, py = d.g1.py + d.g2.py, pz = d.g1.pz + d.g2.pz;
            const double m = std::sqrt(std::max(E * E - px * px - py * py - pz * pz, 0.0));
            h->Fill(m); s += m; ++sw2;
        }
        const double mean = sw2 > 0 ? s / sw2 : 0.0;
        const double dev = std::abs(mean - ma) / ma;
        const bool ok = dev < 1e-3;
        if (!ok) ++failures;
        std::printf("%8.0f %14.4f %12.2e %10s\n", ma, mean, dev, ok ? "ok" : "OFF");

        h->SetTitle(";m_{#gamma#gamma} [MeV];pairs");
        h->SetLineColor(Palette(mi)); h->SetLineWidth(2); h->SetStats(0);
        cM->cd(); h->Draw(mi == 0 ? "HIST" : "HIST SAME");
        legM->AddEntry(h, Form("m_{a} = %.0f MeV", ma), "l");
        ++mi;
    }
    cM->cd(); cM->SetLogx(); legM->Draw();
    SaveCanvas(cM, outDir + "/invariant_mass");
    delete cM;

    std::printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED"
                                        : Form("%d CHECK(S) FAILED", failures));
    std::printf("Plots in %s/\n", outDir.c_str());
    return failures == 0 ? 0 : 1;
}
