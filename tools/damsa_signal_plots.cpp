// Signal and spectrum plots for the ALP analysis.
// Replaces scripts/visualization/alplib_signal_plots.py and root_config_plots.py.
//
//   ./build/damsa_signal_plots --decay 'output/alp_decay_photons_ma*MeV.root' \
//        --flux output/alplib_brems_flux.csv --out-dir plots/signal
//   ./build/damsa_signal_plots --compare output/Tz10_calo_face_particles.root \
//        --compare output/Tz14_calo_face_particles.root --out-dir plots/compare
//
// Everything is written as PNG and collected into one .root file, so the
// histograms stay inspectable in a TBrowser rather than only as images.

#include "damsa_io.h"
#include "plotting.h"

#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TMultiGraph.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

// "alp_decay_photons_ma100MeV.root" -> 100
double MassFromName(const std::string& p)
{
    static const std::regex re(R"(ma(\d+(?:\.\d+)?)MeV)");
    std::smatch m;
    return std::regex_search(p, m, re) ? std::stod(m[1]) : -1.0;
}

std::string Stem(const std::string& p)
{
    return std::filesystem::path(p).stem().string();
}

}  // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> decayFiles, compareFiles;
    std::string fluxPath, outDir = "plots/signal";

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--decay")   decayFiles.push_back(nx());
        else if (s == "--compare") compareFiles.push_back(nx());
        else if (s == "--flux")    fluxPath = nx();
        else if (s == "--out-dir") outDir = nx();
        else {
            std::fprintf(stderr,
                "Usage: %s [--decay FILE]... [--compare FILE]... [--flux CSV] [--out-dir DIR]\n"
                "  --decay    alp_decay_photons_ma*MeV.root (repeatable)\n"
                "  --compare  calo-face particle files to overlay (repeatable)\n", argv[0]);
            return 1;
        }
    }
    if (decayFiles.empty() && compareFiles.empty() && fluxPath.empty()) {
        std::fprintf(stderr, "Nothing to plot: pass --decay, --compare or --flux\n");
        return 1;
    }

    std::filesystem::create_directories(outDir);
    SetPublicationStyle();
    auto* rootOut = TFile::Open((outDir + "/signal_plots.root").c_str(), "RECREATE");

    // ── Bremsstrahlung flux ─────────────────────────────────────────────────
    if (!fluxPath.empty()) {
        const auto f = io::ReadBremsFlux(fluxPath);
        auto* g = new TGraph(f.size(), f.energy_MeV.data(), f.rate_per_s.data());
        g->SetName("g_brems_flux");
        g->SetTitle(Form("Bremsstrahlung flux in target (%s);E_{#gamma} [MeV];"
                         "rate [photons/s]",
                         f.beamMode.empty() ? "beam mode unknown" : f.beamMode.c_str()));
        g->SetLineColor(kAzure + 2); g->SetLineWidth(2);
        auto* c = new TCanvas("c_flux", "", 800, 600);
        c->SetLogx(); c->SetLogy();
        g->Draw("AL");
        SaveCanvas(c, outDir + "/brems_flux");
        rootOut->cd(); g->Write();
        std::printf("[flux] %zu bins, total %.3e photons/s\n", f.size(), f.totalRate());
        delete c;
    }

    // ── ALP decay photons, per mass ─────────────────────────────────────────
    if (!decayFiles.empty()) {
        std::sort(decayFiles.begin(), decayFiles.end(),
                  [](const std::string& a, const std::string& b) {
                      return MassFromName(a) < MassFromName(b);
                  });

        auto* cE = new TCanvas("c_e", "", 800, 600);
        auto* cA = new TCanvas("c_a", "", 800, 600);
        auto* legE = new TLegend(0.62, 0.62, 0.88, 0.88);
        auto* legA = new TLegend(0.62, 0.62, 0.88, 0.88);
        int idx = 0;

        for (const auto& path : decayFiles) {
            const double ma = MassFromName(path);
            const auto rows = io::ReadNTuple<io::AlpDecayRow>(path);
            if (rows.empty()) { std::fprintf(stderr, "  %s: empty\n", path.c_str()); continue; }

            auto* hE = new TH1D(Form("h_E_ma%.0f", ma), "", 100, 0, 8000);
            auto* hA = new TH1D(Form("h_theta_ma%.0f", ma), "", 90, 0, 180);
            auto* hZ = new TH1D(Form("h_z_ma%.0f", ma), "", 100, 0, 0.6);
            double sumW = 0;

            for (const auto& r : rows) {
                if (r.weight <= 0) continue;
                // Both photons enter the energy spectrum; the pair contributes
                // one opening angle.
                hE->Fill(r.E1, r.weight);
                hE->Fill(r.E2, r.weight);
                const double m1 = std::sqrt(r.px1*r.px1 + r.py1*r.py1 + r.pz1*r.pz1);
                const double m2 = std::sqrt(r.px2*r.px2 + r.py2*r.py2 + r.pz2*r.pz2);
                if (m1 > 1e-12 && m2 > 1e-12) {
                    double c = (r.px1*r.px2 + r.py1*r.py2 + r.pz1*r.pz2) / (m1 * m2);
                    c = std::clamp(c, -1.0, 1.0);
                    hA->Fill(std::acos(c) * 180.0 / M_PI, r.weight);
                }
                hZ->Fill(r.decayZ_m, r.weight);
                sumW += r.weight;
            }

            hE->SetTitle(";E_{#gamma} [MeV];weighted photons");
            hA->SetTitle(";#theta_{#gamma#gamma} [deg];weighted pairs");
            hZ->SetTitle(Form("ALP decay vertex, m_{a}=%.0f MeV;z from target centre [m];"
                              "weighted decays", ma));
            for (auto* h : {hE, hA, hZ}) { h->SetLineColor(Palette(idx)); h->SetLineWidth(2); h->SetStats(0); }

            cE->cd(); hE->Draw(idx == 0 ? "HIST" : "HIST SAME");
            cA->cd(); hA->Draw(idx == 0 ? "HIST" : "HIST SAME");
            legE->AddEntry(hE, Form("m_{a} = %.0f MeV", ma), "l");
            legA->AddEntry(hA, Form("m_{a} = %.0f MeV", ma), "l");

            auto* cZ = new TCanvas(Form("c_z_%.0f", ma), "", 800, 600);
            hZ->Draw("HIST");
            SaveCanvas(cZ, Form("%s/decay_vertex_ma%.0fMeV", outDir.c_str(), ma));
            delete cZ;

            rootOut->cd(); hE->Write(); hA->Write(); hZ->Write();
            std::printf("[decay] ma=%6.0f MeV: %zu pairs, total weight %.3e events\n",
                        ma, rows.size(), sumW);
            ++idx;
        }

        cE->cd(); cE->SetLogy(); legE->Draw();
        SaveCanvas(cE, outDir + "/decay_photon_energy");
        // Log-y: total weight spans orders of magnitude between masses, so a
        // linear axis hides every curve but the lightest.
        cA->cd(); cA->SetLogy(); legA->Draw();
        SaveCanvas(cA, outDir + "/opening_angle");
        delete cE; delete cA;
    }

    // ── Configuration comparison at the calo face ───────────────────────────
    if (!compareFiles.empty()) {
        auto* cE = new TCanvas("c_cmp_e", "", 800, 600);
        auto* leg = new TLegend(0.58, 0.62, 0.88, 0.88);
        int idx = 0;
        std::printf("\n%-34s %10s %10s %14s\n", "configuration", "photons", "neutrons",
                    "sum E [MeV]");
        std::printf("%s\n", std::string(72, '-').c_str());

        for (const auto& path : compareFiles) {
            const auto rows = io::ReadNTuple<io::ParticleRow>(path);
            const std::string name = Stem(path);
            auto* h = new TH1D(Form("h_cmp_%d", idx), "", 100, 0, 1000);
            long nPh = 0, nNe = 0; double sumE = 0;
            for (const auto& p : rows) {
                if (p.pdg == 22)        ++nPh;
                else if (p.pdg == 2112) ++nNe;
                else continue;
                h->Fill(p.energy_MeV, p.weight);
                sumE += p.energy_MeV * p.weight;
            }
            h->SetTitle(";E [MeV];weighted particles at calo face");
            h->SetLineColor(Palette(idx)); h->SetLineWidth(2); h->SetStats(0);
            cE->cd(); h->Draw(idx == 0 ? "HIST" : "HIST SAME");
            leg->AddEntry(h, name.c_str(), "l");
            rootOut->cd(); h->Write();
            std::printf("%-34s %10ld %10ld %14.4e\n", name.c_str(), nPh, nNe, sumE);
            ++idx;
        }
        cE->cd(); cE->SetLogy(); leg->Draw();
        SaveCanvas(cE, outDir + "/config_comparison_energy");
        delete cE;
    }

    rootOut->Close();
    std::printf("\nWrote %s/ (PNGs + signal_plots.root)\n", outDir.c_str());
    return 0;
}
