// Figures for the depth-aware calo reconstruction (Calo-Reco-Explained §2.13).
//
//   ./build/damsa_track_plots --dir output --tag pi0_E1000 \
//        [--grid "old summed=output/reco_grid_lesa_summed.csv,depth+energy=output/reco_grid_lesa.csv,depth boolean=output/reco_grid_lesa_noE.csv"] \
//        [--out plots/tracks]
//
// From damsa_calo_tracks (tracks_<tag>.csv vs tracks_<tag>_noE.csv, and the
// track_pairs_ files) -- energy-integrated vs boolean readout:
//   angle_resid_<tag>        |reco - true| photon direction (unit-area histograms)
//   angle_vs_layers_<tag>    68 % angular error vs track length (layer clusters used)
//   energy_ratio_<tag>       E_reco / E_true
//   vertex_dz_<tag>          pair vertex z - true z (closest approach of the two photon lines)
// From damsa_calo_reco grids (--grid label=path,...), beam background vs mu:
//   grid_fake20_vs_mu, grid_eff_vs_mu
// Error bars: Poisson per histogram bin, binomial on fractions, bootstrap
// (200 resamples) on quantiles.

#include "damsa_io.h"

#include <TAxis.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TMultiGraph.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

const char* kCat[] = {"#2a78d6", "#eb6834", "#1baf7a"};   // validated categorical slots 1-3
const int kMark[] = {20, 21, 22};

TCanvas* Canvas(const char* name, bool logy = false)
{
    auto* c = new TCanvas(name, "", 900, 620);
    c->SetLeftMargin(0.13); c->SetRightMargin(0.04); c->SetBottomMargin(0.13); c->SetTopMargin(0.11);
    c->SetLogy(logy);
    c->SetGrid();
    return c;
}

TLegend* Legend(double x1, double y1, double x2, double y2)
{
    auto* l = new TLegend(x1, y1, x2, y2);
    l->SetBorderSize(0); l->SetFillStyle(0); l->SetTextFont(42); l->SetTextSize(0.034);
    return l;
}

void Style(TAttLine* l, TAttMarker* m, int k)
{
    const int col = TColor::GetColor(kCat[k]);
    l->SetLineColor(col); l->SetLineWidth(2);
    m->SetMarkerColor(col); m->SetMarkerStyle(kMark[k]); m->SetMarkerSize(1.1);
}

std::vector<double> Column(const io::CsvTable& t, const std::string& col, const std::string& sel = "matched")
{
    std::vector<double> v;
    for (std::size_t r = 0; r < t.size(); ++r)
        if (sel.empty() || t.get(r, sel) > 0.5) v.push_back(t.get(r, col));
    return v;
}

double Quantile(std::vector<double> v, double q)
{
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))];
}

// Bootstrap standard error of a quantile.
double QuantileErr(const std::vector<double>& v, double q, std::mt19937& rng)
{
    if (v.size() < 5) return 0;
    std::uniform_int_distribution<std::size_t> pick(0, v.size() - 1);
    std::vector<double> qs;
    for (int b = 0; b < 200; ++b) {
        std::vector<double> s(v.size());
        for (auto& x : s) x = v[pick(rng)];
        qs.push_back(Quantile(s, q));
    }
    double m = 0, m2 = 0;
    for (double x : qs) { m += x; m2 += x * x; }
    m /= qs.size();
    return std::sqrt(std::max(0.0, m2 / qs.size() - m * m));
}

// Unit-area histogram with Poisson errors scaled alike.
TH1D* UnitHist(const std::vector<double>& v, const char* name, int nb, double lo, double hi, int k)
{
    auto* h = new TH1D(name, "", nb, lo, hi);
    h->Sumw2();
    for (double x : v) h->Fill(std::clamp(x, lo + 1e-9, hi - 1e-9));   // overflow into the edge bins, visibly
    if (h->Integral() > 0) h->Scale(1.0 / h->Integral());
    Style(h, h, k);
    h->SetStats(0);
    return h;
}

void Overlay(const std::vector<std::vector<double>>& vals, const std::vector<std::string>& labels,
             const char* name, const char* title, int nb, double lo, double hi, const std::string& path)
{
    auto* c = Canvas(name);
    auto* leg = Legend(0.40, 0.74, 0.94, 0.88);
    leg->SetTextSize(0.030);
    double ymax = 0;
    std::vector<TH1D*> hs;
    for (std::size_t k = 0; k < vals.size(); ++k) {
        hs.push_back(UnitHist(vals[k], Form("%s_%zu", name, k), nb, lo, hi, static_cast<int>(k)));
        ymax = std::max(ymax, hs.back()->GetMaximum());
        leg->AddEntry(hs.back(), Form("%s (n = %zu, median %.3g)", labels[k].c_str(), vals[k].size(),
                                      Quantile(vals[k], 0.5)), "lep");
    }
    for (std::size_t k = 0; k < hs.size(); ++k) {
        hs[k]->SetTitle(title);
        hs[k]->SetMaximum(1.25 * ymax);
        hs[k]->GetYaxis()->SetTitleOffset(1.25);
        hs[k]->Draw(k ? "E1 SAME" : "E1");
    }
    leg->Draw();
    c->SaveAs((path + ".png").c_str());
}

}  // namespace

int main(int argc, char** argv)
{
    std::string dir = "output", tag, grid, out = "plots/tracks";
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--dir")  dir = nx();
        else if (s == "--tag")  tag = nx();
        else if (s == "--grid") grid = nx();
        else if (s == "--out")  out = nx();
        else { std::fprintf(stderr, "usage: %s --dir DIR --tag TAG [--grid label=path,...] [--out DIR]\n", argv[0]); return 1; }
    }
    std::filesystem::create_directories(out);
    gStyle->SetOptStat(0);
    gStyle->SetTitleFont(42, "xyzt"); gStyle->SetLabelFont(42, "xyz");
    gStyle->SetTitleSize(0.045, "xyz"); gStyle->SetLabelSize(0.040, "xyz");
    gStyle->SetGridColor(TColor::GetColor("#d9d8d2")); gStyle->SetGridStyle(3);
    std::mt19937 rng(7);

    if (!tag.empty()) {
        const std::vector<std::string> labels = {"energy-integrated", "boolean readout"};
        std::vector<io::CsvTable> tr, pr;
        for (const std::string sfx : {"", "_noE"}) {
            tr.push_back(io::ReadCsv(dir + "/tracks_" + tag + sfx + ".csv"));
            pr.push_back(io::ReadCsv(dir + "/track_pairs_" + tag + sfx + ".csv"));
        }

        // Angle residual.
        Overlay({Column(tr[0], "dtheta_mrad"), Column(tr[1], "dtheta_mrad")}, labels, "ang",
                Form("Photon direction error, %s;|#theta_{reco} - #theta_{true}| [mrad];fraction of matched photons", tag.c_str()),
                40, 0, 400, out + "/angle_resid_" + tag);

        // Energy ratio.
        std::vector<std::vector<double>> er(2);
        for (int k = 0; k < 2; ++k)
            for (std::size_t r = 0; r < tr[k].size(); ++r)
                if (tr[k].get(r, "matched") > 0.5 && tr[k].get(r, "true_E") > 0)
                    er[k].push_back(tr[k].get(r, "E") / tr[k].get(r, "true_E"));
        Overlay(er, labels, "eratio",
                Form("Reconstructed / true photon energy, %s;E_{reco} / E_{true};fraction of matched photons", tag.c_str()),
                40, 0, 2, out + "/energy_ratio_" + tag);

        // Vertex z residual.
        std::vector<std::vector<double>> dz(2);
        for (int k = 0; k < 2; ++k)
            for (std::size_t r = 0; r < pr[k].size(); ++r)
                dz[k].push_back(pr[k].get(r, "vtx_z") - pr[k].get(r, "true_vz"));
        Overlay(dz, labels, "vdz",
                Form("Pair vertex from two photon lines, %s;z_{vertex,reco} - z_{vertex,true} [mm];fraction of reconstructed pairs", tag.c_str()),
                40, -1000, 1000, out + "/vertex_dz_" + tag);

        // 68 % angular error vs track length.
        auto* c = Canvas("avl");
        auto* mg = new TMultiGraph();
        auto* leg = Legend(0.58, 0.74, 0.94, 0.88);
        const int edges[] = {1, 3, 6, 10, 15, 20, 30, 45, 89};
        for (int k = 0; k < 2; ++k) {
            auto* g = new TGraphErrors();
            for (int b = 0; b + 1 < 9; ++b) {
                std::vector<double> v;
                for (std::size_t r = 0; r < tr[k].size(); ++r) {
                    const double n = tr[k].get(r, "nLayers");
                    if (tr[k].get(r, "matched") > 0.5 && n >= edges[b] && n < edges[b + 1]) v.push_back(tr[k].get(r, "dtheta_mrad"));
                }
                if (v.size() < 20) continue;
                const int i = g->GetN();
                g->SetPoint(i, 0.5 * (edges[b] + edges[b + 1] - 1), Quantile(v, 0.68));
                g->SetPointError(i, 0.5 * (edges[b + 1] - edges[b]), QuantileErr(v, 0.68, rng));
            }
            Style(g, g, k);
            mg->Add(g, "PL");
            leg->AddEntry(g, labels[k].c_str(), "lp");
        }
        mg->SetTitle(Form("Longer track, better direction (%s, #geq 20 photons per point);layer clusters in the photon (x + y views);"
                          "68%% angular error [mrad]", tag.c_str()));
        mg->Draw("A");
        mg->GetYaxis()->SetTitleOffset(1.25);
        leg->Draw();
        c->SaveAs((out + "/angle_vs_layers_" + tag + ".png").c_str());
    }

    // Grid comparison on the beam background.
    if (!grid.empty()) {
        std::vector<std::pair<std::string, std::string>> items;
        std::stringstream ss(grid);
        std::string it;
        while (std::getline(ss, it, ','))
            if (const auto eq = it.find('='); eq != std::string::npos) items.push_back({it.substr(0, eq), it.substr(eq + 1)});
        for (const char* what : {"fake20", "eff"}) {
            auto* c = Canvas(what);
            c->SetLogx();
            auto* mg = new TMultiGraph();
            auto* leg = Legend(what == std::string("eff") ? 0.58 : 0.15, 0.70, what == std::string("eff") ? 0.94 : 0.50, 0.88);
            for (std::size_t k = 0; k < items.size() && k < 3; ++k) {
                const auto t = io::ReadCsv(items[k].second);
                std::vector<std::array<double, 3>> pts;   // mu, value, error
                for (std::size_t r = 0; r < t.size(); ++r) {
                    if (std::abs(t.get(r, "mass_MeV") - 20) > 1e-6 || !std::isinf(t.get(r, "epair_hi_MeV")) ||
                        t.get(r, "zcut") != 0 || std::abs(t.get(r, "sigma_t_ns") - 1) > 1e-6) continue;
                    double p, n;
                    if (what == std::string("fake20")) { n = t.get(r, "windows"); p = t.get(r, "FP") / n; }
                    else { n = t.get(r, "truth_photons"); p = t.get(r, "photon_eff"); }
                    if (n < 10) continue;
                    pts.push_back({t.get(r, "mu"), p, std::sqrt(std::max(p * (1 - p), 1.0 / n) / n)});   // binomial
                }
                std::sort(pts.begin(), pts.end());
                auto* g = new TGraphErrors();
                for (const auto& [m, v, e] : pts) { const int i = g->GetN(); g->SetPoint(i, m, v); g->SetPointError(i, 0, e); }
                Style(g, g, static_cast<int>(k));
                mg->Add(g, "PL");
                leg->AddEntry(g, items[k].first.c_str(), "lp");
            }
            mg->SetTitle(what == std::string("fake20")
                ? "Beam background: windows with a fake 2#gamma (m = 20#pm20 MeV, 10 ns);#mu = electrons per window;fraction of windows"
                : "Beam background: photon efficiency (truth #geq 10 MeV);#mu = electrons per window;efficiency");
            mg->Draw("A");
            mg->GetXaxis()->SetLimits(0.05, 2e4);   // keep the dark-current point (mu ~ 0.1) on the axis
            mg->GetYaxis()->SetTitleOffset(1.25);
            leg->Draw();
            c->SaveAs((out + "/grid_" + what + "_vs_mu.png").c_str());
        }
    }
    std::printf("plots written to %s/\n", out.c_str());
    return 0;
}
