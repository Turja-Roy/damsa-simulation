// Figures for the depth-aware calo reconstruction (Calo-Reco-Explained §2.13).
//
//   ./build/damsa_track_plots --dir output --tag pi0_E1000 \
//        [--grid "old summed=output/reco_grid_lesa_summed.csv,depth=output/reco_grid_lesa.csv"] \
//        [--out plots/tracks]
//   ./build/damsa_track_plots --dir output --templates gamma,e-,pi+,proton,neutron \
//        --energies 100,500,1000,2000 [--n-tmpl 2000] --out plots/templates
//
// From damsa_calo_tracks (tracks_<tag>.csv, track_pairs_<tag>.csv) -- every
// reconstructed object vs the photon-ID objects only (Calo-Reco-Explained §2.14):
//   angle_resid_<tag>        |reco - true| photon direction (unit-area histograms)
//   angle_vs_layers_<tag>    68 % angular error vs track length (layer clusters used)
//   energy_ratio_<tag>       E_reco / E_true
//   vertex_dz_<tag>          pair vertex z - true z (closest approach of the two photon lines)
// From the single-particle templates (tracks_tmpl_<p>_E<E>.csv, <p> without +/-):
//   tmpl_photon_rate         fraction of fired particles giving >= 1 photon-ID object, vs E
//   tmpl_<feature>           photon-ID feature per species (all objects, all energies)
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
#include <set>
#include <array>
#include <sstream>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

// Validated categorical slots 1-5 (light); 3-5 are under 3:1 contrast, so every
// series also gets its own marker and a legend.
const char* kCat[] = {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"};
const int kMark[] = {20, 21, 22, 23, 33};

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
    auto* leg = Legend(0.40, 0.88 - 0.05 * vals.size(), 0.94, 0.88);
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
    std::string dir = "output", tag, grid, out = "plots/tracks", tmpls, energies;
    int nTmpl = 2000;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--dir")  dir = nx();
        else if (s == "--tag")  tag = nx();
        else if (s == "--grid") grid = nx();
        else if (s == "--out")  out = nx();
        else if (s == "--templates") tmpls = nx();
        else if (s == "--energies")  energies = nx();
        else if (s == "--n-tmpl")    nTmpl = std::stoi(nx());
        else { std::fprintf(stderr, "usage: %s --dir DIR [--tag TAG] [--grid label=path,...] "
                                    "[--templates p,... --energies E,... --n-tmpl N] [--out DIR]\n", argv[0]); return 1; }
    }
    std::filesystem::create_directories(out);
    gStyle->SetOptStat(0);
    gStyle->SetTitleFont(42, "xyzt"); gStyle->SetLabelFont(42, "xyz");
    gStyle->SetTitleSize(0.045, "xyz"); gStyle->SetLabelSize(0.040, "xyz");
    gStyle->SetGridColor(TColor::GetColor("#d9d8d2")); gStyle->SetGridStyle(3);
    std::mt19937 rng(7);

    if (!tag.empty()) {
        // Same table twice: k = 0 every object, k = 1 photon ID only. Pairs are
        // built from photon-ID objects already, so the vertex plot has one series.
        const std::vector<std::string> labels = {"all objects", "photon ID"};
        const auto t0 = io::ReadCsv(dir + "/tracks_" + tag + ".csv");
        const std::vector<io::CsvTable> tr = {t0, t0};
        const auto pair = io::ReadCsv(dir + "/track_pairs_" + tag + ".csv");
        auto sel = [&](int k, std::size_t r) { return tr[k].get(r, "matched") > 0.5 && (k == 0 || tr[k].get(r, "photon") > 0.5); };
        std::vector<std::vector<double>> dth(2);
        for (int k = 0; k < 2; ++k)
            for (std::size_t r = 0; r < tr[k].size(); ++r)
                if (sel(k, r)) dth[k].push_back(tr[k].get(r, "dtheta_mrad"));

        // Angle residual.
        Overlay(dth, labels, "ang",
                Form("Photon direction error, %s;|#theta_{reco} - #theta_{true}| [mrad];fraction of matched photons", tag.c_str()),
                40, 0, 400, out + "/angle_resid_" + tag);

        // Energy ratio.
        std::vector<std::vector<double>> er(2);
        for (int k = 0; k < 2; ++k)
            for (std::size_t r = 0; r < tr[k].size(); ++r)
                if (sel(k, r) && tr[k].get(r, "true_E") > 0)
                    er[k].push_back(tr[k].get(r, "E") / tr[k].get(r, "true_E"));
        Overlay(er, labels, "eratio",
                Form("Reconstructed / true photon energy, %s;E_{reco} / E_{true};fraction of matched photons", tag.c_str()),
                40, 0, 2, out + "/energy_ratio_" + tag);

        // Vertex z residual.
        std::vector<std::vector<double>> dz(1);
        for (std::size_t r = 0; r < pair.size(); ++r)
            dz[0].push_back(pair.get(r, "vtx_z") - pair.get(r, "true_vz"));
        Overlay(dz, {"photon-ID pairs"}, "vdz",
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
                    if (sel(k, r) && n >= edges[b] && n < edges[b + 1]) v.push_back(tr[k].get(r, "dtheta_mrad"));
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

    // Single-particle templates: photon-ID pass rate vs energy, feature shapes per species.
    if (!tmpls.empty()) {
        auto split = [](const std::string& s) {
            std::vector<std::string> v; std::stringstream ss(s); std::string x;
            while (std::getline(ss, x, ',')) v.push_back(x);
            return v;
        };
        const auto ps = split(tmpls), es = split(energies);
        auto* c = Canvas("rate", true);
        c->SetLogx();
        auto* mg = new TMultiGraph();
        auto* leg = Legend(0.15, 0.15, 0.45, 0.42);
        const std::vector<std::array<const char*, 4>> feats = {   // column, title, lo, hi
            {"start", "shower start layer", "0", "44"}, {"tmaxPull", "shower-max pull (from start)", "-10", "10"},
            {"medLayerE", "median energy per lit layer [MeV]", "0", "60"}, {"meanBars", "mean bars per lit layer", "0", "6"},
            {"peakRatio", "peak / median layer energy", "0", "30"}, {"nLit", "lit layers", "0", "45"},
            {"widthLate", "lateral RMS, late layers [bars]", "0", "4"}};
        std::vector<std::vector<std::vector<double>>> fv(feats.size(), std::vector<std::vector<double>>(ps.size()));
        std::vector<std::string> names;
        for (std::size_t k = 0; k < ps.size(); ++k) {
            std::string p = ps[k];
            p.erase(std::remove_if(p.begin(), p.end(), [](char ch) { return ch == '+' || ch == '-'; }), p.end());
            names.push_back(ps[k]);
            auto* g = new TGraphErrors();
            for (const auto& e : es) {
                const std::string f = dir + "/tracks_tmpl_" + p + "_E" + e + ".csv";
                if (!std::filesystem::exists(f)) continue;
                const auto t = io::ReadCsv(f);
                std::set<long> ev;
                for (std::size_t r = 0; r < t.size(); ++r) {
                    if (t.get(r, "photon") > 0.5) ev.insert(static_cast<long>(t.get(r, "eventID")));
                    for (std::size_t j = 0; j < feats.size(); ++j) fv[j][k].push_back(t.get(r, feats[j][0]));
                }
                const double n = nTmpl, q = ev.size() / n;
                const int i = g->GetN();
                g->SetPoint(i, std::stod(e) * (1 + 0.03 * (k - 2.0)), std::max(q, 0.5 / n));   // small x offset per species
                g->SetPointError(i, 0, std::sqrt(std::max(q * (1 - q), 1.0 / n) / n));         // binomial
            }
            Style(g, g, static_cast<int>(k));
            mg->Add(g, "PL");
            leg->AddEntry(g, ps[k].c_str(), "lp");
        }
        mg->SetTitle(Form("Photon ID on single particles (%d fired per point);particle energy [MeV];"
                          "fraction with #geq 1 photon-ID object", nTmpl));
        mg->Draw("A");
        mg->SetMinimum(1e-4); mg->SetMaximum(2);
        mg->GetXaxis()->SetLimits(50, 4000);
        mg->GetYaxis()->SetTitleOffset(1.25);
        leg->Draw();
        c->SaveAs((out + "/tmpl_photon_rate.png").c_str());
        for (std::size_t j = 0; j < feats.size(); ++j)
            Overlay(fv[j], names, Form("f%zu", j), Form("%s, every object (all energies);%s;fraction of objects", feats[j][1], feats[j][1]),
                    44, std::stod(feats[j][2]), std::stod(feats[j][3]), out + "/tmpl_" + feats[j][0]);
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
            for (std::size_t k = 0; k < items.size() && k < 5; ++k) {
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
