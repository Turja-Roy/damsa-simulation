// Plots for the SM two-photon study (Random-2-photons plan §4.4, §5.7).
//
//   ./build/damsa_reco_plots [--dir output] [--out plots/reco]
//
// Reads reco_grid_{dark,lesa,xleap}.csv (damsa_calo_reco) and
// pileup_grid_exp.csv (damsa_pileup --expected); writes PNGs:
//   photon_eff_vs_mu, photon_purity_vs_mu   calo saturation vs occupancy
//   fake_frac_vs_mass_<mode>                fake windows vs mass, per f
//   fake_classes_vs_mu                      what the fakes are, vs occupancy
//   truth_vs_reco_rate_lesa                 truth-level vs reconstructed fake rate
//   fake_frac_vs_window_lesa                window and sigma_t dependence
//   truth_rate_vs_f                         truth fake rate vs f (expect f^2)
//
// Colour: categorical slots (mode, class) and a 4-step blue ramp (f, sigma_t),
// both run through the dataviz validator. Every multi-series plot has a legend
// and a distinct marker per series, since three categorical hues sit below 3:1
// contrast on white.

#include "damsa_io.h"

#include <TAxis.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TMultiGraph.h>
#include <TStyle.h>
#include <TF1.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

// Validated: categorical slots 1-5 and the 4-step ordinal ramp (light surface).
const char* kCat[] = {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"};
const char* kRamp[] = {"#86b6ef", "#5598e7", "#256abf", "#104281"};
const int kMark[] = {20, 21, 22, 23, 33};   // secondary encoding, per series slot

const char* kModes[] = {"dark", "lesa", "xleap"};
const char* kModeLabel[] = {"dark current", "LESA laser", "X-LEAP"};

bool Eq(double a, double b) { return std::abs(a - b) <= 1e-6 * std::max(1.0, std::abs(b)); }

using Pred = std::function<bool(const io::CsvRowView&)>;

// (x, y) of the rows passing `keep`, sorted by x; rows with y <= 0 dropped when
// the plot is log-y (they cannot be drawn).
TGraph* Graph(const io::CsvTable& t, const Pred& keep, const std::function<double(const io::CsvRowView&)>& x,
              const std::function<double(const io::CsvRowView&)>& y, const char* hex, int marker, bool logy)
{
    std::vector<std::pair<double, double>> pts;
    for (const auto& r : t.views())
        if (keep(r)) {
            const double yy = y(r);
            if (!std::isfinite(yy) || (logy && yy <= 0)) continue;
            pts.push_back({x(r), yy});
        }
    std::sort(pts.begin(), pts.end());
    auto* g = new TGraph();
    for (const auto& [a, b] : pts) g->AddPoint(a, b);
    const int col = TColor::GetColor(hex);
    g->SetLineColor(col);
    g->SetMarkerColor(col);
    g->SetLineWidth(2);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.2);
    return g;
}

TCanvas* Canvas(const char* name, bool logx, bool logy)
{
    auto* c = new TCanvas(name, "", 900, 620);
    c->SetLeftMargin(0.13);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.13);
    c->SetTopMargin(0.11);   // room for a "x10^-n" axis exponent under the title
    c->SetLogx(logx);
    c->SetLogy(logy);
    c->SetGrid();
    return c;
}

TLegend* Legend(double x1 = 0.62, double y1 = 0.66, double x2 = 0.94, double y2 = 0.89)
{
    auto* l = new TLegend(x1, y1, x2, y2);
    l->SetBorderSize(0);
    l->SetFillStyle(0);
    l->SetTextFont(42);
    l->SetTextSize(0.034);
    return l;
}

void Draw(TCanvas* c, TMultiGraph* mg, TLegend* leg, const char* title, const std::string& path)
{
    c->cd();
    mg->SetTitle(title);
    mg->Draw("ALP");
    mg->GetXaxis()->SetTitleOffset(1.1);
    mg->GetYaxis()->SetTitleOffset(1.25);
    if (leg) leg->Draw();
    c->SaveAs((path + ".png").c_str());
}

// Reco rows that carry one value per (mode, f, window, sigma_t): mass 100, all E_pair, no z cut.
bool CellRow(const io::CsvRowView& r)
{
    return Eq(r.get("mass_MeV"), 100) && std::isinf(r.get("epair_hi_MeV")) && Eq(r.get("zcut"), 0);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string dir = "output", out = "plots/reco";
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (s == "--out" && i + 1 < argc) out = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--dir output] [--out plots/reco]\n", argv[0]); return 1; }
    }
    std::filesystem::create_directories(out);
    gStyle->SetOptStat(0);
    gStyle->SetTitleFont(42, "xyzt");
    gStyle->SetLabelFont(42, "xyz");
    gStyle->SetTitleSize(0.045, "xyz");
    gStyle->SetLabelSize(0.040, "xyz");
    gStyle->SetGridColor(TColor::GetColor("#d9d8d2"));   // recessive grid
    gStyle->SetGridStyle(3);

    std::vector<io::CsvTable> reco;
    for (const char* m : kModes) reco.push_back(io::ReadCsv(dir + "/reco_grid_" + m + ".csv"));
    auto num = [](const char* col) { return [col](const io::CsvRowView& r) { return r.get(col); }; };

    // ── 1. photon efficiency / purity vs mu ────────────────────────────────
    for (const char* what : {"photon_eff", "photon_purity"}) {
        auto* c = Canvas(what, true, false);
        auto* mg = new TMultiGraph();
        // Efficiency data sits low on the right, purity data high: legend goes where the data is not.
        auto* leg = std::string(what) == "photon_eff" ? Legend(0.66, 0.70, 0.94, 0.89) : Legend(0.66, 0.18, 0.94, 0.37);
        for (int m = 0; m < 3; ++m) {
            // Fewer than 10 truth (efficiency) / reco (purity) photons is noise: skip.
            const char* den = std::string(what) == "photon_eff" ? "truth_photons" : "reco_photons";
            auto* g = Graph(reco[m], [den](const io::CsvRowView& r) {
                                return CellRow(r) && Eq(r.get("sigma_t_ns"), 1) && r.get(den) >= 10; },
                            num("mu"), num(what), kCat[m], kMark[m], false);
            g->SetLineWidth(0);   // scatter: points from many (f, window) cells
            mg->Add(g, "P");
            leg->AddEntry(g, kModeLabel[m], "p");
        }
        const std::string y = std::string(what) == "photon_eff" ? "photon efficiency (truth #geq 10 MeV)"
                                                                : "photon purity (matched / reconstructed)";
        Draw(c, mg, leg, (std::string(what == std::string("photon_eff") ? "Photon reconstruction efficiency"
                                                                        : "Photon reconstruction purity")
                          + ";#mu = electrons per window;" + y).c_str(), out + "/" + what + "_vs_mu");
    }

    // ── 2. fake-window fraction vs mass, per f (W = 10 ns, sigma_t = 1 ns) ──
    const double fShow[] = {0.1, 0.2, 0.5, 1.0};
    for (int m = 0; m < 3; ++m) {
        auto* c = Canvas(Form("ff_%d", m), false, false);
        auto* mg = new TMultiGraph();
        auto* leg = Legend(0.70, 0.66, 0.94, 0.89);
        for (int k = 0; k < 4; ++k) {
            const double f = fShow[k];
            auto* g = Graph(reco[m], [f](const io::CsvRowView& r) {
                                return std::isinf(r.get("epair_hi_MeV")) && Eq(r.get("zcut"), 0) &&
                                       Eq(r.get("sigma_t_ns"), 1) && Eq(r.get("window_ns"), 10) && Eq(r.get("bunch_frac"), f); },
                            num("mass_MeV"), [](const io::CsvRowView& r) { return r.get("FP") / r.get("windows"); },
                            kRamp[k], kMark[k], false);
            mg->Add(g, "LP");
            leg->AddEntry(g, Form("f = %g", f), "lp");
        }
        Draw(c, mg, leg, Form("Fake 2#gamma windows, %s (10 ns window, #sigma_{t} = 1 ns);"
                              "m_{#gamma#gamma} window centre [MeV] (#pm 20 MeV);fraction of windows with a fake",
                              kModeLabel[m]),
             out + "/fake_frac_vs_mass_" + kModes[m]);
    }

    // ── 3. fake classes vs mu (m = 20 MeV, W = 10 ns, sigma_t = 1 ns) ──────
    {
        const char* cls[] = {"cand_FP_acc", "cand_FP_shower", "cand_FP_split", "cand_FP_ghost", "cand_FP_unm"};
        const char* lab[] = {"accidental", "same shower", "split", "ghost (x/y)", "unmatched"};
        auto* c = Canvas("cls", true, false);
        auto* mg = new TMultiGraph();
        auto* leg = Legend(0.40, 0.35, 0.66, 0.62);
        auto keep = [](const io::CsvRowView& r) {
            return Eq(r.get("mass_MeV"), 20) && std::isinf(r.get("epair_hi_MeV")) && Eq(r.get("zcut"), 0) &&
                   Eq(r.get("sigma_t_ns"), 1) && Eq(r.get("window_ns"), 10);
        };
        for (int k = 0; k < 5; ++k) {
            const char* col = cls[k];
            // Fractions from fewer than 20 candidates are noise (1 of 1 = 100 %): skip.
            auto frac = [&cls, col](const io::CsvRowView& r) {
                double tot = 0;
                for (const char* x : cls) tot += r.get(x);
                return tot >= 20 ? r.get(col) / tot : std::nan("");
            };
            auto* g = new TGraph();
            for (int m = 0; m < 3; ++m) {
                auto* part = Graph(reco[m], keep, num("mu"), frac, kCat[k], kMark[k], false);
                for (int i = 0; i < part->GetN(); ++i) g->AddPoint(part->GetX()[i], part->GetY()[i]);
                delete part;
            }
            g->Sort();
            const int colr = TColor::GetColor(kCat[k]);
            g->SetLineColor(colr); g->SetMarkerColor(colr); g->SetLineWidth(2);
            g->SetMarkerStyle(kMark[k]); g->SetMarkerSize(1.2);
            mg->Add(g, "P");   // points: the mu axis has a gap, lines would bridge it
            leg->AddEntry(g, lab[k], "p");
        }
        Draw(c, mg, leg, "Fake-candidate composition, m_{#gamma#gamma} = 20#pm20 MeV, 10 ns (#geq 20 per point);"
                         "#mu = electrons per window;fraction of fake candidates", out + "/fake_classes_vs_mu");
    }

    // ── 4/6. truth-level rates (pileup --expected) ──────────────────────────
    const std::string expPath = dir + "/pileup_grid_exp.csv";
    if (std::filesystem::exists(expPath)) {
        const auto e = io::ReadCsv(expPath);
        // beam_mode is a string column (reads 0); recover it from the bunch charge q = mu / (f <nb>).
        auto modeOf = [](const io::CsvRowView& r) {
            const double q = r.get("mu") / (r.get("bunch_frac") * r.get("mean_bunches"));
            return q < 1 ? 0 : q < 1e5 ? 1 : 2;
        };

        // 4. truth vs reco fake rate, LESA f = 0.1, W = 10 ns
        auto* c = Canvas("tvr", false, true);
        auto* mg = new TMultiGraph();
        auto* leg = Legend(0.58, 0.72, 0.94, 0.89);
        auto* gt = Graph(e, [&](const io::CsvRowView& r) {
                             return modeOf(r) == 1 && Eq(r.get("bunch_frac"), 0.1) && Eq(r.get("window_ns"), 10); },
                         num("mass_MeV"), num("R_acc_Hz"), kCat[0], kMark[0], true);   // windows with >=1 fake, like FP_rate_Hz
        auto* gr = Graph(reco[1], [](const io::CsvRowView& r) {
                             return std::isinf(r.get("epair_hi_MeV")) && Eq(r.get("zcut"), 0) && Eq(r.get("sigma_t_ns"), 1) &&
                                    Eq(r.get("window_ns"), 10) && Eq(r.get("bunch_frac"), 0.1); },
                         num("mass_MeV"), num("FP_rate_Hz"), kCat[1], kMark[1], true);
        mg->Add(gt, "LP"); leg->AddEntry(gt, "truth (calo-face photons)", "lp");
        mg->Add(gr, "LP"); leg->AddEntry(gr, "reconstructed (calo)", "lp");
        Draw(c, mg, leg, "Windows with a fake 2#gamma, LESA laser f = 0.1, 10 ns window;"
                         "m_{#gamma#gamma} window centre [MeV] (#pm 20 MeV);rate of windows with #geq 1 fake [Hz]", out + "/truth_vs_reco_rate_lesa");

        // 6. truth fake rate vs f, m = 100 MeV, W = 10 ns: expect slope 2 (rate ~ mu^2)
        auto* c6 = Canvas("tf", true, true);
        auto* mg6 = new TMultiGraph();
        auto* leg6 = Legend(0.15, 0.70, 0.45, 0.89);
        for (int m = 0; m < 3; ++m) {
            auto* g = Graph(e, [&, m](const io::CsvRowView& r) {
                                return modeOf(r) == m && Eq(r.get("mass_MeV"), 100) && Eq(r.get("window_ns"), 10); },
                            num("bunch_frac"), num("fake_pair_rate_Hz"), kCat[m], kMark[m], true);
            if (g->GetN() == 0) continue;
            mg6->Add(g, "LP");
            leg6->AddEntry(g, kModeLabel[m], "lp");
        }
        Draw(c6, mg6, leg6, "Truth-level fake 2#gamma rate vs beam intensity, m_{#gamma#gamma} = 100#pm20 MeV, 10 ns;"
                            "bunch-charge fraction f;rate [Hz]", out + "/truth_rate_vs_f");
    } else {
        std::fprintf(stderr, "[skip] %s not found: truth-level plots skipped\n", expPath.c_str());
    }

    // ── 5. fake fraction vs window, per sigma_t (LESA f = 0.1, m = 20 MeV) ──
    {
        const double sShow[] = {1, 10, 50, 100};
        auto* c = Canvas("win", false, false);
        auto* mg = new TMultiGraph();
        auto* leg = Legend(0.16, 0.15, 0.42, 0.38);
        for (int k = 0; k < 4; ++k) {
            const double st = sShow[k];
            auto* g = Graph(reco[1], [st](const io::CsvRowView& r) {
                                return Eq(r.get("mass_MeV"), 20) && std::isinf(r.get("epair_hi_MeV")) && Eq(r.get("zcut"), 0) &&
                                       Eq(r.get("bunch_frac"), 0.1) && Eq(r.get("sigma_t_ns"), st); },
                            num("window_ns"), [](const io::CsvRowView& r) { return r.get("FP") / r.get("windows"); },
                            kRamp[k], kMark[k], false);
            mg->Add(g, "LP");
            leg->AddEntry(g, Form("#sigma_{t} = %g ns", st), "lp");
        }
        Draw(c, mg, leg, "Fake windows vs coincidence window, LESA laser f = 0.1 (m = 20 #pm 20 MeV);"
                         "coincidence window [ns];fraction of windows with a fake", out + "/fake_frac_vs_window_lesa");
    }

    std::printf("plots written to %s/\n", out.c_str());
    return 0;
}
