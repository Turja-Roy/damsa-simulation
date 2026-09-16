// Final configuration report: rank the scan results, pick the optimal geometry,
// and produce the presentation plots.
// Replaces scripts/visualization/final_report.py.
//
//   ./build/damsa_report --pareto output/joint_pareto/Tz10_pareto.csv \
//        --grid output/joint_pareto/Tz10_grid.csv \
//        --proj output/target_projection/target_projection_grid.csv \
//        --flux output/alplib_brems_flux.csv
//
// Outputs: optimal_config.csv, global_3d_pareto.csv, sensitivity_*.csv and plots.

#include "damsa_io.h"
#include "plotting.h"
#include "scan.h"

#include <TCanvas.h>
#include <TGraph.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TMultiGraph.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <set>

using namespace damsa::alp;
namespace io = damsa::io;

namespace {

constexpr double kMagnet_cm = 12.0;

struct Row {
    double ma, vdc_cm, calo_cm, bkg, acc, sep, fom, target_cm = 0;
};

std::vector<Row> LoadRows(const std::string& path, bool& ok)
{
    std::vector<Row> rows;
    ok = false;
    if (!std::filesystem::exists(path)) return rows;
    const auto csv = io::ReadCsv(path);
    for (std::size_t i = 0; i < csv.size(); ++i)
        rows.push_back({csv.get(i, "ma_MeV"), csv.get(i, "vdc_cm"), csv.get(i, "calo_cm"),
                        csv.get(i, "bkg_exposure"), csv.get(i, "accepted_fraction"),
                        csv.get(i, "separable_fraction"), csv.get(i, "fom"),
                        csv.has("target_cm") ? csv.get(i, "target_cm") : 0.0});
    ok = true;
    return rows;
}

// final_report.py:118 weighted_score. Each objective is min-max normalised
// across the candidate set, then combined: separable 60%, background -25%,
// accepted 15%. A column with no spread contributes nothing.
std::size_t BestByWeightedScore(const std::vector<Row>& sub,
                                double wSep = 0.60, double wBkg = 0.25, double wAcc = 0.15)
{
    auto norm = [&](double v, double lo, double hi, bool minimize) {
        if (hi == lo) return 0.0;
        const double n = (v - lo) / (hi - lo);
        return minimize ? 1.0 - n : n;
    };
    double sepLo = 1e300, sepHi = -1e300, bkgLo = 1e300, bkgHi = -1e300,
           accLo = 1e300, accHi = -1e300;
    for (const auto& r : sub) {
        sepLo = std::min(sepLo, r.sep); sepHi = std::max(sepHi, r.sep);
        bkgLo = std::min(bkgLo, r.bkg); bkgHi = std::max(bkgHi, r.bkg);
        accLo = std::min(accLo, r.acc); accHi = std::max(accHi, r.acc);
    }
    std::size_t best = 0;
    double bestScore = -1e300;
    for (std::size_t i = 0; i < sub.size(); ++i) {
        const double s = wSep * norm(sub[i].sep, sepLo, sepHi, false)
                       + wBkg * norm(sub[i].bkg, bkgLo, bkgHi, true)
                       + wAcc * norm(sub[i].acc, accLo, accHi, false);
        if (s > bestScore) { bestScore = s; best = i; }
    }
    return best;
}

// final_report.py:56 sensitivity_at_geometry. Scans the coupling grid at one
// fixed geometry and returns the excluded band edges.
void SensitivityAtGeometry(const io::BremsFlux& flux, const AbsCrossSection& absXs,
                           double vdc_cm, double calo_cm,
                           const std::vector<double>& masses,
                           const std::vector<double>& couplings,
                           double exposureDays, double clEvents, bool fixWeights,
                           std::vector<double>& gLo, std::vector<double>& gHi)
{
    const double detDist_m = (vdc_cm + kMagnet_cm) / 100.0;
    const double detArea   = std::pow(calo_cm / 100.0, 2);
    const double maxE = flux.energy_MeV.back();

    gLo.assign(masses.size(), std::nan(""));
    gHi.assign(masses.size(), std::nan(""));

    for (std::size_t i = 0; i < masses.size(); ++i) {
        const double ma = masses[i];
        if (ma >= maxE) continue;
        for (double gGeV : couplings) {
            FluxPrimakoffIsotropic f;
            f.ma = ma; f.gagamma = gGeV / 1000.0; f.targetZ = kTungstenZ;
            f.detDist_m = detDist_m; f.detLength_m = detDist_m; f.detArea_m2 = detArea;
            f.Simulate(flux.energy_MeV, flux.rate_per_s, absXs);
            if (f.axionEnergy.empty()) continue;
            PropagateForScan(f, gGeV / 1000.0, fixWeights);
            if (f.Decays(exposureDays, 0.1) > clEvents) {
                if (std::isnan(gLo[i])) gLo[i] = gGeV;
                gHi[i] = gGeV;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    std::string paretoPath = "output/joint_pareto/Tz10_pareto.csv";
    std::string gridPath   = "output/joint_pareto/Tz10_grid.csv";
    std::string projPath   = "output/target_projection/target_projection_grid.csv";
    std::string fluxPath   = "output/alplib_brems_flux.csv";
    std::string alplibDir  = "alplib";
    std::string outDir     = "output/final_report";
    std::vector<double> massGrid = {1, 5, 10, 20, 50, 100, 200, 500};
    bool skipSensitivity = false, fixWeights = false;
    double exposureDays = 30.0;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--pareto")            paretoPath = nx();
        else if (s == "--grid")              gridPath = nx();
        else if (s == "--proj")              projPath = nx();
        else if (s == "--flux")              fluxPath = nx();
        else if (s == "--alplib")            alplibDir = nx();
        else if (s == "--output-dir")        outDir = nx();
        else if (s == "--exposure-days")     exposureDays = std::stod(nx());
        else if (s == "--skip-sensitivity")  skipSensitivity = true;
        else if (s == "--fix-weights")       fixWeights = true;
        else if (s == "--mass-grid") {
            massGrid.clear(); std::string v = nx(), tok; std::istringstream ss(v);
            while (std::getline(ss, tok, ',')) massGrid.push_back(std::stod(tok));
        }
        else { std::fprintf(stderr, "Unknown argument: %s\n", s.c_str()); return 1; }
    }

    std::printf("=== DAMSA final configuration report ===\n\n");

    bool okPareto = false, okGrid = false, okProj = false;
    const auto pareto = LoadRows(paretoPath, okPareto);
    const auto grid   = LoadRows(gridPath,   okGrid);
    const auto proj   = LoadRows(projPath,   okProj);
    if (!okPareto) { std::fprintf(stderr, "ERROR: cannot read %s\n", paretoPath.c_str()); return 1; }
    std::printf("Phase A Pareto:  %s (%zu rows)\n", paretoPath.c_str(), pareto.size());
    if (okGrid) std::printf("Phase A grid:    %s (%zu rows)\n", gridPath.c_str(), grid.size());
    else        std::printf("WARNING: %s not found - heatmaps skipped\n", gridPath.c_str());
    if (okProj) std::printf("Target proj:     %s (%zu rows)\n", projPath.c_str(), proj.size());
    else        std::printf("WARNING: %s not found - target plots skipped\n", projPath.c_str());

    std::filesystem::create_directories(outDir);
    std::filesystem::create_directories(outDir + "/plots");
    SetPublicationStyle();

    // ── Weighted score -> optimal configuration per mass ────────────────────
    std::printf("\n=== Optimal configuration (weighted score) ===\n");
    std::printf("Weights: separable_frac=60%%, bkg=-25%%, accepted_frac=15%%\n\n");

    std::set<double> masses;
    for (const auto& r : pareto) masses.insert(r.ma);

    std::vector<Row> best;
    std::map<double, int> vdcVotes, caloVotes;
    for (double ma : masses) {
        std::vector<Row> sub;
        for (const auto& r : pareto) if (r.ma == ma) sub.push_back(r);
        if (sub.empty()) continue;
        const auto& b = sub[BestByWeightedScore(sub)];
        best.push_back(b);
        ++vdcVotes[b.vdc_cm]; ++caloVotes[b.calo_cm];
        std::printf("  ma=%5.0f MeV: VDC=%.0f cm, calo=%.0f cm  "
                    "(sep=%.4f, bkg=%.2e, FoM=%.3e)\n",
                    ma, b.vdc_cm, b.calo_cm, b.sep, b.bkg, b.fom);
    }

    // Consensus geometry = the most frequently chosen value (pandas .mode(),
    // which breaks ties by taking the smallest).
    auto mode = [](const std::map<double, int>& votes) {
        double bestV = 0; int bestN = -1;
        for (const auto& [v, n] : votes) if (n > bestN) { bestN = n; bestV = v; }
        return bestV;
    };
    const double bestVdc = mode(vdcVotes), bestCalo = mode(caloVotes);
    std::printf("\nConsensus optimal geometry: VDC=%.0f cm, Calo=%.0f cm\n", bestVdc, bestCalo);

    {
        CsvOut out(outDir + "/optimal_config.csv",
                   "ma_MeV,optimal_vdc_cm,optimal_calo_cm,separable_fraction,bkg_exposure,fom");
        for (const auto& b : best)
            out.Row(b.ma, b.vdc_cm, b.calo_cm, b.sep, b.bkg, b.fom);
    }
    std::printf("Saved config table: %s/optimal_config.csv\n", outDir.c_str());

    // ── Global 3D Pareto across (target, VDC, calo) ─────────────────────────
    if (okProj) {
        std::set<double> pm;
        for (const auto& r : proj) pm.insert(r.ma);
        std::vector<Row> global3d;
        for (double ma : pm) {
            std::vector<Row> sub;
            for (const auto& r : proj) if (r.ma == ma) sub.push_back(r);
            std::vector<std::vector<double>> obj;
            for (const auto& r : sub) obj.push_back({r.bkg, r.sep});
            const auto keep = IsParetoOptimal(obj, {true, false});
            for (std::size_t i = 0; i < sub.size(); ++i) if (keep[i]) global3d.push_back(sub[i]);
        }
        {
            CsvOut out(outDir + "/global_3d_pareto.csv",
                       "ma_MeV,target_cm,vdc_cm,calo_cm,bkg_exposure,"
                       "accepted_fraction,separable_fraction,fom");
            for (const auto& r : global3d)
                out.Row(r.ma, r.target_cm, r.vdc_cm, r.calo_cm, r.bkg, r.acc, r.sep, r.fom);
        }
        std::printf("Saved global 3D Pareto: %s/global_3d_pareto.csv (%zu rows)\n",
                    outDir.c_str(), global3d.size());

        // bkg vs separable, coloured by target length.
        auto* c = new TCanvas("c3d", "", 900, 600);
        c->SetLogx();
        auto* leg = new TLegend(0.15, 0.6, 0.42, 0.88);
        std::set<double> targets;
        for (const auto& r : global3d) targets.insert(r.target_cm);
        int ci = 0;
        for (double t : targets) {
            std::vector<double> xs, ys;
            for (const auto& r : global3d) if (r.target_cm == t) { xs.push_back(r.bkg); ys.push_back(r.sep); }
            if (xs.empty()) continue;
            auto* g = new TGraph(xs.size(), xs.data(), ys.data());
            g->SetMarkerStyle(20 + (ci % 10)); g->SetMarkerColor(kAzure + ci);
            g->SetTitle(";Weighted background [events];Separable signal fraction");
            g->Draw(ci == 0 ? "AP" : "P SAME");
            leg->AddEntry(g, Form("target = %.0f cm", t), "p");
            ++ci;
        }
        leg->Draw();
        SaveCanvas(c, outDir + "/plots/global_3d_pareto");
        delete c;
    }

    // ── FoM heatmaps from the full grid ─────────────────────────────────────
    if (okGrid) {
        std::set<double> vs, cs, gm;
        for (const auto& r : grid) { vs.insert(r.vdc_cm); cs.insert(r.calo_cm); gm.insert(r.ma); }
        const std::vector<double> vv(vs.begin(), vs.end()), cv(cs.begin(), cs.end());
        const double dv = vv.size() > 1 ? vv[1] - vv[0] : 1.0;
        const double dc = cv.size() > 1 ? cv[1] - cv[0] : 1.0;
        for (double ma : gm) {
            auto* h = new TH2D(Form("hf%.0f", ma), "", cv.size(), cv.front() - dc / 2,
                               cv.back() + dc / 2, vv.size(), vv.front() - dv / 2,
                               vv.back() + dv / 2);
            for (const auto& r : grid) if (r.ma == ma) h->Fill(r.calo_cm, r.vdc_cm, r.fom);
            h->SetTitle(Form("FoM = sep / #sqrt{bkg}  m_{a}=%.0f MeV;"
                             "Calo XY size [cm];VDC length [cm]", ma));
            auto* c = new TCanvas(Form("cf%.0f", ma), "", 800, 600);
            h->Draw("COLZ");
            SaveCanvas(c, Form("%s/plots/fom_heatmap_ma%.0fMeV", outDir.c_str(), ma));
            delete c;
        }
    }

    // ── Sensitivity at the consensus geometry ───────────────────────────────
    if (!skipSensitivity) {
        std::printf("\n=== Sensitivity at VDC=%.0f cm, calo=%.0f cm ===\n", bestVdc, bestCalo);
        const auto flux = io::ReadBremsFlux(fluxPath);
        const AbsCrossSection absXs(alplibDir + "/data/photon_absorption/photon_abs_W.txt");

        std::vector<double> couplings;
        for (int i = 0; i < 60; ++i) couplings.push_back(std::pow(10.0, -8.0 + 6.0 * i / 59.0));

        std::vector<double> gLo, gHi;
        SensitivityAtGeometry(flux, absXs, bestVdc, bestCalo, massGrid, couplings,
                              exposureDays, 2.3, fixWeights, gLo, gHi);

        const std::string sp = Form("%s/sensitivity_VDC%.0f_Calo%.0f.csv",
                                    outDir.c_str(), bestVdc, bestCalo);
        {
            CsvOut out(sp, "ma_MeV,g_lower_GeVinv,g_upper_GeVinv");
            for (std::size_t i = 0; i < massGrid.size(); ++i)
                out.Row(massGrid[i], gLo[i], gHi[i]);
        }
        for (std::size_t i = 0; i < massGrid.size(); ++i) {
            if (std::isnan(gLo[i])) std::printf("  ma=%6.0f MeV -> not excluded\n", massGrid[i]);
            else std::printf("  ma=%6.0f MeV -> [%.2e, %.2e] GeV^-1\n", massGrid[i], gLo[i], gHi[i]);
        }
        std::printf("Saved %s\n", sp.c_str());

        // Exclusion band: lower and upper edge vs mass.
        std::vector<double> mx, lo, hi;
        for (std::size_t i = 0; i < massGrid.size(); ++i)
            if (!std::isnan(gLo[i])) { mx.push_back(massGrid[i]); lo.push_back(gLo[i]); hi.push_back(gHi[i]); }
        if (!mx.empty()) {
            auto* c = new TCanvas("csens", "", 800, 600);
            c->SetLogx(); c->SetLogy();
            auto* mg = new TMultiGraph();
            auto* gl = new TGraph(mx.size(), mx.data(), lo.data());
            auto* gh = new TGraph(mx.size(), mx.data(), hi.data());
            gl->SetLineColor(kAzure + 2); gl->SetLineWidth(3); gl->SetMarkerStyle(20);
            gh->SetLineColor(kRed + 1);   gh->SetLineWidth(3); gh->SetMarkerStyle(21);
            mg->Add(gl, "LP"); mg->Add(gh, "LP");
            mg->SetTitle(Form("DAMSA projected exclusion, VDC=%.0f cm calo=%.0f cm;"
                              "m_{a} [MeV];g_{a#gamma#gamma} [GeV^{-1}]", bestVdc, bestCalo));
            mg->Draw("A");
            auto* leg = new TLegend(0.15, 0.75, 0.45, 0.88);
            leg->AddEntry(gl, "lower edge", "lp");
            leg->AddEntry(gh, "upper edge", "lp");
            leg->Draw();
            SaveCanvas(c, outDir + "/plots/sensitivity_curve");
            delete c;
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
