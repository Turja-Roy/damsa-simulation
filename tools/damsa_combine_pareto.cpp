// Combine per-target-length Pareto CSVs into a global 3D Pareto front.
// Replaces scripts/optimization/combine_pareto.py.
//
//   ./build/damsa_combine_pareto output/joint_pareto/Tz*_pareto.csv
//
// Each input is one damsa_joint_scan run at a fixed target length; the length is
// parsed from the "Tz<N>" in the filename. The global front is then taken across
// all (target, VDC, calo) combinations, per mass.

#include "damsa_io.h"
#include "plotting.h"
#include "scan.h"

#include <TCanvas.h>
#include <TGraph.h>
#include <TLegend.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <regex>
#include <set>
#include <string>
#include <vector>

using namespace damsa::alp;
namespace io = damsa::io;

namespace {

struct Row {
    double ma, vdc_cm, calo_cm, bkg, acc, sep, sepEff, nAlp, fom, target_cm;
};

// "Tz14" -> 14. Accepts a bare label or a full path.
bool ParseTargetLength(const std::string& s, double& out)
{
    static const std::regex re(R"(Tz(\d+(?:\.\d+)?))");
    std::smatch m;
    if (!std::regex_search(s, m, re)) return false;
    out = std::stod(m[1]);
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> files;
    std::string outPath = "output/global_pareto.csv";
    std::string plotDir = "plots/global_pareto";

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if      (s == "--output")   outPath = argv[++i];
        else if (s == "--plot-dir") plotDir = argv[++i];
        else if (s.rfind("--", 0) == 0) {
            std::fprintf(stderr, "usage: %s <Tz*_pareto.csv ...> [--output CSV] [--plot-dir DIR]\n", argv[0]);
            return 1;
        }
        else files.push_back(s);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: %s <Tz*_pareto.csv ...> [--output CSV] [--plot-dir DIR]\n", argv[0]);
        return 1;
    }
    std::sort(files.begin(), files.end());

    // ── Load and tag ────────────────────────────────────────────────────────
    std::vector<Row> combined;
    for (const auto& f : files) {
        double target = 0;
        if (!ParseTargetLength(f, target)) {
            std::fprintf(stderr, "WARNING: cannot parse target length from '%s' "
                                 "(expected 'Tz<N>') -- skipping\n", f.c_str());
            continue;
        }
        const auto csv = io::ReadCsv(f);
        std::size_t n = 0;
        for (std::size_t i = 0; i < csv.size(); ++i) {
            Row r;
            r.ma      = csv.get(i, "ma_MeV");
            r.vdc_cm  = csv.get(i, "vdc_cm");
            r.calo_cm = csv.get(i, "calo_cm");
            r.bkg     = csv.get(i, "bkg_exposure");
            r.acc     = csv.get(i, "accepted_fraction");
            r.sep     = csv.get(i, "separable_fraction");
            r.sepEff  = csv.get(i, "sep_efficiency");
            r.nAlp    = csv.get(i, "n_alp_events");
            // Recompute when absent, as the Python does.
            r.fom     = csv.has("fom") ? csv.get(i, "fom")
                                       : r.sep / std::sqrt(r.bkg + 1e-30);
            r.target_cm = target;
            combined.push_back(r);
            ++n;
        }
        std::printf("Loaded %s: %zu rows, target=%.0f cm\n", f.c_str(), n, target);
    }
    if (combined.empty()) { std::fprintf(stderr, "No valid files loaded.\n"); return 1; }
    std::printf("\nCombined: %zu rows total\n", combined.size());

    // ── Global Pareto per mass: minimise background, maximise separable ─────
    std::set<double> masses;
    for (const auto& r : combined) masses.insert(r.ma);

    std::vector<Row> global;
    for (double ma : masses) {
        std::vector<Row> sub;
        for (const auto& r : combined) if (r.ma == ma) sub.push_back(r);
        std::vector<std::vector<double>> obj;
        for (const auto& r : sub) obj.push_back({r.bkg, r.sep});
        const auto keep = IsParetoOptimal(obj, {true, false});
        for (std::size_t i = 0; i < sub.size(); ++i) if (keep[i]) global.push_back(sub[i]);
    }

    io::EnsureParentDir(outPath);
    {
        CsvOut out(outPath, "ma_MeV,vdc_cm,calo_cm,bkg_exposure,accepted_fraction,"
                            "separable_fraction,sep_efficiency,n_alp_events,fom,"
                            "target_cm,global_pareto");
        for (const auto& r : global)
            out.Row(r.ma, r.vdc_cm, r.calo_cm, r.bkg, r.acc, r.sep, r.sepEff,
                    r.nAlp, r.fom, r.target_cm, 1);
    }
    std::printf("\nGlobal Pareto saved: %s  (%zu rows)\n", outPath.c_str(), global.size());

    // ── Summary ─────────────────────────────────────────────────────────────
    std::printf("\n=== Best configuration per mass (highest FoM on global Pareto) ===\n");
    std::printf("%10s %12s %9s %10s %10s %12s %12s\n",
                "ma [MeV]", "target [cm]", "VDC [cm]", "calo [cm]", "sep_frac", "bkg", "FoM");
    std::printf("%s\n", std::string(77, '-').c_str());
    for (double ma : masses) {
        const Row* best = nullptr;
        for (const auto& r : global) if (r.ma == ma && (!best || r.fom > best->fom)) best = &r;
        if (best)
            std::printf("%10.0f %12.0f %9.0f %10.0f %10.4f %12.3e %12.4e\n",
                        ma, best->target_cm, best->vdc_cm, best->calo_cm,
                        best->sep, best->bkg, best->fom);
    }

    // ── Plot: global front coloured by mass ─────────────────────────────────
    std::filesystem::create_directories(plotDir);
    SetPublicationStyle();
    auto* c = new TCanvas("c_global", "", 900, 600);
    c->SetLogx();
    auto* leg = new TLegend(0.15, 0.6, 0.45, 0.88);
    int ci = 0;
    for (double ma : masses) {
        std::vector<double> xs, ys;
        for (const auto& r : global) if (r.ma == ma) { xs.push_back(r.bkg); ys.push_back(r.sep); }
        if (xs.empty()) continue;
        auto* g = new TGraph(xs.size(), xs.data(), ys.data());
        g->SetMarkerStyle(20 + (ci % 10));
        g->SetMarkerColor(kAzure + ci);
        g->SetTitle(";Weighted background [events];Separable signal fraction");
        g->Draw(ci == 0 ? "AP" : "P SAME");
        leg->AddEntry(g, Form("m_{a} = %.0f MeV", ma), "p");
        ++ci;
    }
    leg->Draw();
    SaveCanvas(c, plotDir + "/global_pareto_front");
    delete c;
    return 0;
}
