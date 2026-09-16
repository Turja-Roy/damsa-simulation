// Joint analytic scan of VDC length x calorimeter XY size.
// Replaces scripts/optimization/joint_pareto_scan.py.
//
//   ./build/damsa_joint_scan --particles output/all_particles_target_exit.root \
//        --flux output/alplib_brems_flux.csv --label Tz10
//
// No new Geant4 runs: background comes from straight-line propagation of a fixed
// target-exit particle set, signal from resampling a -> gg decays with the
// geometric acceptance of each (VDC, calo) pair.
//
// "gap" in the older scripts and flags means VDC length only, not VDC + magnet.

#include "damsa_io.h"
#include "pipeline.h"
#include "plotting.h"
#include "scan.h"

#include <TCanvas.h>
#include <TGraph.h>
#include <TH2D.h>
#include <TLegend.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <set>

using namespace damsa::alp;
namespace io = damsa::io;

namespace {

struct Args {
    std::string particles = "output/all_particles_target_exit.root";
    std::string flux      = "output/alplib_brems_flux.csv";
    std::string alplibDir = "alplib";
    std::string outDir    = "output/joint_pareto";
    std::string label     = "Tz10";
    double vdcMin = 30, vdcMax = 60, vdcStep = 5;
    double caloMin = 10, caloMax = 24, caloStep = 2;
    std::vector<double> masses = {1, 5, 10, 20, 50, 100};
    double coupling = -1;              // GeV^-1; <0 = auto per mass
    double beam_uA = 4200.0 * 18 * 929e3 * kCharge_C * 1e6;   // delivered LESA-Laser
    long   nPrimaries = 1000000;
    double exposureDays = 30.0;
    int    nSamples = 10000;
    double angleCut = 10.0;            // separability opening angle [deg]
    double energyCut = 100.0;          // separability per-photon energy [MeV]
    double neutronWeight = 10.0;
    std::uint64_t seed = 42;
    bool   fixWeights = false;         // see PropagateForScan below
};

// One row of the output grid.
struct Row {
    double ma, vdc_cm, calo_cm, bkg, acc, sep, sepEff, nAlp, fom;
};

}  // namespace

int main(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--particles")     a.particles = nx();
        else if (s == "--flux")          a.flux = nx();
        else if (s == "--alplib")        a.alplibDir = nx();
        else if (s == "--output-dir")    a.outDir = nx();
        else if (s == "--label")         a.label = nx();
        else if (s == "--vdc-min")       a.vdcMin = std::stod(nx());
        else if (s == "--vdc-max")       a.vdcMax = std::stod(nx());
        else if (s == "--vdc-step")      a.vdcStep = std::stod(nx());
        else if (s == "--calo-min")      a.caloMin = std::stod(nx());
        else if (s == "--calo-max")      a.caloMax = std::stod(nx());
        else if (s == "--calo-step")     a.caloStep = std::stod(nx());
        else if (s == "--coupling")      a.coupling = std::stod(nx());
        else if (s == "--beam-uA")       a.beam_uA = std::stod(nx());
        else if (s == "--n-primaries")   a.nPrimaries = std::stol(nx());
        else if (s == "--exposure-days") a.exposureDays = std::stod(nx());
        else if (s == "--n-samples")     a.nSamples = std::stoi(nx());
        else if (s == "--angle-cut")     a.angleCut = std::stod(nx());
        else if (s == "--energy-cut")    a.energyCut = std::stod(nx());
        else if (s == "--seed")          a.seed = std::stoull(nx());
        else if (s == "--fix-weights")   a.fixWeights = true;
        else if (s == "--ma-list") {
            a.masses.clear();
            std::string v = nx(), tok; std::istringstream ss(v);
            while (std::getline(ss, tok, ',')) a.masses.push_back(std::stod(tok));
        }
        else { std::fprintf(stderr, "Unknown argument: %s\n", s.c_str()); return 1; }
    }

    std::vector<double> vdc, calo;
    for (double v = a.vdcMin;  v <= a.vdcMax  + 0.1; v += a.vdcStep)  vdc.push_back(v);
    for (double c = a.caloMin; c <= a.caloMax + 0.1; c += a.caloStep) calo.push_back(c);

    std::printf("VDC range:  %.0f - %.0f cm (%zu points)\n", vdc.front(), vdc.back(), vdc.size());
    std::printf("Calo range: %.0f - %.0f cm (%zu points)\n", calo.front(), calo.back(), calo.size());
    std::printf("Grid size:  %zu\n", vdc.size() * calo.size() * a.masses.size());
    if (!a.fixWeights)
        std::printf("[weights] reproducing joint_pareto_scan.py's propagate(decay_width) "
                    "call; pass --fix-weights for the intended physics\n");

    // ── Background grid ─────────────────────────────────────────────────────
    std::printf("\nLoading particles: %s\n", a.particles.c_str());
    const auto parts = io::ReadNTuple<io::ParticleRow>(a.particles);
    std::printf("  %zu particles loaded\n", parts.size());

    const double eps  = (a.beam_uA * 1e-6) / kCharge_C;
    const double norm = (eps * a.exposureDays * kSecPerDay) / double(a.nPrimaries);
    std::printf("  Norm factor: %.3e\n", norm);

    const auto bkg = BackgroundGrid(parts, vdc, calo, norm, a.neutronWeight);
    std::printf("\nBackground grid done.\n");

    // ── Signal ──────────────────────────────────────────────────────────────
    const auto flux = io::ReadBremsFlux(a.flux);
    const AbsCrossSection absXs(a.alplibDir + "/data/photon_absorption/photon_abs_W.txt");
    const double maxE = flux.energy_MeV.back();

    double meanVdc = 0; for (double v : vdc) meanVdc += v; meanVdc /= vdc.size();
    const double nominalDetDist_m = (meanVdc + kMagnetLength_mm / 10.0) / 100.0;

    std::mt19937_64 rng(a.seed);
    std::vector<Row> rows;

    for (double ma : a.masses) {
        std::printf("\n%s\nALP mass: %.0f MeV\n%s\n",
                    std::string(55, '=').c_str(), ma, std::string(55, '=').c_str());
        if (ma >= maxE) { std::printf("  ma >= max photon E=%.1f MeV, skipping\n", maxE); continue; }

        const double gGeV = (a.coupling > 0) ? a.coupling
                                             : AutoCoupling(flux.energy_MeV, flux.rate_per_s, ma);
        const double gMeV = gGeV / 1000.0;

        double nTotal = 0;
        const auto events = GenerateScanEvents(
            flux, absXs, ma, gMeV, a.nSamples, nominalDetDist_m,
            std::pow(a.caloMax / 100.0, 2), a.exposureDays, rng, a.fixWeights, nTotal);
        std::printf("  [alp] %zu samples, n_total=%.3g\n", events.size(), nTotal);
        if (events.empty()) { std::printf("  no ALP production\n"); continue; }

        std::vector<std::vector<double>> acc, sep;
        AcceptanceGrid(events, vdc, calo, a.angleCut, a.energyCut, rng, acc, sep);

        for (std::size_t iv = 0; iv < vdc.size(); ++iv)
            for (std::size_t ic = 0; ic < calo.size(); ++ic) {
                const double b = bkg[iv][ic];
                rows.push_back({ma, vdc[iv], calo[ic], b, acc[iv][ic], sep[iv][ic],
                                acc[iv][ic] > 0 ? sep[iv][ic] / acc[iv][ic] : 0.0,
                                nTotal, sep[iv][ic] / std::sqrt(b + 1e-30)});
            }
    }

    if (rows.empty()) { std::fprintf(stderr, "No results generated.\n"); return 1; }

    // ── Pareto front per mass: minimise background, maximise separable ──────
    std::set<double> massesSeen;
    for (const auto& r : rows) massesSeen.insert(r.ma);
    std::vector<Row> pareto;
    for (double ma : massesSeen) {
        std::vector<Row> sub;
        for (const auto& r : rows) if (r.ma == ma) sub.push_back(r);
        std::vector<std::vector<double>> obj;
        for (const auto& r : sub) obj.push_back({r.bkg, r.sep});
        const auto keep = IsParetoOptimal(obj, {true, false});
        for (std::size_t i = 0; i < sub.size(); ++i) if (keep[i]) pareto.push_back(sub[i]);
    }

    // ── Output ──────────────────────────────────────────────────────────────
    std::filesystem::create_directories(a.outDir);
    std::filesystem::create_directories(a.outDir + "/plots");
    const std::string hdr = "ma_MeV,vdc_cm,calo_cm,bkg_exposure,accepted_fraction,"
                            "separable_fraction,sep_efficiency,n_alp_events,fom";
    for (const auto& [path, data] : std::vector<std::pair<std::string, const std::vector<Row>*>>{
             {a.outDir + "/" + a.label + "_grid.csv",   &rows},
             {a.outDir + "/" + a.label + "_pareto.csv", &pareto}}) {
        CsvOut out(path, hdr);
        for (const auto& r : *data)
            out.Row(r.ma, r.vdc_cm, r.calo_cm, r.bkg, r.acc, r.sep, r.sepEff, r.nAlp, r.fom);
        out.Close();
        std::printf("Saved %s (%zu rows)\n", path.c_str(), data->size());
    }

    // ── Plots ───────────────────────────────────────────────────────────────
    SetPublicationStyle();
    for (double ma : massesSeen) {
        for (const auto& [field, title] : std::vector<std::pair<int, const char*>>{
                 {0, "Separability efficiency"}, {1, "Signal separable fraction"},
                 {2, "Weighted background (log_{10})"}, {3, "FoM = sep / #sqrt{bkg}"}}) {
            auto* h = new TH2D(Form("h_%d_%.0f", field, ma), "",
                               calo.size(), calo.front() - a.caloStep / 2,
                               calo.back() + a.caloStep / 2,
                               vdc.size(), vdc.front() - a.vdcStep / 2,
                               vdc.back() + a.vdcStep / 2);
            for (const auto& r : rows) {
                if (r.ma != ma) continue;
                double v = (field == 0) ? r.sepEff : (field == 1) ? r.sep
                         : (field == 2) ? (r.bkg > 0 ? std::log10(r.bkg) : 0.0) : r.fom;
                h->Fill(r.calo_cm, r.vdc_cm, v);
            }
            h->SetTitle(Form("%s  m_{a}=%.0f MeV (%s);Calo XY size [cm];VDC length [cm]",
                             title, ma, a.label.c_str()));
            auto* c = new TCanvas(Form("c_%d_%.0f", field, ma), "", 800, 600);
            h->Draw("COLZ");
            static const char* names[] = {"sep_efficiency", "separable_fraction",
                                          "bkg_exposure", "fom"};
            SaveCanvas(c, Form("%s/plots/%s_heatmap_%s_ma%.0fMeV",
                               a.outDir.c_str(), a.label.c_str(), names[field], ma));
            delete c;
        }
    }

    // Pareto front: background vs separable fraction, one graph per mass.
    auto* cp = new TCanvas("c_pareto", "", 800, 600);
    cp->SetLogx();
    auto* leg = new TLegend(0.15, 0.6, 0.45, 0.88);
    int ci = 0;
    for (double ma : massesSeen) {
        std::vector<double> xs, ys;
        for (const auto& r : pareto) if (r.ma == ma) { xs.push_back(r.bkg); ys.push_back(r.sep); }
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
    SaveCanvas(cp, Form("%s/plots/%s_pareto_front", a.outDir.c_str(), a.label.c_str()));
    delete cp;

    // ── Summary ─────────────────────────────────────────────────────────────
    std::printf("\n=== Best configuration per mass (highest FoM on the Pareto front) ===\n");
    for (double ma : massesSeen) {
        const Row* best = nullptr;
        for (const auto& r : pareto) if (r.ma == ma && (!best || r.fom > best->fom)) best = &r;
        if (best)
            std::printf("  ma=%5.0f MeV | VDC=%.0f cm calo=%.0f cm | sep=%.4f bkg=%.3e FoM=%.4e\n",
                        ma, best->vdc_cm, best->calo_cm, best->sep, best->bkg, best->fom);
    }
    return 0;
}
