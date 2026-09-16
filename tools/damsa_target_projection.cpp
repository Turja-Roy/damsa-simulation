// Analytic projection of the VDC x calo scan across target lengths.
// Replaces scripts/analysis/project_target_length.py.
//
//   ./build/damsa_target_projection --particles output/all_particles_target_exit.root \
//        --flux output/alplib_brems_flux.csv
//
// No new Geant4 runs for 11-20 cm targets: the 10 cm target-exit particle set is
// attenuated through the extra tungsten, and the ALP signal is scaled by the
// shower-saturation factor. The brems flux itself is target-length independent,
// so the signal events are generated once and reused.

#include "damsa_io.h"
#include "plotting.h"
#include "scan.h"

#include <TCanvas.h>
#include <TGraph.h>
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

// NIST XCOM, Z=74 tungsten, total attenuation with coherent scattering [cm^2/g].
constexpr double kWRho = 19.3;   // g/cm^3
const std::vector<double> kMuE = {
    0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0,
    8.0, 10., 15., 20., 30., 40., 50., 60., 80., 100.};
const std::vector<double> kMuPerRho = {
    0.2260, 0.1220, 0.09914, 0.08939, 0.07421, 0.05893, 0.04824, 0.04212,
    0.04540, 0.04082, 0.03651, 0.03267, 0.03031, 0.02761, 0.02885, 0.02806,
    0.02773, 0.02751, 0.02743, 0.02737, 0.02732, 0.02726, 0.02721};

// Total photon attenuation coefficient in W [cm^-1]. Energies are CLAMPED to the
// table range before interpolating, as np.clip does in the Python -- this is not
// the log-space left/right fill that alplib's AbsCrossSection uses.
double MuW(double energy_MeV)
{
    const double e = std::clamp(energy_MeV, kMuE.front(), kMuE.back());
    const auto it = std::upper_bound(kMuE.begin(), kMuE.end(), e);
    if (it == kMuE.end()) return kMuPerRho.back() * kWRho;
    const std::size_t i = std::distance(kMuE.begin(), it);
    if (i == 0) return kMuPerRho.front() * kWRho;
    const double x0 = kMuE[i - 1], x1 = kMuE[i];
    const double y0 = kMuPerRho[i - 1] * kWRho, y1 = kMuPerRho[i] * kWRho;
    return y0 + (y1 - y0) * (e - x0) / (x1 - x0);
}

// Shower fraction relative to a 10 cm target, pre-computed from the Longo
// parameterisation. The whole range spans 0.02%, so the target length barely
// changes the ALP yield -- it is tracked anyway rather than assumed to be 1.
const std::vector<double> kShowerL = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
const std::vector<double> kShowerScale = {
    1.000000, 1.000051, 1.000103, 1.000142, 1.000167, 1.000181,
    1.000191, 1.000197, 1.000201, 1.000204, 1.000206};

double ShowerSignalScale(double target_cm)
{
    if (target_cm <= kShowerL.front()) return kShowerScale.front();
    if (target_cm >= kShowerL.back())  return kShowerScale.back();
    const auto it = std::upper_bound(kShowerL.begin(), kShowerL.end(), target_cm);
    const std::size_t i = std::distance(kShowerL.begin(), it);
    const double x0 = kShowerL[i - 1], x1 = kShowerL[i];
    const double y0 = kShowerScale[i - 1], y1 = kShowerScale[i];
    return y0 + (y1 - y0) * (target_cm - x0) / (x1 - x0);
}

// Scale weights by survival through `extra_cm` of tungsten beyond the 10 cm
// baseline. Photons use the energy-resolved NIST mu; electrons and positrons are
// treated as exp(-extra/0.3 cm), since their range in W at shower energies is
// well under a centimetre.
std::vector<io::ParticleRow> ApplyAttenuation(const std::vector<io::ParticleRow>& in,
                                              double extra_cm)
{
    if (extra_cm <= 0) return in;
    auto out = in;
    for (auto& p : out) {
        if (p.pdg == 22)                      p.weight *= std::exp(-MuW(p.energy_MeV) * extra_cm);
        else if (p.pdg == 11 || p.pdg == -11) p.weight *= std::exp(-extra_cm / 0.30);
    }
    return out;
}

struct Row {
    double target_cm, ma, vdc_cm, calo_cm, sigScale, bkg, acc, sep, sepEff, nAlp, fom;
};

}  // namespace

int main(int argc, char** argv)
{
    std::string particles = "output/all_particles_target_exit.root";
    std::string fluxPath  = "output/alplib_brems_flux.csv";
    std::string alplibDir = "alplib";
    std::string outDir    = "output/target_projection";
    int targetMin = 10, targetMax = 20;
    double vdcMin = 30, vdcMax = 40, vdcStep = 2;
    double caloMin = 12, caloMax = 20, caloStep = 4;
    std::vector<double> masses = {1, 5, 10, 20, 50, 100};
    double beam_uA = 4200.0 * 18 * 929e3 * kCharge_C * 1e6;
    long nPrimaries = 100000;
    double exposureDays = 30.0, coupling = -1;
    int nSamples = 20000;
    std::uint64_t seed = 42;
    bool fixWeights = false;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--particles")     particles = nx();
        else if (s == "--flux")          fluxPath = nx();
        else if (s == "--alplib")        alplibDir = nx();
        else if (s == "--output-dir")    outDir = nx();
        else if (s == "--target-min")    targetMin = std::stoi(nx());
        else if (s == "--target-max")    targetMax = std::stoi(nx());
        else if (s == "--vdc-min")       vdcMin = std::stod(nx());
        else if (s == "--vdc-max")       vdcMax = std::stod(nx());
        else if (s == "--vdc-step")      vdcStep = std::stod(nx());
        else if (s == "--calo-min")      caloMin = std::stod(nx());
        else if (s == "--calo-max")      caloMax = std::stod(nx());
        else if (s == "--calo-step")     caloStep = std::stod(nx());
        else if (s == "--beam-uA")       beam_uA = std::stod(nx());
        else if (s == "--n-primaries")   nPrimaries = std::stol(nx());
        else if (s == "--exposure-days") exposureDays = std::stod(nx());
        else if (s == "--n-samples")     nSamples = std::stoi(nx());
        else if (s == "--coupling")      coupling = std::stod(nx());
        else if (s == "--seed")          seed = std::stoull(nx());
        else if (s == "--fix-weights")   fixWeights = true;
        else if (s == "--ma-list") {
            masses.clear(); std::string v = nx(), tok; std::istringstream ss(v);
            while (std::getline(ss, tok, ',')) masses.push_back(std::stod(tok));
        }
        else { std::fprintf(stderr, "Unknown argument: %s\n", s.c_str()); return 1; }
    }

    std::vector<double> vdc, calo;
    for (double v = vdcMin;  v <= vdcMax  + 0.1; v += vdcStep)  vdc.push_back(v);
    for (double c = caloMin; c <= caloMax + 0.1; c += caloStep) calo.push_back(c);

    const double eps  = (beam_uA * 1e-6) / kCharge_C;
    const double norm = (eps * exposureDays * kSecPerDay) / double(nPrimaries);

    std::printf("=== DAMSA target length projection (analytic) ===\n");
    std::printf("Target range: %d-%d cm   VDC: %.0f-%.0f cm   calo: %.0f-%.0f cm\n",
                targetMin, targetMax, vdc.front(), vdc.back(), calo.front(), calo.back());
    std::printf("Norm factor:  %.3e\n\n", norm);
    if (!fixWeights)
        std::printf("[weights] reproducing the Python's propagate(decay_width) call; "
                    "pass --fix-weights for the intended physics\n\n");

    const auto base = io::ReadNTuple<io::ParticleRow>(particles);
    const auto flux = io::ReadBremsFlux(fluxPath);
    const AbsCrossSection absXs(alplibDir + "/data/photon_absorption/photon_abs_W.txt");
    std::printf("Loaded %zu particles, %zu flux bins\n\n", base.size(), flux.size());

    // ── Diagnostics: how much does the target length actually change? ───────
    std::printf("Signal scale factors (relative to 10 cm):\n");
    for (int L = targetMin; L <= targetMax; ++L)
        std::printf("  Tz%02d: %.6f  (delta = %+.4f %%)\n",
                    L, ShowerSignalScale(L), (ShowerSignalScale(L) - 1) * 100);

    std::printf("\nMean photon survival fractions:\n");
    for (int L = targetMin; L <= targetMax; ++L) {
        const double extra = L - 10;
        if (extra == 0) { std::printf("  Tz%02d: baseline (extra = 0 cm)\n", L); continue; }
        double sw = 0, swx = 0;
        for (const auto& p : base)
            if (p.pdg == 22) { sw += p.weight; swx += p.weight * std::exp(-MuW(p.energy_MeV) * extra); }
        std::printf("  Tz%02d: %.4f  (extra %.0f cm W)\n", L, sw > 0 ? swx / sw : 0.0, extra);
    }

    // ── Signal events, generated once (flux is target-length independent) ───
    std::mt19937_64 rng(seed);
    double meanVdc = 0; for (double v : vdc) meanVdc += v; meanVdc /= vdc.size();
    const double detDist_m = (meanVdc + kMagnetLength_mm / 10.0) / 100.0;

    std::map<double, std::pair<std::vector<ScanEvent>, double>> signal;
    std::printf("\n");
    for (double ma : masses) {
        double nTot = 0;
        const double gGeV = (coupling > 0) ? coupling
                                           : AutoCoupling(flux.energy_MeV, flux.rate_per_s, ma);
        auto ev = GenerateScanEvents(flux, absXs, ma, gGeV / 1000.0, nSamples,
                                     detDist_m, std::pow(caloMax / 100.0, 2),
                                     exposureDays, rng, fixWeights, nTot);
        std::printf("ma = %5.0f MeV -> %zu events, n_total = %.3g\n", ma, ev.size(), nTot);
        signal[ma] = {std::move(ev), nTot};
    }

    // ── Main loop over target lengths ───────────────────────────────────────
    std::vector<Row> rows;
    for (int L = targetMin; L <= targetMax; ++L) {
        const double extra = L - 10.0;
        const double sigScale = ShowerSignalScale(L);
        std::printf("\n--- Tz%d (extra W = %.0f cm, signal scale = %.6f) ---\n", L, extra, sigScale);

        const auto parts = ApplyAttenuation(base, extra);
        const auto bkg = BackgroundGrid(parts, vdc, calo, norm);

        for (double ma : masses) {
            auto& [evBase, nTotBase] = signal[ma];
            if (evBase.empty()) continue;

            auto ev = evBase;
            for (auto& e : ev) e.weight *= sigScale;
            const double nTot = nTotBase * sigScale;

            std::vector<std::vector<double>> acc, sep;
            AcceptanceGrid(ev, vdc, calo, 10.0, 100.0, rng, acc, sep);

            for (std::size_t iv = 0; iv < vdc.size(); ++iv)
                for (std::size_t ic = 0; ic < calo.size(); ++ic) {
                    const double b = bkg[iv][ic];
                    rows.push_back({double(L), ma, vdc[iv], calo[ic], sigScale, b,
                                    acc[iv][ic], sep[iv][ic],
                                    acc[iv][ic] > 0 ? sep[iv][ic] / acc[iv][ic] : 0.0,
                                    nTot, sep[iv][ic] / std::sqrt(b + 1e-30)});
                }
        }
        const Row* best = nullptr;
        for (const auto& r : rows) if (r.target_cm == L && (!best || r.fom > best->fom)) best = &r;
        if (best)
            std::printf("  Best: VDC=%.0f cm calo=%.0f cm  sep=%.4f bkg=%.3e FoM=%.4e\n",
                        best->vdc_cm, best->calo_cm, best->sep, best->bkg, best->fom);
    }

    if (rows.empty()) { std::fprintf(stderr, "No results generated.\n"); return 1; }

    // ── Output ──────────────────────────────────────────────────────────────
    std::filesystem::create_directories(outDir);
    std::filesystem::create_directories(outDir + "/plots");
    const std::string path = outDir + "/target_projection_grid.csv";
    {
        CsvOut out(path, "target_cm,ma_MeV,vdc_cm,calo_cm,signal_scale,bkg_exposure,"
                         "accepted_fraction,separable_fraction,sep_efficiency,"
                         "n_alp_events,fom");
        for (const auto& r : rows)
            out.Row(r.target_cm, r.ma, r.vdc_cm, r.calo_cm, r.sigScale, r.bkg,
                    r.acc, r.sep, r.sepEff, r.nAlp, r.fom);
    }
    std::printf("\nSaved %s (%zu rows)\n", path.c_str(), rows.size());

    // ── Plot: best FoM vs target length, one curve per mass ─────────────────
    SetPublicationStyle();
    std::set<double> massSet(masses.begin(), masses.end());
    auto* c = new TCanvas("c_fom", "", 800, 600);
    auto* mg = new TMultiGraph();
    auto* leg = new TLegend(0.15, 0.62, 0.42, 0.88);
    int ci = 0;
    for (double ma : massSet) {
        std::vector<double> xs, ys;
        for (int L = targetMin; L <= targetMax; ++L) {
            const Row* best = nullptr;
            for (const auto& r : rows)
                if (r.target_cm == L && r.ma == ma && (!best || r.fom > best->fom)) best = &r;
            if (best) { xs.push_back(L); ys.push_back(best->fom); }
        }
        if (xs.empty()) continue;
        auto* g = new TGraph(xs.size(), xs.data(), ys.data());
        g->SetLineColor(kAzure + ci); g->SetMarkerColor(kAzure + ci);
        g->SetMarkerStyle(20 + (ci % 10)); g->SetLineWidth(2);
        mg->Add(g, "LP");
        leg->AddEntry(g, Form("m_{a} = %.0f MeV", ma), "lp");
        ++ci;
    }
    mg->SetTitle(";Target length [cm];Best FoM = sep / #sqrt{bkg}");
    mg->Draw("A");
    leg->Draw();
    SaveCanvas(c, outDir + "/plots/fom_vs_target");
    delete c;
    return 0;
}
