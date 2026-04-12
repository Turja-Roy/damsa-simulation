/*
 * SNR Separability Analysis for DAMSA
 *
 * Computes signal-to-noise ratio (SNR) for ALP signal vs background at the
 * calorimeter face, using 2D (E, theta) likelihood-region cuts.
 *
 * Inputs:
 *   - Background: output/brem_calo_face_particles.csv (from electron-beam run)
 *   - Signal:     output/alp_decay_photons_ma{ma}MeV_calo_face_particles.csv
 *
 * Outputs:
 *   - output/snr_summary.csv            : per-mass SNR numbers
 *   - plots/snr/snr_vs_mass.png         : SNR vs ALP mass curve
 *   - plots/snr/cut_contour_ma{ma}MeV.png : 2D (E, theta) cut visualization
 *
 * Usage:
 *   ./build/snr_separability \
 *       --bkg-csv output/brem_calo_face_particles.csv \
 *       --n-primaries 100000 \
 *       --beam-current-uA 62.5 \
 *       --signal-pattern "output/alp_decay_photons_ma{ma}MeV_calo_face_particles.csv" \
 *       --ma-list 1,5,10,20,50,100,200,500 \
 *       --out-csv output/snr_summary.csv \
 *       --plot-dir plots/snr
 */

#include "TCanvas.h"
#include "TGraph.h"
#include "TH2D.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TSystem.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>

// ─────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────
static constexpr double CHARGE_COULOMBS = 1.602176634e-19;
static constexpr double SECONDS_PER_DAY = 86400.0;

// ─────────────────────────────────────────────────────────────────
// CSV reader (same lightweight approach as flux_converter)
// ─────────────────────────────────────────────────────────────────

static std::string Trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

struct CSVRow {
    std::map<std::string, double> cols;
    double get(const std::string& name, double fallback = 0.0) const {
        auto it = cols.find(name);
        return (it != cols.end()) ? it->second : fallback;
    }
    bool has(const std::string& name) const { return cols.count(name) > 0; }
};

struct CSVData {
    std::vector<std::string> header;
    std::vector<CSVRow> rows;
};

CSVData ReadCSV(const std::string& path) {
    CSVData csv;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "Error: cannot open %s\n", path.c_str());
        return csv;
    }

    std::string line;
    if (!std::getline(f, line)) return csv;
    std::istringstream hss(line);
    std::string tok;
    while (std::getline(hss, tok, ',')) csv.header.push_back(Trim(tok));

    while (std::getline(f, line)) {
        if (Trim(line).empty()) continue;
        std::istringstream rss(line);
        CSVRow row;
        for (size_t i = 0; i < csv.header.size(); ++i) {
            std::string cell;
            if (!std::getline(rss, cell, ',')) break;
            try { row.cols[csv.header[i]] = std::stod(cell); }
            catch (...) { row.cols[csv.header[i]] = 0.0; }
        }
        csv.rows.push_back(row);
    }
    return csv;
}

// ─────────────────────────────────────────────────────────────────
// Particle filtering
// ─────────────────────────────────────────────────────────────────

std::vector<CSVRow> FilterParticles(const std::vector<CSVRow>& rows,
                                     const std::vector<int>& pdgCodes) {
    std::vector<CSVRow> out;
    for (auto& r : rows) {
        int pdg = (int)r.get("pdg");
        for (int code : pdgCodes) {
            if (pdg == code) { out.push_back(r); break; }
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────
// Energy smearing: sigma/E = a/sqrt(E[GeV]) + b
// ─────────────────────────────────────────────────────────────────

std::vector<double> SmearEnergy(const std::vector<double>& E_MeV,
                                 double a, double b,
                                 std::mt19937_64& rng) {
    std::vector<double> out(E_MeV.size());
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (size_t i = 0; i < E_MeV.size(); ++i) {
        double E_GeV = E_MeV[i] / 1000.0;
        double rel_sigma = std::sqrt(
            std::pow(a / std::sqrt(std::max(E_GeV, 1e-6)), 2) + b * b);
        double sigma_MeV = E_MeV[i] * rel_sigma;
        out[i] = std::max(E_MeV[i] + gauss(rng) * sigma_MeV, 0.0);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────
// Polar angle from momentum components
// ─────────────────────────────────────────────────────────────────

double PolarAngle(double px, double py, double pz) {
    double pmag = std::sqrt(px * px + py * py + pz * pz);
    if (pmag < 1e-12) return 0.0;
    double cosTheta = std::clamp(pz / pmag, -1.0, 1.0);
    return std::acos(cosTheta);
}

// ─────────────────────────────────────────────────────────────────
// 2D histogram helpers
// ─────────────────────────────────────────────────────────────────

// Build a 2D array (flat, row-major) from data and bin edges
// Returns [nE x nTheta] array
struct Hist2D {
    int nE, nTheta;
    std::vector<double> data;   // nE * nTheta, row-major
    std::vector<double> eBins;     // nE+1 edges
    std::vector<double> thetaBins; // nTheta+1 edges

    double& at(int ie, int it) { return data[ie * nTheta + it]; }
    double  at(int ie, int it) const { return data[ie * nTheta + it]; }
    int size() const { return nE * nTheta; }
};

Hist2D MakeHist2D(const std::vector<double>& E,
                   const std::vector<double>& theta,
                   const std::vector<double>& weights,
                   const std::vector<double>& eBins,
                   const std::vector<double>& thetaBins) {
    int nE = (int)eBins.size() - 1;
    int nT = (int)thetaBins.size() - 1;
    Hist2D h;
    h.nE = nE;
    h.nTheta = nT;
    h.data.assign(nE * nT, 0.0);
    h.eBins = eBins;
    h.thetaBins = thetaBins;

    for (size_t i = 0; i < E.size(); ++i) {
        // Binary search for bin
        int ie = (int)(std::upper_bound(eBins.begin(), eBins.end(), E[i]) - eBins.begin()) - 1;
        int it = (int)(std::upper_bound(thetaBins.begin(), thetaBins.end(), theta[i]) - thetaBins.begin()) - 1;
        if (ie >= 0 && ie < nE && it >= 0 && it < nT) {
            h.at(ie, it) += weights[i];
        }
    }
    return h;
}

// Gaussian smooth a 2D histogram (sigma in bins, reflect boundary)
void GaussianSmooth2D(Hist2D& h, double sigma) {
    int radius = (int)std::ceil(3.0 * sigma);
    // Build 1D kernel
    std::vector<double> kernel(2 * radius + 1);
    double sum = 0;
    for (int k = -radius; k <= radius; ++k) {
        kernel[k + radius] = std::exp(-0.5 * k * k / (sigma * sigma));
        sum += kernel[k + radius];
    }
    for (auto& v : kernel) v /= sum;

    // Smooth along E axis (rows)
    std::vector<double> tmp(h.data.size());
    for (int ie = 0; ie < h.nE; ++ie) {
        for (int it = 0; it < h.nTheta; ++it) {
            double val = 0;
            for (int k = -radius; k <= radius; ++k) {
                int je = ie + k;
                // Reflect boundary
                if (je < 0) je = -je;
                if (je >= h.nE) je = 2 * h.nE - je - 2;
                je = std::clamp(je, 0, h.nE - 1);
                val += kernel[k + radius] * h.at(je, it);
            }
            tmp[ie * h.nTheta + it] = val;
        }
    }

    // Smooth along theta axis (columns)
    for (int ie = 0; ie < h.nE; ++ie) {
        for (int it = 0; it < h.nTheta; ++it) {
            double val = 0;
            for (int k = -radius; k <= radius; ++k) {
                int jt = it + k;
                if (jt < 0) jt = -jt;
                if (jt >= h.nTheta) jt = 2 * h.nTheta - jt - 2;
                jt = std::clamp(jt, 0, h.nTheta - 1);
                val += kernel[k + radius] * tmp[ie * h.nTheta + jt];
            }
            h.data[ie * h.nTheta + it] = val;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// S/B threshold optimization
// ─────────────────────────────────────────────────────────────────

// Build data-driven k grid from S/B ratios
std::vector<double> DataDrivenKGrid(const Hist2D& S, const Hist2D& B, int n = 60) {
    std::vector<double> valid;
    for (int i = 0; i < S.size(); ++i) {
        if (B.data[i] > 0 && S.data[i] > 0) {
            valid.push_back(S.data[i] / B.data[i]);
        }
    }
    if (valid.size() < 4) return {0.0};

    std::sort(valid.begin(), valid.end());
    int i1 = (int)(0.01 * valid.size());
    int i99 = (int)(0.999 * valid.size());
    i1 = std::max(i1, 0);
    i99 = std::min(i99, (int)valid.size() - 1);
    double lo = valid[i1];
    double hi = valid[i99];
    if (lo <= 0 || hi <= lo) return {0.0};

    std::vector<double> grid(n);
    double logLo = std::log10(lo);
    double logHi = std::log10(hi);
    for (int i = 0; i < n; ++i) {
        grid[i] = std::pow(10.0, logLo + i * (logHi - logLo) / (n - 1));
    }
    return grid;
}

struct ThresholdResult {
    double bestK;
    double bestSNR;
    double S_acc;
    double B_acc;
    std::vector<bool> mask; // accepted bins
};

ThresholdResult BestSBThreshold(const Hist2D& S, const Hist2D& B,
                                 const std::vector<double>& kGrid) {
    int N = S.size();
    ThresholdResult result;
    result.mask.resize(N, false);

    // Baseline: all bins with S > 0
    double baseS = 0, baseB = 0;
    for (int i = 0; i < N; ++i) {
        if (S.data[i] > 0) {
            baseS += S.data[i];
            baseB += B.data[i];
            result.mask[i] = true;
        }
    }
    double baseSNR = (baseS + baseB > 0) ? baseS / std::sqrt(baseS + baseB) : 0.0;

    result.bestK = 0.0;
    result.bestSNR = baseSNR;
    result.S_acc = baseS;
    result.B_acc = baseB;

    // Precompute S/B ratios
    std::vector<double> sbRatio(N);
    for (int i = 0; i < N; ++i) {
        sbRatio[i] = (B.data[i] > 0) ? S.data[i] / B.data[i] : 1e30;
    }

    for (double k : kGrid) {
        double sAcc = 0, bAcc = 0;
        for (int i = 0; i < N; ++i) {
            if (sbRatio[i] > k) {
                sAcc += S.data[i];
                bAcc += B.data[i];
            }
        }
        double snr = (sAcc + bAcc > 0) ? sAcc / std::sqrt(sAcc + bAcc) : 0.0;
        if (snr > result.bestSNR) {
            result.bestSNR = snr;
            result.bestK = k;
            result.S_acc = sAcc;
            result.B_acc = bAcc;
            for (int i = 0; i < N; ++i) result.mask[i] = (sbRatio[i] > k);
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────
// Plotting with ROOT
// ─────────────────────────────────────────────────────────────────

// Convert Hist2D to ROOT TH2D
TH2D* ToROOT(const Hist2D& h, const char* name, const char* title) {
    TH2D* rh = new TH2D(name, title,
                          h.nE, h.eBins.data(),
                          h.nTheta, h.thetaBins.data());
    for (int ie = 0; ie < h.nE; ++ie) {
        for (int it = 0; it < h.nTheta; ++it) {
            rh->SetBinContent(ie + 1, it + 1, h.at(ie, it));
        }
    }
    return rh;
}

void PlotCutContour(const Hist2D& S, const Hist2D& B,
                     const std::vector<bool>& mask,
                     int ma_MeV, double k, double snr,
                     const std::string& outputPath,
                     double exposureDays, double beamCurrentUA) {
    TCanvas* c = new TCanvas("c_contour", "", 1500, 430);
    c->Divide(3, 1);

    // Convert theta bins to degrees for display
    std::vector<double> thetaDeg(B.thetaBins.size());
    for (size_t i = 0; i < B.thetaBins.size(); ++i)
        thetaDeg[i] = B.thetaBins[i] * 180.0 / M_PI;

    // Background
    c->cd(1);
    gPad->SetLogx();
    gPad->SetLogz();
    gPad->SetRightMargin(0.15);
    TH2D* hB = new TH2D("hB", "Background (brem)",
                          B.nE, B.eBins.data(), B.nTheta, thetaDeg.data());
    for (int ie = 0; ie < B.nE; ++ie)
        for (int it = 0; it < B.nTheta; ++it)
            hB->SetBinContent(ie + 1, it + 1, std::max(B.at(ie, it), 1e-10));
    hB->GetXaxis()->SetTitle("E [MeV]");
    hB->GetYaxis()->SetTitle("#theta [deg]");
    hB->SetStats(0);
    hB->Draw("COLZ");

    // Signal
    c->cd(2);
    gPad->SetLogx();
    gPad->SetLogz();
    gPad->SetRightMargin(0.15);
    TH2D* hS = new TH2D("hS", Form("Signal (m_{a}=%d MeV)", ma_MeV),
                          S.nE, S.eBins.data(), S.nTheta, thetaDeg.data());
    for (int ie = 0; ie < S.nE; ++ie)
        for (int it = 0; it < S.nTheta; ++it)
            hS->SetBinContent(ie + 1, it + 1, std::max(S.at(ie, it), 1e-10));
    hS->GetXaxis()->SetTitle("E [MeV]");
    hS->GetYaxis()->SetTitle("#theta [deg]");
    hS->SetStats(0);
    hS->Draw("COLZ");

    // S/B ratio with cut mask
    c->cd(3);
    gPad->SetLogx();
    gPad->SetLogz();
    gPad->SetRightMargin(0.15);
    TH2D* hSB = new TH2D("hSB", Form("S/B  cut k=%.2e  SNR=%.3g", k, snr),
                           S.nE, S.eBins.data(), S.nTheta, thetaDeg.data());
    // Also create a contour histogram for the mask
    TH2D* hMask = new TH2D("hMask", "",
                             S.nE, S.eBins.data(), S.nTheta, thetaDeg.data());
    for (int ie = 0; ie < S.nE; ++ie) {
        for (int it = 0; it < S.nTheta; ++it) {
            double sb = (B.at(ie, it) > 0) ? S.at(ie, it) / B.at(ie, it) : 0;
            hSB->SetBinContent(ie + 1, it + 1, std::max(sb, 1e-3));
            hMask->SetBinContent(ie + 1, it + 1,
                                  mask[ie * S.nTheta + it] ? 1.0 : 0.0);
        }
    }
    hSB->GetXaxis()->SetTitle("E [MeV]");
    hSB->GetYaxis()->SetTitle("#theta [deg]");
    hSB->SetStats(0);
    hSB->Draw("COLZ");
    // Draw contour at 0.5 to show accepted region boundary
    double contourLevel = 0.5;
    hMask->SetContour(1, &contourLevel);
    hMask->SetLineColor(kBlack);
    hMask->SetLineWidth(2);
    hMask->Draw("CONT3 SAME");

    c->SaveAs(outputPath.c_str());
    delete c;
    // ROOT owns the histograms drawn on the canvas pads
}

void PlotSNRvsMass(const std::vector<int>& masses,
                    const std::vector<double>& snrs,
                    const std::string& outputPath,
                    double exposureDays, double beamCurrentUA) {
    TCanvas* c = new TCanvas("c_snr", "SNR vs ALP Mass", 800, 500);
    gPad->SetLogx();
    gPad->SetLogy();
    gPad->SetGrid();
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);

    int N = masses.size();
    std::vector<double> xd(N), yd(N);
    for (int i = 0; i < N; ++i) { xd[i] = masses[i]; yd[i] = snrs[i]; }

    TGraph* g = new TGraph(N, xd.data(), yd.data());
    g->SetTitle(Form("DAMSA SNR vs ALP mass  —  I_{avg}=%.1f #muA CW, T=%.0f days",
                       beamCurrentUA, exposureDays));
    g->SetMarkerStyle(20);
    g->SetMarkerSize(1.5);
    g->SetMarkerColor(kRed + 1);
    g->SetLineColor(kRed + 1);
    g->SetLineWidth(2);
    g->GetXaxis()->SetTitle("ALP mass [MeV]");
    g->GetYaxis()->SetTitle("SNR = S / #sqrt{S+B}  (full exposure)");
    g->GetXaxis()->SetTitleFont(42);
    g->GetYaxis()->SetTitleFont(42);
    g->Draw("ALP");

    c->SaveAs(outputPath.c_str());
    delete c;
}

// ─────────────────────────────────────────────────────────────────
// String replacement helper for signal pattern
// ─────────────────────────────────────────────────────────────────

std::string ReplaceMA(const std::string& pattern, int ma) {
    std::string result = pattern;
    std::string placeholder = "{ma}";
    size_t pos = result.find(placeholder);
    if (pos != std::string::npos) {
        result.replace(pos, placeholder.size(), std::to_string(ma));
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────────────────

void PrintUsage(const char* prog) {
    std::printf(
        "Usage: %s --bkg-csv <path> --n-primaries <N> [options]\n\n"
        "Options:\n"
        "  --bkg-csv          Background calo face CSV (required)\n"
        "  --n-primaries      Number of primary electrons in bkg run (required)\n"
        "  --beam-current-uA  Beam current in uA (default: 62.5)\n"
        "  --signal-pattern   Signal CSV pattern with {ma} placeholder (required)\n"
        "  --ma-list          Comma-separated ALP masses in MeV (default: 1,5,10,20,50,100,200,500)\n"
        "  --calo-sigma-a     Calorimeter resolution stochastic term (default: 0.02)\n"
        "  --calo-sigma-b     Calorimeter resolution constant term (default: 0.01)\n"
        "  --calo-noise-cut   Energy threshold cut in MeV (default: 5.0)\n"
        "  --exposure-days    Exposure time in days (default: 30.0)\n"
        "  --out-csv          Output summary CSV path (default: output/snr_summary.csv)\n"
        "  --plot-dir         Output directory for plots (default: plots/snr)\n"
        "  --seed             Random seed for energy smearing (default: 42)\n",
        prog);
}

std::vector<int> ParseIntList(const std::string& s) {
    std::vector<int> out;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
        out.push_back(std::atoi(tok.c_str()));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    gROOT->SetBatch(true);

    // Defaults
    std::string bkgCsv;
    int nPrimaries = 0;
    double beamCurrentUA = 62.5;
    std::string signalPattern;
    std::vector<int> maList = {1, 5, 10, 20, 50, 100, 200, 500};
    double caloSigmaA = 0.02;
    double caloSigmaB = 0.01;
    double caloNoiseCut = 5.0;
    double exposureDays = 30.0;
    std::string outCsv = "output/snr_summary.csv";
    std::string plotDir = "plots/snr";
    uint64_t seed = 42;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bkg-csv") bkgCsv = argv[++i];
        else if (arg == "--n-primaries") nPrimaries = std::atoi(argv[++i]);
        else if (arg == "--beam-current-uA") beamCurrentUA = std::atof(argv[++i]);
        else if (arg == "--signal-pattern") signalPattern = argv[++i];
        else if (arg == "--ma-list") maList = ParseIntList(argv[++i]);
        else if (arg == "--calo-sigma-a") caloSigmaA = std::atof(argv[++i]);
        else if (arg == "--calo-sigma-b") caloSigmaB = std::atof(argv[++i]);
        else if (arg == "--calo-noise-cut") caloNoiseCut = std::atof(argv[++i]);
        else if (arg == "--exposure-days") exposureDays = std::atof(argv[++i]);
        else if (arg == "--out-csv") outCsv = argv[++i];
        else if (arg == "--plot-dir") plotDir = argv[++i];
        else if (arg == "--seed") seed = std::stoull(argv[++i]);
        else if (arg == "-h" || arg == "--help") { PrintUsage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); return 1; }
    }

    if (bkgCsv.empty() || nPrimaries <= 0 || signalPattern.empty()) {
        std::fprintf(stderr, "Error: --bkg-csv, --n-primaries, and --signal-pattern are required.\n");
        PrintUsage(argv[0]);
        return 1;
    }

    std::mt19937_64 rng(seed);

    // Create output directories
    gSystem->mkdir(plotDir.c_str(), true);
    {
        std::string outDir = outCsv.substr(0, outCsv.rfind('/'));
        if (!outDir.empty()) gSystem->mkdir(outDir.c_str(), true);
    }

    // Normalization: per-primary → events in full exposure
    double eps = (beamCurrentUA * 1e-6) / CHARGE_COULOMBS;
    double exposureS = exposureDays * SECONDS_PER_DAY;
    double electronsInExposure = eps * exposureS;
    double normFactor = electronsInExposure / nPrimaries;

    std::printf("Beam current (avg): %.1f uA   ->  %.3e e/s\n", beamCurrentUA, eps);
    std::printf("Exposure:           %.0f days  ->  %.3e s\n", exposureDays, exposureS);
    std::printf("N primaries (MC):   %d\n", nPrimaries);
    std::printf("Per-primary -> exposure-events scale: %.3e\n\n", normFactor);

    // ── Load background ─────────────────────────────────────────
    std::printf("Loading background: %s\n", bkgCsv.c_str());
    CSVData bkgData = ReadCSV(bkgCsv);
    if (bkgData.rows.empty()) {
        std::fprintf(stderr, "Error: no data in background CSV\n");
        return 1;
    }
    // Filter photons (22) + neutrons (2112)
    auto bkgRows = FilterParticles(bkgData.rows, {22, 2112});
    std::printf("  Loaded %zu particles (photons + neutrons)\n", bkgRows.size());

    // Extract E, theta, weights; smear and cut
    std::vector<double> bkgE, bkgTheta, bkgWeights;
    {
        std::vector<double> rawE(bkgRows.size());
        for (size_t i = 0; i < bkgRows.size(); ++i)
            rawE[i] = bkgRows[i].get("energy_MeV");

        auto smearedE = SmearEnergy(rawE, caloSigmaA, caloSigmaB, rng);

        for (size_t i = 0; i < bkgRows.size(); ++i) {
            if (smearedE[i] >= caloNoiseCut) {
                bkgE.push_back(smearedE[i]);
                bkgWeights.push_back(bkgRows[i].get("weight", 1.0) * normFactor);
                bkgTheta.push_back(PolarAngle(
                    bkgRows[i].get("px"), bkgRows[i].get("py"), bkgRows[i].get("pz")));
            }
        }
    }
    std::printf("  After smearing and %.1f MeV cut: %zu particles\n",
                caloNoiseCut, bkgE.size());

    // ── Define binning ──────────────────────────────────────────
    // Log-spaced E bins, linear theta bins (25x25)
    const int nEBins = 25;
    const int nTBins = 25;
    std::vector<double> eBins(nEBins + 1), thetaBins(nTBins + 1);
    {
        double logMin = std::log10(caloNoiseCut);
        double logMax = std::log10(10000.0);
        for (int i = 0; i <= nEBins; ++i)
            eBins[i] = std::pow(10.0, logMin + i * (logMax - logMin) / nEBins);
        for (int i = 0; i <= nTBins; ++i)
            thetaBins[i] = i * (M_PI / 2.0) / nTBins;
    }

    // Build background histogram
    Hist2D B_raw = MakeHist2D(bkgE, bkgTheta, bkgWeights, eBins, thetaBins);
    double bRawTotal = 0;
    for (auto v : B_raw.data) bRawTotal += v;
    std::printf("  Background total (raw): %.3e events / %.0f-day run\n",
                bRawTotal, exposureDays);

    // Regularize: Gaussian smooth + Poisson-1 floor
    Hist2D B = B_raw;
    GaussianSmooth2D(B, 1.5);

    double bFloor = normFactor; // weight of a single MC entry
    int nFloored = 0, nRawNz = 0, nSmthNz = 0;
    for (int i = 0; i < B.size(); ++i) {
        if (B_raw.data[i] > 0) nRawNz++;
        if (B.data[i] > 0) nSmthNz++;
        if (B.data[i] <= 0) {
            B.data[i] = bFloor;
            nFloored++;
        }
    }
    double bTotal = 0;
    for (auto v : B.data) bTotal += v;
    std::printf("  Background regularized: %.3e events / %.0f-day run\n",
                bTotal, exposureDays);
    std::printf("    nonzero cells: raw=%d/%d  smoothed=%d/%d  floored_empty=%d  (floor=%.3e/cell)\n",
                nRawNz, B.size(), nSmthNz, B.size(), nFloored, bFloor);

    // ── Process each mass point ─────────────────────────────────
    struct Result {
        int ma;
        double S_exposure, B_exposure, snr, thresholdK;
        int nSignalRows, nBkgRows;
    };
    std::vector<Result> results;

    for (int ma : maList) {
        std::string sigPath = ReplaceMA(signalPattern, ma);
        std::printf("\nProcessing ma = %d MeV: %s\n", ma, sigPath.c_str());

        CSVData sigData = ReadCSV(sigPath);
        if (sigData.rows.empty()) {
            std::printf("  WARNING: File not found or empty, skipping\n");
            continue;
        }

        auto sigRows = FilterParticles(sigData.rows, {22});
        std::printf("  Loaded %zu signal photons\n", sigRows.size());
        if (sigRows.empty()) {
            std::printf("  WARNING: No signal photons, skipping\n");
            continue;
        }

        // Extract, smear, cut
        std::vector<double> sigE, sigTheta, sigWeights;
        {
            std::vector<double> rawE(sigRows.size());
            for (size_t i = 0; i < sigRows.size(); ++i)
                rawE[i] = sigRows[i].get("energy_MeV");

            auto smearedE = SmearEnergy(rawE, caloSigmaA, caloSigmaB, rng);

            for (size_t i = 0; i < sigRows.size(); ++i) {
                if (smearedE[i] >= caloNoiseCut) {
                    sigE.push_back(smearedE[i]);
                    sigWeights.push_back(sigRows[i].get("weight", 1.0));
                    sigTheta.push_back(PolarAngle(
                        sigRows[i].get("px"), sigRows[i].get("py"), sigRows[i].get("pz")));
                }
            }
        }
        std::printf("  After smearing and cut: %zu photons\n", sigE.size());

        // Build signal histogram
        Hist2D S = MakeHist2D(sigE, sigTheta, sigWeights, eBins, thetaBins);
        double sTotal = 0;
        for (auto v : S.data) sTotal += v;
        std::printf("  Signal total: %.3e events / %.0f-day run\n", sTotal, exposureDays);

        // Data-driven k grid
        auto kGrid = DataDrivenKGrid(S, B, 60);

        // Find best threshold
        auto tr = BestSBThreshold(S, B, kGrid);
        std::printf("  Best threshold k = %.3e\n", tr.bestK);
        std::printf("  S_accepted = %.3e   B_accepted = %.3e   (events / run)\n",
                    tr.S_acc, tr.B_acc);
        std::printf("  SNR (S/sqrt(S+B)) = %.4g\n", tr.bestSNR);

        results.push_back({ma, tr.S_acc, tr.B_acc, tr.bestSNR, tr.bestK,
                           (int)sigRows.size(), (int)bkgRows.size()});

        // Plot
        std::string plotPath = plotDir + "/cut_contour_ma" + std::to_string(ma) + "MeV.png";
        PlotCutContour(S, B, tr.mask, ma, tr.bestK, tr.bestSNR, plotPath,
                        exposureDays, beamCurrentUA);
        std::printf("  Saved: %s\n", plotPath.c_str());
    }

    // ── Save results CSV ────────────────────────────────────────
    if (!results.empty()) {
        std::ofstream fout(outCsv);
        fout << "ma_MeV,S_exposure,B_exposure,snr,threshold_k,n_signal_rows,n_bkg_rows\n";
        for (auto& r : results) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%d,%.6e,%.6e,%.6g,%.6e,%d,%d\n",
                          r.ma, r.S_exposure, r.B_exposure, r.snr,
                          r.thresholdK, r.nSignalRows, r.nBkgRows);
            fout << buf;
        }
        std::printf("\nSaved: %s\n", outCsv.c_str());

        // Plot SNR vs mass
        std::vector<int> mVec;
        std::vector<double> snrVec;
        for (auto& r : results) { mVec.push_back(r.ma); snrVec.push_back(r.snr); }
        std::string snrPlot = plotDir + "/snr_vs_mass.png";
        PlotSNRvsMass(mVec, snrVec, snrPlot, exposureDays, beamCurrentUA);
        std::printf("Saved: %s\n", snrPlot.c_str());

        // Summary table
        std::printf("\n=== Summary ===\n");
        std::printf("%8s %14s %14s %12s %14s %14s %12s\n",
                    "ma_MeV", "S_exposure", "B_exposure", "snr",
                    "threshold_k", "n_signal", "n_bkg");
        for (auto& r : results) {
            std::printf("%8d %14.6e %14.6e %12.4g %14.6e %14d %12d\n",
                        r.ma, r.S_exposure, r.B_exposure, r.snr,
                        r.thresholdK, r.nSignalRows, r.nBkgRows);
        }
    } else {
        std::printf("\nNo results to save (no signal files found)\n");
        return 1;
    }

    return 0;
}
