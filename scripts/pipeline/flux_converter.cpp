/*
 * DAMSA Flux Converter for alplib Integration
 *
 * Converts Geant4 simulation output (CSV) to the text format
 * required by alplib for ALP signal calculation.
 *
 * Input:  CSV files from DAMSA Geant4 simulation
 * Output: Text file with columns [energy_MeV, rate_per_second]
 *
 * Usage:
 *   ./build/flux_converter <input.csv> -n <nprimaries> [options]
 *
 * Options:
 *   -n, --nprimaries   Number of primary electrons simulated (required)
 *   -o, --output       Output file path (default: <input>_alplib.txt)
 *   --beam-current     Beam current in uA (default: 62.5)
 *   --bin-width        Energy bin width in MeV (default: 1.0)
 *   --min-energy       Minimum photon energy cut [MeV]
 *   --max-energy       Maximum photon energy cut [MeV]
 *   --max-angle        Maximum angle from beam axis cut [degrees]
 *   --max-time         Maximum arrival time cut [ns]
 *   --unbinned         Output unbinned (per-photon) data
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

// ─────────────────────────────────────────────────────────────────
// CSV reader — reads header, then stores rows as map<column, value>
// ─────────────────────────────────────────────────────────────────

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

// Trim whitespace from both ends
static std::string Trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

CSVData ReadCSV(const std::string& path) {
    CSVData csv;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open " << path << "\n";
        std::exit(1);
    }

    // Read header
    std::string line;
    if (!std::getline(f, line)) {
        std::cerr << "Error: empty file " << path << "\n";
        std::exit(1);
    }
    std::istringstream hss(line);
    std::string tok;
    while (std::getline(hss, tok, ',')) {
        csv.header.push_back(Trim(tok));
    }

    // Read rows
    while (std::getline(f, line)) {
        if (Trim(line).empty()) continue;
        std::istringstream rss(line);
        CSVRow row;
        for (size_t i = 0; i < csv.header.size(); ++i) {
            std::string cell;
            if (!std::getline(rss, cell, ',')) break;
            try {
                row.cols[csv.header[i]] = std::stod(cell);
            } catch (...) {
                row.cols[csv.header[i]] = 0.0;
            }
        }
        csv.rows.push_back(row);
    }
    return csv;
}

// ─────────────────────────────────────────────────────────────────
// Apply cuts
// ─────────────────────────────────────────────────────────────────

struct Cuts {
    double minEnergy = -1;  // MeV, negative = no cut
    double maxEnergy = -1;
    double maxAngle  = -1;  // degrees
    double maxTime   = -1;  // ns
};

std::vector<CSVRow> ApplyCuts(const std::vector<CSVRow>& rows, const Cuts& cuts) {
    std::vector<CSVRow> out;
    for (auto& r : rows) {
        double E = r.get("energy_MeV");
        if (cuts.minEnergy >= 0 && E < cuts.minEnergy) continue;
        if (cuts.maxEnergy >= 0 && E > cuts.maxEnergy) continue;

        if (cuts.maxAngle >= 0 && r.has("pz")) {
            double cosTheta = r.get("pz");
            if (cosTheta < -1) cosTheta = -1;
            if (cosTheta > 1) cosTheta = 1;
            double angleDeg = std::acos(cosTheta) * 180.0 / M_PI;
            if (angleDeg > cuts.maxAngle) continue;
        }

        if (cuts.maxTime >= 0 && r.has("time_ns")) {
            if (r.get("time_ns") > cuts.maxTime) continue;
        }
        out.push_back(r);
    }
    int orig = rows.size();
    int kept = out.size();
    if (orig > 0) {
        std::printf("Photons after cuts: %d / %d (%.1f%%)\n",
                    kept, orig, 100.0 * kept / orig);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────
// Flux conversion
// ─────────────────────────────────────────────────────────────────

static constexpr double E_CHARGE = 1.602176634e-19; // Coulombs

struct FluxBin {
    double energy;  // MeV (bin center)
    double rate;    // photons/second
};

std::vector<FluxBin> ConvertToBinnedFlux(const std::vector<CSVRow>& rows,
                                          int nPrimaries, double beamCurrentUA,
                                          double binWidth) {
    double electronsPerSec = (beamCurrentUA * 1e-6) / E_CHARGE;
    double scaleFactor = electronsPerSec / nPrimaries;

    // Collect energies and weights
    std::vector<double> energies, weights;
    energies.reserve(rows.size());
    weights.reserve(rows.size());
    for (auto& r : rows) {
        energies.push_back(r.get("energy_MeV"));
        weights.push_back(r.has("weight") ? r.get("weight") : 1.0);
    }

    if (energies.empty()) return {};

    double maxE = std::ceil(*std::max_element(energies.begin(), energies.end()));
    int nBins = (int)(maxE / binWidth) + 1;

    // Histogram
    std::vector<double> counts(nBins, 0.0);
    for (size_t i = 0; i < energies.size(); ++i) {
        int bin = (int)(energies[i] / binWidth);
        if (bin >= 0 && bin < nBins) counts[bin] += weights[i];
    }

    // Build result (skip empty bins)
    std::vector<FluxBin> flux;
    for (int i = 0; i < nBins; ++i) {
        if (counts[i] > 0) {
            double center = (i + 0.5) * binWidth;
            flux.push_back({center, counts[i] * scaleFactor});
        }
    }
    return flux;
}

std::vector<FluxBin> ConvertToUnbinnedFlux(const std::vector<CSVRow>& rows,
                                            int nPrimaries, double beamCurrentUA) {
    double electronsPerSec = (beamCurrentUA * 1e-6) / E_CHARGE;
    double scaleFactor = electronsPerSec / nPrimaries;

    std::vector<FluxBin> flux;
    flux.reserve(rows.size());
    for (auto& r : rows) {
        double E = r.get("energy_MeV");
        double w = r.has("weight") ? r.get("weight") : 1.0;
        flux.push_back({E, w * scaleFactor});
    }
    return flux;
}

// ─────────────────────────────────────────────────────────────────
// Output
// ─────────────────────────────────────────────────────────────────

void SaveAlplibFlux(const std::vector<FluxBin>& flux, const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot write to " << path << "\n";
        std::exit(1);
    }
    f << "# Photon flux for alplib input\n";
    f << "# Generated by DAMSA flux_converter\n";
    f << "# Format: energy_MeV, rate_per_second\n";
    f << "# Usage in alplib:\n";
    f << "#   flux_data = np.loadtxt('this_file.txt', delimiter=',')\n";
    f << "#   flux = FluxPrimakoffIsotropic(photon_flux=flux_data, ...)\n";

    for (auto& bin : flux) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.3f,%.6e\n", bin.energy, bin.rate);
        f << buf;
    }
    std::cout << "Saved alplib flux to: " << path << "\n";
}

void PrintFluxSummary(const std::vector<FluxBin>& flux, int nOrigPhotons) {
    std::cout << "\n=== Flux Summary ===\n";
    std::cout << "Number of energy bins: " << flux.size() << "\n";
    if (!flux.empty()) {
        double minE = flux.front().energy, maxE = flux.front().energy;
        double totalRate = 0, peakRate = 0, peakE = 0;
        for (auto& b : flux) {
            if (b.energy < minE) minE = b.energy;
            if (b.energy > maxE) maxE = b.energy;
            totalRate += b.rate;
            if (b.rate > peakRate) { peakRate = b.rate; peakE = b.energy; }
        }
        std::printf("Energy range: %.1f - %.1f MeV\n", minE, maxE);
        std::printf("Total rate: %.3e photons/second\n", totalRate);
        std::printf("Peak rate at: %.1f MeV\n", peakE);
        std::printf("Peak rate: %.3e photons/second\n", peakRate);
    }
    if (nOrigPhotons > 0) {
        std::printf("\nOriginal data: %d photons\n", nOrigPhotons);
    }
    std::cout << "===================\n\n";
}

// ─────────────────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────────────────

void PrintUsage(const char* prog) {
    std::printf(
        "Usage: %s <input.csv> -n <nprimaries> [options]\n\n"
        "Options:\n"
        "  -n, --nprimaries   Number of primary electrons (required)\n"
        "  -o, --output       Output file path (default: <input>_alplib.txt)\n"
        "  --beam-current     Beam current in uA (default: 62.5)\n"
        "  --bin-width        Energy bin width in MeV (default: 1.0)\n"
        "  --min-energy       Minimum photon energy cut [MeV]\n"
        "  --max-energy       Maximum photon energy cut [MeV]\n"
        "  --max-angle        Maximum angle from beam axis cut [degrees]\n"
        "  --max-time         Maximum arrival time cut [ns]\n"
        "  --unbinned         Output unbinned (per-photon) data\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { PrintUsage(argv[0]); return 1; }

    std::string inputPath;
    std::string outputPath;
    int nPrimaries = -1;
    double beamCurrent = 62.5;
    double binWidth = 1.0;
    bool unbinned = false;
    Cuts cuts;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--nprimaries") { nPrimaries = std::atoi(argv[++i]); }
        else if (arg == "-o" || arg == "--output") { outputPath = argv[++i]; }
        else if (arg == "--beam-current") { beamCurrent = std::atof(argv[++i]); }
        else if (arg == "--bin-width") { binWidth = std::atof(argv[++i]); }
        else if (arg == "--min-energy") { cuts.minEnergy = std::atof(argv[++i]); }
        else if (arg == "--max-energy") { cuts.maxEnergy = std::atof(argv[++i]); }
        else if (arg == "--max-angle") { cuts.maxAngle = std::atof(argv[++i]); }
        else if (arg == "--max-time") { cuts.maxTime = std::atof(argv[++i]); }
        else if (arg == "--unbinned") { unbinned = true; }
        else if (arg == "-h" || arg == "--help") { PrintUsage(argv[0]); return 0; }
        else if (inputPath.empty()) { inputPath = arg; }
        else { std::cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    if (inputPath.empty() || nPrimaries <= 0) {
        std::cerr << "Error: input file and --nprimaries are required.\n";
        PrintUsage(argv[0]);
        return 1;
    }

    // Default output path
    if (outputPath.empty()) {
        std::string stem = inputPath;
        size_t dot = stem.rfind('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        outputPath = stem + "_alplib.txt";
    }

    // Load CSV
    std::printf("Loading: %s\n", inputPath.c_str());
    CSVData csv = ReadCSV(inputPath);
    std::printf("Loaded %zu photons\n", csv.rows.size());

    // Apply cuts
    auto rows = csv.rows;
    bool hasCuts = (cuts.minEnergy >= 0 || cuts.maxEnergy >= 0 ||
                    cuts.maxAngle >= 0 || cuts.maxTime >= 0);
    if (hasCuts) {
        rows = ApplyCuts(rows, cuts);
    }

    // Convert
    std::vector<FluxBin> flux;
    if (unbinned) {
        flux = ConvertToUnbinnedFlux(rows, nPrimaries, beamCurrent);
    } else {
        flux = ConvertToBinnedFlux(rows, nPrimaries, beamCurrent, binWidth);
    }

    // Summary and save
    PrintFluxSummary(flux, csv.rows.size());
    SaveAlplibFlux(flux, outputPath);

    std::cout << "Done!\n";
    return 0;
}
