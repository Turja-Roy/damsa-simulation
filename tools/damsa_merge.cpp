// Merge per-chunk DAMSA library outputs into the canonical files.
// Replaces scripts/local/merge_library.py.
//
//   ./build/damsa_merge --nchunks 20 [--outdir output]
//
// Three files, three merge rules:
//
//   alplib_brems_flux.csv   Binned "energy_MeV,rate_per_second" where
//                           rate = count * (I/e) / N_electrons. NOT concatenable:
//                           recover the integer count per bin from each chunk,
//                           sum, then renormalize by the TOTAL electrons.
//                           This is the one place a naive concat gives silently
//                           wrong physics. Stays CSV (see damsa_io.h).
//
//   pi0_decays.root         Per-decay rows.
//   calo_face_particles.root Per-particle rows.
//                           Concatenate, OFFSETTING eventID per chunk so the
//                           pileup overlay's per-electron grouping never merges
//                           two different electrons that were both event 0 in
//                           their own chunk.
//
// Note the Python wrote pi0_decays.csv through Python's csv module, which emits
// CRLF; these RNTuple outputs have no such quirk.

#include "damsa_io.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

constexpr double kECharge = 1.602176634e-19;   // C, matches FluxData.h

struct FluxHeader {
    double current_A = 0;
    double nPrimaries = 0;
    std::vector<std::string> lines;
};

FluxHeader ParseFluxHeader(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);

    FluxHeader h;
    std::string line;
    const std::regex reCur(R"(Beam current \[A\]:\s*([0-9.eE+-]+))");
    const std::regex reNp(R"(Primary electrons simulated:\s*([0-9.eE+-]+))");

    while (std::getline(f, line)) {
        if (line.empty() || line[0] != '#') break;
        h.lines.push_back(line);
        std::smatch m;
        if (std::regex_search(line, m, reCur)) h.current_A = std::stod(m[1]);
        if (std::regex_search(line, m, reNp))  h.nPrimaries = std::stod(m[1]);
    }
    if (h.current_A <= 0 || h.nPrimaries <= 0)
        throw std::runtime_error("could not parse current/primaries from " + path);
    return h;
}

void MergeFlux(const std::vector<std::string>& prefixes, const std::string& outDir)
{
    std::map<double, long long> counts;      // bin centre -> summed raw count
    double totalElectrons = 0, beamCurrent = 0;
    std::vector<std::string> templateHeader;
    int nFiles = 0;

    for (const auto& p : prefixes) {
        const std::string path = outDir + "/" + p + "alplib_brems_flux.csv";
        if (!std::filesystem::exists(path)) continue;
        ++nFiles;

        const auto h = ParseFluxHeader(path);
        beamCurrent = h.current_A;
        templateHeader = h.lines;
        totalElectrons += h.nPrimaries;

        // Undo this chunk's own normalization to recover integer photon counts.
        const double scale = (h.current_A / kECharge) / h.nPrimaries;

        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#' || line.rfind("energy", 0) == 0) continue;
            const auto c = line.find(',');
            if (c == std::string::npos) continue;
            const double centre = std::stod(line.substr(0, c));
            const double rate   = std::stod(line.substr(c + 1));
            counts[centre] += static_cast<long long>(std::llround(rate / scale));
        }
    }
    if (nFiles == 0) throw std::runtime_error("no per-chunk alplib_brems_flux.csv found");

    const double scale = (beamCurrent / kECharge) / totalElectrons;
    const std::string outPath = outDir + "/alplib_brems_flux.csv";
    io::EnsureParentDir(outPath);
    std::ofstream out(outPath);

    out << "# Bremsstrahlung photon flux INSIDE target for alplib Primakoff input\n";
    out << "# MERGED from " << nFiles << " chunks by damsa_merge\n";
    // Keep the beam-mode / current lines so the flux reader still finds them.
    for (const auto& h : templateHeader)
        if (h.find("Beam mode") != std::string::npos ||
            h.find("Beam current") != std::string::npos)
            out << h << "\n";
    out << "# Primary electrons simulated: " << (long long)totalElectrons << "\n";
    out << "# Scale factor: " << scale << "\n";
    out << "# Format: energy_MeV, rate_per_second\n#\n";

    out << std::fixed;
    for (const auto& [centre, cnt] : counts) {
        out.precision(3);
        out << centre << ",";
        out << std::scientific;
        out.precision(6);
        out << double(cnt) * scale << "\n";
        out << std::fixed;
    }
    std::printf("[flux] %d chunks, %lld electrons, %zu bins -> %s\n",
                nFiles, (long long)totalElectrons, counts.size(), outPath.c_str());
}

// Concatenate an RNTuple across chunks, offsetting eventID so IDs stay unique.
template <class T>
void MergeRows(const std::vector<std::string>& prefixes, const std::string& outDir,
               const std::string& basename)
{
    std::vector<std::string> found;
    for (const auto& p : prefixes) {
        const std::string path = outDir + "/" + p + basename;
        if (std::filesystem::exists(path)) found.push_back(path);
    }
    if (found.empty()) {
        std::printf("[rows] no chunk files for %s, skipping\n", basename.c_str());
        return;
    }

    const std::string outPath = outDir + "/" + basename;
    io::NTupleWriter<T> w(outPath);
    int offset = 0;
    std::size_t total = 0;

    for (const auto& path : found) {
        auto rows = io::ReadNTuple<T>(path);
        int localMax = -1;
        for (auto& r : rows) {
            localMax = std::max(localMax, r.eventID);
            r.eventID += offset;
            w.Fill(r);
            ++total;
        }
        if (localMax >= 0) offset += localMax + 1;   // next chunk starts past this one
    }
    w.Finish();
    std::printf("[rows] %s: %zu chunks, %zu rows, eventID span 0..%d -> %s\n",
                basename.c_str(), found.size(), total, offset - 1, outPath.c_str());
}

}  // namespace

int main(int argc, char** argv)
{
    int nchunks = -1;
    std::string outDir = "output";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--nchunks") && i + 1 < argc) nchunks = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--outdir") && i + 1 < argc) outDir = argv[++i];
        else { std::fprintf(stderr, "usage: %s --nchunks N [--outdir DIR]\n", argv[0]); return 1; }
    }
    if (nchunks <= 0) { std::fprintf(stderr, "usage: %s --nchunks N [--outdir DIR]\n", argv[0]); return 1; }

    std::vector<std::string> prefixes;
    for (int i = 0; i < nchunks; ++i) {
        char b[32];
        std::snprintf(b, sizeof(b), "chunk%02d_", i);
        prefixes.emplace_back(b);
    }

    try {
        MergeFlux(prefixes, outDir);
        MergeRows<io::Pi0Row>(prefixes, outDir, "pi0_decays.root");
        MergeRows<io::ParticleRow>(prefixes, outDir, "calo_face_particles.root");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
