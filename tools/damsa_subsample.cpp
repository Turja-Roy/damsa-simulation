// Subsample a decay-photon file to N nonzero-weight rows.
// Replaces scripts/analysis/subsample_alp_csv.cpp (CSV-only).
//
//   ./build/damsa_subsample <in.root|.csv> <out.root|.csv> <n_target> [seed]
//
// Uniform sampling WITHOUT replacement over the nonzero-weight subset, each kept
// row's weight multiplied by (n_nonzero / n_kept). That is a Horvitz-Thompson
// estimator: total events/day is preserved in expectation, which plain
// truncation (head -n) would not do.
//
// Exists because damsa_alp_inject fires one Geant4 event per row, and a full
// mass point is ~4M rows.

#include "damsa_io.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {
bool IsRoot(const std::string& p) { return p.size() > 5 && p.compare(p.size() - 5, 5, ".root") == 0; }
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::printf("Usage: %s <in.root|.csv> <out.root|.csv> <n_target> [seed]\n\n"
                    "Subsample decay photons to N nonzero-weight rows, rescaling\n"
                    "weights so the total rate is preserved (Horvitz-Thompson).\n", argv[0]);
        return 1;
    }
    const std::string inPath = argv[1], outPath = argv[2];
    const long nTarget = std::atol(argv[3]);
    const std::uint64_t seed = (argc >= 5) ? std::stoull(argv[4]) : 12345;
    if (nTarget <= 0) { std::fprintf(stderr, "n_target must be positive\n"); return 1; }

    // ── Read ────────────────────────────────────────────────────────────────
    std::vector<io::AlpDecayRow> rows;
    if (IsRoot(inPath)) {
        rows = io::ReadNTuple<io::AlpDecayRow>(inPath);
    } else {
        const auto csv = io::ReadCsv(inPath);
        rows.reserve(csv.size());
        for (std::size_t i = 0; i < csv.size(); ++i)
            rows.push_back({csv.get(i, "E1_MeV"), csv.get(i, "px1"), csv.get(i, "py1"),
                            csv.get(i, "pz1"),    csv.get(i, "E2_MeV"), csv.get(i, "px2"),
                            csv.get(i, "py2"),    csv.get(i, "pz2"),
                            csv.get(i, "weight_evts_per_day"), csv.get(i, "decay_z_m")});
    }

    // alplib emits zero-weight rows for grid bins with no ALP production; they
    // carry no rate and firing them only wastes Geant4 events.
    std::vector<std::size_t> nonzero;
    double wIn = 0;
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (rows[i].weight > 0) { nonzero.push_back(i); wIn += rows[i].weight; }

    if (nonzero.empty()) { std::fprintf(stderr, "Error: no nonzero-weight rows in %s\n", inPath.c_str()); return 1; }

    const std::size_t nNonzero = nonzero.size();
    const std::size_t nKeep = std::min<std::size_t>(nTarget, nNonzero);
    const double factor = double(nNonzero) / double(nKeep);

    // Partial shuffle: only the first nKeep need to be drawn.
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < nKeep; ++i) {
        std::uniform_int_distribution<std::size_t> pick(i, nonzero.size() - 1);
        std::swap(nonzero[i], nonzero[pick(rng)]);
    }
    nonzero.resize(nKeep);
    // Keep the original file order, so the output stays comparable to the input.
    std::sort(nonzero.begin(), nonzero.end());

    // ── Write ───────────────────────────────────────────────────────────────
    double wOut = 0;
    if (IsRoot(outPath)) {
        io::NTupleWriter<io::AlpDecayRow> w(outPath);
        for (auto i : nonzero) { auto r = rows[i]; r.weight *= factor; wOut += r.weight; w.Fill(r); }
        w.Finish();
    } else {
        io::EnsureParentDir(outPath);
        std::ofstream out(outPath);
        out << io::kAlpDecayCsvHeader << "\n";
        for (auto i : nonzero) {
            auto r = rows[i]; r.weight *= factor; wOut += r.weight;
            io::WriteAlpDecayCsvRow(out, r);
        }
    }

    std::printf("total=%zu nonzero=%zu kept=%zu reweight=%.6f w_in=%.6e w_out=%.6e\n",
                rows.size(), nNonzero, nKeep, factor, wIn, wOut);
    return 0;
}
