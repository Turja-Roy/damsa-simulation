/*
 * Subsample an alplib decay-photon CSV to N nonzero-weight rows.
 *
 * Uniform random sampling without replacement over the nonzero-weight subset,
 * with each kept row's weight multiplied by (n_nonzero / n_kept) so the total
 * weight (events/day) is preserved in expectation (Horvitz-Thompson estimator).
 *
 * Usage:
 *   ./build/subsample_alp_csv <in.csv> <out.csv> <n_target> [seed]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf(
            "Usage: %s <in.csv> <out.csv> <n_target> [seed]\n\n"
            "Subsample an alplib decay-photon CSV to N nonzero-weight rows.\n"
            "Weights are rescaled to preserve total rate (Horvitz-Thompson).\n",
            argv[0]);
        return 1;
    }

    std::string inPath  = argv[1];
    std::string outPath = argv[2];
    int nTarget = std::atoi(argv[3]);
    uint64_t seed = (argc >= 5) ? std::stoull(argv[4]) : 12345;

    // ── Read input CSV ──────────────────────────────────────────
    std::ifstream fin(inPath);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open " << inPath << "\n";
        return 1;
    }

    std::string headerLine;
    std::getline(fin, headerLine);

    // Find weight column index
    int weightCol = -1;
    {
        std::istringstream hss(headerLine);
        std::string col;
        int idx = 0;
        while (std::getline(hss, col, ',')) {
            // Trim
            size_t a = col.find_first_not_of(" \t\r\n");
            size_t b = col.find_last_not_of(" \t\r\n");
            if (a != std::string::npos) col = col.substr(a, b - a + 1);
            if (col == "weight_evts_per_day") { weightCol = idx; break; }
            ++idx;
        }
    }
    if (weightCol < 0) {
        std::cerr << "Error: column 'weight_evts_per_day' not found in header.\n";
        return 1;
    }

    // Read all rows, separate nonzero-weight rows
    struct Row {
        std::string line;
        double weight;
    };

    std::vector<Row> allRows;
    std::vector<size_t> nonzeroIdx; // indices into allRows

    std::string line;
    while (std::getline(fin, line)) {
        // Trim trailing whitespace
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end == std::string::npos) continue;
        line = line.substr(0, end + 1);
        if (line.empty()) continue;

        // Parse weight column
        std::istringstream rss(line);
        std::string cell;
        double w = 0;
        for (int c = 0; c <= weightCol; ++c) {
            if (!std::getline(rss, cell, ',')) break;
        }
        try { w = std::stod(cell); } catch (...) { w = 0; }

        size_t rowIdx = allRows.size();
        allRows.push_back({line, w});
        if (w > 0) nonzeroIdx.push_back(rowIdx);
    }
    fin.close();

    int nTotal = allRows.size();
    int nNz = nonzeroIdx.size();

    // ── Subsample ───────────────────────────────────────────────
    double factor = 1.0;
    std::vector<size_t> keptIdx;

    if (nNz <= nTarget) {
        // Keep all nonzero rows
        keptIdx = nonzeroIdx;
    } else {
        // Shuffle and pick first nTarget
        std::mt19937_64 rng(seed);
        std::vector<size_t> shuffled = nonzeroIdx;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        keptIdx.assign(shuffled.begin(), shuffled.begin() + nTarget);
        factor = (double)nNz / nTarget;
    }

    // ── Compute weight sums ─────────────────────────────────────
    double wIn = 0;
    for (auto i : nonzeroIdx) wIn += allRows[i].weight;

    double wOut = 0;
    for (auto i : keptIdx) wOut += allRows[i].weight * factor;

    // ── Write output CSV ────────────────────────────────────────
    std::ofstream fout(outPath);
    if (!fout.is_open()) {
        std::cerr << "Error: cannot write to " << outPath << "\n";
        return 1;
    }
    fout << headerLine << "\n";

    for (auto idx : keptIdx) {
        if (factor == 1.0) {
            fout << allRows[idx].line << "\n";
        } else {
            // Rewrite the line with rescaled weight
            std::istringstream rss(allRows[idx].line);
            std::string cell;
            for (int c = 0; ; ++c) {
                if (!std::getline(rss, cell, ',')) break;
                if (c > 0) fout << ",";
                if (c == weightCol) {
                    double newW = allRows[idx].weight * factor;
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.6e", newW);
                    fout << buf;
                } else {
                    fout << cell;
                }
            }
            fout << "\n";
        }
    }
    fout.close();

    std::printf("%s: total=%d nonzero=%d kept=%zu reweight=%.4g w_in=%.6e w_out=%.6e\n",
                inPath.c_str(), nTotal, nNz, keptIdx.size(), factor, wIn, wOut);

    return 0;
}
