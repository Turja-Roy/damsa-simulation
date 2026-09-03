// Convert DAMSA data between CSV and RNTuple, in either direction.
//
// Transitional: once damsa_alp_signal and run.h write RNTuple directly, only the
// already-generated output/*.csv files need this. Streams line by line — the
// alp_decay_photons_ma*MeV.csv files are ~1 GB and must never be fully resident.
//
//   ./build/damsa_convert <in.csv>  <out.root> [--limit N]   # CSV  -> RNTuple
//   ./build/damsa_convert <in.root> <out.csv>  [--limit N]   # RNTuple -> CSV
//
// The reverse direction exists for verification: CSV -> RNTuple -> CSV must be
// byte-identical, which proves the RNTuple carries everything the CSV did. It
// re-emits through the same formatters in damsa_io.h that the simulation uses,
// so the two cannot drift.
//
// Schema is picked from the CSV header (or the RNTuple name). Columns absent
// from the file are left at their struct defaults; extra columns are ignored.

#include "damsa_io.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace damsa::io;

namespace {

std::vector<std::string> SplitLine(const std::string& s)
{
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        const auto c = s.find(',', start);
        if (c == std::string::npos) { out.push_back(s.substr(start)); break; }
        out.push_back(s.substr(start, c - start));
        start = c + 1;
    }
    return out;
}

// Position of each schema field in this file's header, or npos if the file
// lacks it (e.g. legacy 9-column decay CSVs with no decay_z_m).
template <class Arr>
std::vector<std::size_t> MapColumns(const Arr& fields,
                                    const std::vector<std::string>& header)
{
    std::vector<std::size_t> idx;
    for (const auto& [name, mem] : fields) {
        std::size_t found = std::string::npos;
        for (std::size_t i = 0; i < header.size(); ++i)
            if (header[i] == name) { found = i; break; }
        idx.push_back(found);
    }
    return idx;
}

template <class T>
std::size_t Convert(const std::string& inPath, const std::string& outPath, long limit)
{
    std::ifstream f(inPath);
    if (!f.is_open()) throw std::runtime_error("cannot open " + inPath);

    std::string line;
    std::vector<std::string> header;
    while (std::getline(f, line)) {
        const std::string t = Trim(line);
        if (t.empty() || t[0] == '#') continue;
        for (auto& h : SplitLine(t)) header.push_back(Trim(h));
        break;
    }

    const auto ints = Schema<T>::Ints();
    const auto dbls = Schema<T>::Dbls();
    const auto intIdx = MapColumns(ints, header);
    const auto dblIdx = MapColumns(dbls, header);

    for (std::size_t i = 0; i < dbls.size(); ++i)
        if (dblIdx[i] == std::string::npos)
            std::fprintf(stderr, "  note: column '%s' absent, using default\n", dbls[i].first);

    NTupleWriter<T> w(outPath);
    std::size_t n = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto cells = SplitLine(line);
        T row;
        for (std::size_t i = 0; i < ints.size(); ++i) {
            const auto c = intIdx[i];
            if (c < cells.size()) { try { row.*(ints[i].second) = std::stoi(cells[c]); } catch (...) {} }
        }
        for (std::size_t i = 0; i < dbls.size(); ++i) {
            const auto c = dblIdx[i];
            if (c < cells.size()) { try { row.*(dbls[i].second) = std::stod(cells[c]); } catch (...) {} }
        }
        w.Fill(row);
        if (++n % 500000 == 0) std::fprintf(stderr, "  %zu rows...\n", n);
        if (limit > 0 && static_cast<long>(n) >= limit) break;
    }
    return n;
}

// ── RNTuple -> CSV ──────────────────────────────────────────────────────────

template <class T>
std::size_t ToCsv(const std::string& inPath, const std::string& outPath,
                  const char* header, void (*emit)(std::ostream&, const T&),
                  long limit)
{
    auto reader = ROOT::RNTupleReader::Open(Schema<T>::kName, inPath);
    std::uint64_t n = reader->GetNEntries();
    if (limit > 0 && n > static_cast<std::uint64_t>(limit)) n = limit;

    // Views are created once, not per entry.
    std::vector<ROOT::RNTupleView<int>>    intViews;
    std::vector<ROOT::RNTupleView<double>> dblViews;
    for (const auto& [name, mem] : Schema<T>::Ints())
        intViews.push_back(reader->template GetView<int>(name));
    for (const auto& [name, mem] : Schema<T>::Dbls())
        dblViews.push_back(reader->template GetView<double>(name));

    EnsureParentDir(outPath);
    std::ofstream out(outPath);
    if (!out.is_open()) throw std::runtime_error("cannot write " + outPath);
    out << header << "\n";

    for (std::uint64_t i = 0; i < n; ++i) {
        T row;
        std::size_t k = 0;
        for (const auto& [name, mem] : Schema<T>::Ints()) row.*mem = intViews[k++](i);
        k = 0;
        for (const auto& [name, mem] : Schema<T>::Dbls()) row.*mem = dblViews[k++](i);
        emit(out, row);
        if ((i + 1) % 500000 == 0) std::fprintf(stderr, "  %llu rows...\n",
                                                (unsigned long long)(i + 1));
    }
    return n;
}

// Which schema does this file use? Decided by a column only that schema has.
std::optional<std::string> SniffSchema(const std::string& path)
{
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        const std::string t = Trim(line);
        if (t.empty() || t[0] == '#') continue;
        if (t.find("E1_MeV") != std::string::npos)     return "alp";
        if (t.find("pi0TrackID") != std::string::npos) return "pi0";
        if (t.find("energy_MeV") != std::string::npos) return "particle";
        return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <in.csv> <out.root> [--limit N]\n", argv[0]);
        return 1;
    }
    const std::string in = argv[1], out = argv[2];
    long limit = 0;
    for (int i = 3; i < argc; ++i)
        if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc) limit = std::atol(argv[++i]);

    const bool reverse = in.size() > 5 && in.substr(in.size() - 5) == ".root";

    std::optional<std::string> schema;
    if (reverse) {
        for (const char* s : {"alp", "pi0", "particle"}) {
            const char* nt = (std::strcmp(s, "alp") == 0)   ? Schema<AlpDecayRow>::kName
                           : (std::strcmp(s, "pi0") == 0)   ? Schema<Pi0Row>::kName
                                                            : Schema<ParticleRow>::kName;
            try { ROOT::RNTupleReader::Open(nt, in); schema = s; break; } catch (...) {}
        }
        if (!schema) { std::fprintf(stderr, "error: no known RNTuple in %s\n", in.c_str()); return 1; }
    } else {
        schema = SniffSchema(in);
        if (!schema) { std::fprintf(stderr, "error: unrecognized CSV header in %s\n", in.c_str()); return 1; }
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::size_t n = 0;
    if (reverse) {
        if      (*schema == "alp") n = ToCsv<AlpDecayRow>(in, out, kAlpDecayCsvHeader, WriteAlpDecayCsvRow, limit);
        else if (*schema == "pi0") n = ToCsv<Pi0Row>(in, out, kPi0CsvHeader, WritePi0CsvRow, limit);
        else                       n = ToCsv<ParticleRow>(in, out, kParticleCsvHeader, WriteParticleCsvRow, limit);
    }
    else if (*schema == "alp")      n = Convert<AlpDecayRow>(in, out, limit);
    else if (*schema == "pi0")      n = Convert<Pi0Row>(in, out, limit);
    else                            n = Convert<ParticleRow>(in, out, limit);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const auto inSz  = std::filesystem::file_size(in);
    const auto outSz = std::filesystem::file_size(out);
    std::printf("%s [%s] %zu rows in %.1fs\n", in.c_str(), schema->c_str(), n, secs);
    std::printf("  %.1f MB -> %.1f MB  (%.2fx)\n",
                inSz / 1e6, outSz / 1e6, double(inSz) / double(outSz));
    return 0;
}
