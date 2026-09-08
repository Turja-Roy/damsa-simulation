#ifndef DAMSA_IO_H
#define DAMSA_IO_H

// Shared I/O for DAMSA: RNTuple storage schemas + the one CSV reader.
//
// Deliberately Geant4-free so the standalone ROOT-only tools (flux_converter,
// subsample_alp_csv, snr_separability) read exactly the same files the Geant4
// executables write. Include this, not a hand-rolled parser — there used to be
// three separate copies of the CSV reader in this repo.
//
// Storage rationale: the ASCII CSVs reached 7.6 GB in output/, ~1 GB per
// alp_decay_photons_ma*MeV.csv (4M rows x 10 float64 at 18 significant digits).
// RNTuple is columnar, compressed, and already available via ROOT.
//
// Storage is TTree, not RNTuple. RNTuple is ~10% smaller, but its stable
// ROOT:: API only exists from 6.36; the cluster runs root/6.26, where RNTuple
// is ROOT::Experimental:: with a different API and an unstable on-disk format.
// TTree reads and writes identically from 6.26 through 6.40, and RDataFrame,
// rootls and TBrowser all work on it. One backend everywhere beats two behind
// version guards for a 10% file-size difference.
//
// Measured on a 300k-row slice of alp_decay_photons_ma1MeV.csv (75.6 MB ASCII):
//     TTree   float64 + zstd-5   20.5 MB   3.7x   <- what this uses
//     TTree   float64 + lzma-9   20.1 MB   3.8x   (much slower, not worth it)
//     RNTuple float64 + zstd-5   18.6 MB   4.1x   (needs ROOT >= 6.36)
//     RNTuple float32 + zstd-5    8.8 MB   8.6x
// Fields are float64 so the CSV-vs-RNTuple cross-check is exact rather than
// tolerance-limited. Direction cosines are unit vectors and energies are MC
// quantities, so float32 is physically ample and would halve the size again.
// ponytail: float64 for an exact cross-check; switch the Dbls() fields to float
// once Phase 1 has passed, if 4.1x is not enough.

#include <TFile.h>
#include <TTree.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace damsa::io {

// ── Paths ───────────────────────────────────────────────────────────────────
// Recursive, unlike the mkdir("output", 0755) the CSV writers used to call.
// That non-recursive call is why an output prefix containing a slash silently
// produced no file at all (see CLAUDE.md, "Critical gotcha").
inline void EnsureParentDir(const std::string& path)
{
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

// ── Row schemas ─────────────────────────────────────────────────────────────

// Replaces all four CSV writers in FluxData.h (WriteCSV, WritePhotonFluxCSV,
// WriteBackgroundCSV, WriteCaloFaceCSV) — they emitted identical columns.
// The background view is a pdg != 22 filter at read time, not a separate file.
struct ParticleRow {
    int    pdg = 0, trackID = 0, eventID = 0;
    double energy_MeV = 0, time_ns = 0;
    double x_mm = 0, y_mm = 0, z_mm = 0;
    double px = 0, py = 0, pz = 0;
    double weight = 1.0;
};

struct Pi0Row {
    int    eventID = 0, pi0TrackID = 0;
    double vx_mm = 0, vy_mm = 0, vz_mm = 0;
    int    gamma1TrackID = 0;
    double e1_MeV = 0, px1 = 0, py1 = 0, pz1 = 0;
    int    gamma2TrackID = 0;
    double e2_MeV = 0, px2 = 0, py2 = 0, pz2 = 0;
    double openingAngle_deg = 0, pi0Energy_MeV = 0;
    int    gamma1AtCalo = 0, gamma2AtCalo = 0;
    int    gamma1GeomAccept = 0, gamma2GeomAccept = 0;
    double caloEnergyMeV = 0;
};

// The ~1 GB files. Momenta are direction components; weight is events/day.
struct AlpDecayRow {
    double E1 = 0, px1 = 0, py1 = 0, pz1 = 0;
    double E2 = 0, px2 = 0, py2 = 0, pz2 = 0;
    double weight = 0;    // weight_evts_per_day
    double decayZ_m = 0;  // decay vertex z offset from target centre [m]
};

// ── Schema traits ───────────────────────────────────────────────────────────
// One declaration per row type; the generic writer/reader below do the rest.
// Column names match the CSV headers they replace, so the cross-check against
// the old files is a straight name-to-name comparison.

template <class T> struct Schema;

template <> struct Schema<ParticleRow> {
    static constexpr const char* kName = "particles";
    static constexpr auto Ints() {
        return std::array{ std::pair{"pdg",     &ParticleRow::pdg},
                           std::pair{"trackID", &ParticleRow::trackID},
                           std::pair{"eventID", &ParticleRow::eventID} };
    }
    static constexpr auto Dbls() {
        return std::array{ std::pair{"energy_MeV", &ParticleRow::energy_MeV},
                           std::pair{"time_ns",    &ParticleRow::time_ns},
                           std::pair{"x_mm",       &ParticleRow::x_mm},
                           std::pair{"y_mm",       &ParticleRow::y_mm},
                           std::pair{"z_mm",       &ParticleRow::z_mm},
                           std::pair{"px",         &ParticleRow::px},
                           std::pair{"py",         &ParticleRow::py},
                           std::pair{"pz",         &ParticleRow::pz},
                           std::pair{"weight",     &ParticleRow::weight} };
    }
};

template <> struct Schema<Pi0Row> {
    static constexpr const char* kName = "pi0";
    static constexpr auto Ints() {
        return std::array{ std::pair{"eventID",          &Pi0Row::eventID},
                           std::pair{"pi0TrackID",       &Pi0Row::pi0TrackID},
                           std::pair{"gamma1TrackID",    &Pi0Row::gamma1TrackID},
                           std::pair{"gamma2TrackID",    &Pi0Row::gamma2TrackID},
                           std::pair{"gamma1AtCalo",     &Pi0Row::gamma1AtCalo},
                           std::pair{"gamma2AtCalo",     &Pi0Row::gamma2AtCalo},
                           std::pair{"gamma1GeomAccept", &Pi0Row::gamma1GeomAccept},
                           std::pair{"gamma2GeomAccept", &Pi0Row::gamma2GeomAccept} };
    }
    static constexpr auto Dbls() {
        return std::array{ std::pair{"vx_mm",            &Pi0Row::vx_mm},
                           std::pair{"vy_mm",            &Pi0Row::vy_mm},
                           std::pair{"vz_mm",            &Pi0Row::vz_mm},
                           std::pair{"e1_MeV",           &Pi0Row::e1_MeV},
                           std::pair{"px1",              &Pi0Row::px1},
                           std::pair{"py1",              &Pi0Row::py1},
                           std::pair{"pz1",              &Pi0Row::pz1},
                           std::pair{"e2_MeV",           &Pi0Row::e2_MeV},
                           std::pair{"px2",              &Pi0Row::px2},
                           std::pair{"py2",              &Pi0Row::py2},
                           std::pair{"pz2",              &Pi0Row::pz2},
                           std::pair{"openingAngle_deg", &Pi0Row::openingAngle_deg},
                           std::pair{"pi0Energy_MeV",    &Pi0Row::pi0Energy_MeV},
                           std::pair{"caloEnergyMeV",    &Pi0Row::caloEnergyMeV} };
    }
};

template <> struct Schema<AlpDecayRow> {
    static constexpr const char* kName = "alp_decays";
    static constexpr auto Ints() { return std::array<std::pair<const char*, int AlpDecayRow::*>, 0>{}; }
    static constexpr auto Dbls() {
        return std::array{ std::pair{"E1_MeV",              &AlpDecayRow::E1},
                           std::pair{"px1",                 &AlpDecayRow::px1},
                           std::pair{"py1",                 &AlpDecayRow::py1},
                           std::pair{"pz1",                 &AlpDecayRow::pz1},
                           std::pair{"E2_MeV",              &AlpDecayRow::E2},
                           std::pair{"px2",                 &AlpDecayRow::px2},
                           std::pair{"py2",                 &AlpDecayRow::py2},
                           std::pair{"pz2",                 &AlpDecayRow::pz2},
                           std::pair{"weight_evts_per_day", &AlpDecayRow::weight},
                           std::pair{"decay_z_m",           &AlpDecayRow::decayZ_m} };
    }
};

// ── Generic TTree writer ────────────────────────────────────────────────────
// Streaming: Fill() one row at a time so multi-million-row outputs never need
// to exist in memory all at once.
//
// Compression 505 = ZSTD level 5 (algorithm*100 + level). Level 9 and LZMA were
// measured and buy under 2%, at much higher CPU cost.

template <class T>
class NTupleWriter {
public:
    explicit NTupleWriter(const std::string& path,
                          const std::string& treeName = Schema<T>::kName,
                          int compression = 505)
    {
        EnsureParentDir(path);
        fFile = TFile::Open(path.c_str(), "RECREATE", "", compression);
        if (!fFile || fFile->IsZombie())
            throw std::runtime_error("NTupleWriter: cannot create " + path);

        fTree = new TTree(treeName.c_str(), treeName.c_str());
        fTree->SetDirectory(fFile);
        for (const auto& [name, mem] : Schema<T>::Ints()) fTree->Branch(name, &(fRow.*mem));
        for (const auto& [name, mem] : Schema<T>::Dbls()) fTree->Branch(name, &(fRow.*mem));
    }

    void Fill(const T& row)
    {
        fRow = row;               // branches point into fRow
        fTree->Fill();
        ++fEntries;
    }

    std::size_t Entries() const { return fEntries; }

    // Flush and close. Safe to call more than once; the destructor calls it.
    void Finish()
    {
        if (!fFile) return;
        fFile->cd();
        fTree->Write();
        fFile->Close();
        delete fFile;             // deletes the tree it owns
        fFile = nullptr;
        fTree = nullptr;
    }
    ~NTupleWriter() { Finish(); }

    NTupleWriter(const NTupleWriter&) = delete;
    NTupleWriter& operator=(const NTupleWriter&) = delete;

private:
    TFile* fFile = nullptr;
    TTree* fTree = nullptr;
    T      fRow{};
    std::size_t fEntries = 0;
};

// One-shot write for data already held in a vector (what run.h has).
template <class T>
inline void WriteNTuple(const std::string& path, const std::vector<T>& rows,
                        const std::string& ntupleName = Schema<T>::kName)
{
    NTupleWriter<T> w(path, ntupleName);
    for (const auto& r : rows) w.Fill(r);
}

// ── Generic TTree reader ────────────────────────────────────────────────────
// Streaming reader, so the caller can process multi-million-row files without
// materialising them. ReadNTuple() below is the load-everything convenience.

template <class T>
class NTupleReader {
public:
    explicit NTupleReader(const std::string& path,
                          const std::string& treeName = Schema<T>::kName)
    {
        fFile = TFile::Open(path.c_str(), "READ");
        if (!fFile || fFile->IsZombie())
            throw std::runtime_error("NTupleReader: cannot open " + path);
        fTree = dynamic_cast<TTree*>(fFile->Get(treeName.c_str()));
        if (!fTree)
            throw std::runtime_error("NTupleReader: no tree '" + treeName + "' in " + path);

        for (const auto& [name, mem] : Schema<T>::Ints()) fTree->SetBranchAddress(name, &(fRow.*mem));
        for (const auto& [name, mem] : Schema<T>::Dbls()) fTree->SetBranchAddress(name, &(fRow.*mem));
    }

    std::uint64_t Entries() const { return static_cast<std::uint64_t>(fTree->GetEntries()); }

    const T& At(std::uint64_t i) { fTree->GetEntry(static_cast<Long64_t>(i)); return fRow; }

    ~NTupleReader() { if (fFile) { fFile->Close(); delete fFile; } }

    NTupleReader(const NTupleReader&) = delete;
    NTupleReader& operator=(const NTupleReader&) = delete;

private:
    TFile* fFile = nullptr;
    TTree* fTree = nullptr;
    T      fRow{};
};

template <class T>
inline std::vector<T> ReadNTuple(const std::string& path,
                                 const std::string& treeName = Schema<T>::kName)
{
    NTupleReader<T> r(path, treeName);
    const auto n = r.Entries();
    std::vector<T> rows;
    rows.reserve(n);
    for (std::uint64_t i = 0; i < n; ++i) rows.push_back(r.At(i));
    return rows;
}

// Entry count without binding any schema, so it works for any tree.
// Returns -1 if the file or tree is missing.
inline long long NTupleEntries(const std::string& path, const std::string& treeName)
{
    std::unique_ptr<TFile> f(TFile::Open(path.c_str(), "READ"));
    if (!f || f->IsZombie()) return -1;
    auto* t = dynamic_cast<TTree*>(f->Get(treeName.c_str()));
    return t ? t->GetEntries() : -1;
}

// Which of our schemas does this file hold? Empty if none.
inline std::string SniffTree(const std::string& path)
{
    for (const char* n : {Schema<AlpDecayRow>::kName, Schema<Pi0Row>::kName,
                          Schema<ParticleRow>::kName})
        if (NTupleEntries(path, n) >= 0) return n;
    return "";
}

// ── Legacy CSV emission ─────────────────────────────────────────────────────
// The exact text format the FluxData.h / pi0DecayData.h writers produced, kept
// in ONE place so the simulation writer and the verification tool cannot drift.
//
// These formats are lossy: particles at setprecision(6), pi0 at (4). So the
// migration check is not "do the values match" (the RNTuple holds more digits)
// but "does re-emitting the RNTuple reproduce the CSV byte for byte".

inline constexpr const char* kParticleCsvHeader =
    "pdg,energy_MeV,time_ns,x_mm,y_mm,z_mm,px,py,pz,weight,trackID,eventID";

inline void WriteParticleCsvRow(std::ostream& o, const ParticleRow& p)
{
    o << p.pdg << ","
      << std::scientific << std::setprecision(6)
      << p.energy_MeV << ","
      << p.time_ns << ","
      << p.x_mm << ","
      << p.y_mm << ","
      << p.z_mm << ","
      << p.px << ","
      << p.py << ","
      << p.pz << ","
      << p.weight << ","
      << p.trackID << ","
      << p.eventID << "\n";
}

inline constexpr const char* kPi0CsvHeader =
    "eventID,pi0TrackID,vx_mm,vy_mm,vz_mm,"
    "gamma1TrackID,e1_MeV,px1,py1,pz1,"
    "gamma2TrackID,e2_MeV,px2,py2,pz2,"
    "openingAngle_deg,pi0Energy_MeV,"
    "gamma1AtCalo,gamma2AtCalo,"
    "gamma1GeomAccept,gamma2GeomAccept,"
    "caloEnergyMeV";

inline void WritePi0CsvRow(std::ostream& o, const Pi0Row& d)
{
    o << d.eventID    << ","
      << d.pi0TrackID << ","
      << std::scientific << std::setprecision(4)
      << d.vx_mm << "," << d.vy_mm << "," << d.vz_mm << ","
      << d.gamma1TrackID << ","
      << d.e1_MeV << "," << d.px1 << "," << d.py1 << "," << d.pz1 << ","
      << d.gamma2TrackID << ","
      << d.e2_MeV << "," << d.px2 << "," << d.py2 << "," << d.pz2 << ","
      << std::fixed << std::setprecision(4)
      << d.openingAngle_deg << ","
      << std::scientific
      << d.pi0Energy_MeV << ","
      << d.gamma1AtCalo << "," << d.gamma2AtCalo << ","
      << d.gamma1GeomAccept << "," << d.gamma2GeomAccept << ","
      << std::fixed << std::setprecision(4)
      << d.caloEnergyMeV << "\n";
}

inline constexpr const char* kAlpDecayCsvHeader =
    "E1_MeV,px1,py1,pz1,E2_MeV,px2,py2,pz2,weight_evts_per_day,decay_z_m";

// alp_signal_pipeline.py wrote these via numpy at full float64 repr.
inline void WriteAlpDecayCsvRow(std::ostream& o, const AlpDecayRow& r)
{
    o << std::scientific << std::setprecision(18)
      << r.E1 << "," << r.px1 << "," << r.py1 << "," << r.pz1 << ","
      << r.E2 << "," << r.px2 << "," << r.py2 << "," << r.pz2 << ","
      << r.weight << "," << r.decayZ_m << "\n";
}

// ── CSV (for the files that stay CSV, by design) ────────────────────────────
// Promoted from flux_converter.cpp, which snr_separability.cpp had copied
// verbatim. Column-indexed rather than a std::map per row — the old shape cost
// one map allocation per row on multi-million-row files.

inline std::string Trim(const std::string& s)
{
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

struct CsvTable {
    std::vector<std::string>          header;
    std::vector<std::string>          comments;   // '#' lines, verbatim
    std::vector<std::vector<double>>  rows;
    std::map<std::string, std::size_t> index;

    bool has(const std::string& name) const { return index.count(name) > 0; }
    std::size_t size() const { return rows.size(); }

    double get(std::size_t row, const std::string& name, double fallback = 0.0) const
    {
        const auto it = index.find(name);
        if (it == index.end() || it->second >= rows[row].size()) return fallback;
        return rows[row][it->second];
    }

    // One view per row, for the filter-and-pass-along call sites.
    std::vector<struct CsvRowView> views() const;

    // Value of a "# Key: value" header line, empty if absent.
    std::string comment(const std::string& key) const
    {
        for (const auto& c : comments) {
            const auto p = c.find(key);
            if (p != std::string::npos) {
                const auto colon = c.find(':', p + key.size() - 1);
                if (colon != std::string::npos) return Trim(c.substr(colon + 1));
            }
        }
        return "";
    }
};

// Borrowed reference to one row. Keeps the row-oriented call sites that the old
// per-row std::map served, without the per-row allocation.
// Lifetime: views borrow the CsvTable; it must outlive them.
struct CsvRowView {
    const CsvTable* table = nullptr;
    std::size_t     row   = 0;

    double get(const std::string& name, double fallback = 0.0) const
    {
        return table->get(row, name, fallback);
    }
    bool has(const std::string& name) const { return table->has(name); }
};

inline CsvTable ReadCsv(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("damsa::io::ReadCsv: cannot open " + path);

    CsvTable csv;
    std::string line;
    bool haveHeader = false;

    while (std::getline(f, line)) {
        const std::string t = Trim(line);
        if (t.empty()) continue;
        if (t[0] == '#') { csv.comments.push_back(t); continue; }

        std::istringstream ss(t);
        std::string cell;

        if (!haveHeader) {
            while (std::getline(ss, cell, ',')) csv.header.push_back(Trim(cell));
            for (std::size_t i = 0; i < csv.header.size(); ++i) csv.index[csv.header[i]] = i;
            haveHeader = true;
            continue;
        }

        std::vector<double> row;
        row.reserve(csv.header.size());
        while (std::getline(ss, cell, ',')) {
            try { row.push_back(std::stod(cell)); }
            catch (...) { row.push_back(0.0); }
        }
        csv.rows.push_back(std::move(row));
    }
    return csv;
}

// ── Bremsstrahlung flux ─────────────────────────────────────────────────────
// Stays CSV on purpose: 175 KB, human-readable, and its '#' header carries the
// beam-mode provenance that must never be re-applied downstream (CLAUDE.md).
// The file has no column-header row, only '#' comments then bare numbers, so it
// gets its own reader rather than contorting ReadCsv.

struct BremsFlux {
    std::vector<double> energy_MeV;
    std::vector<double> rate_per_s;
    std::string beamMode;             // empty if the header predates the stamp
    double      beamCurrent_A = 0.0;  // 0 if absent

    std::size_t size() const { return energy_MeV.size(); }
    double totalRate() const
    {
        double s = 0.0;
        for (double r : rate_per_s) s += r;
        return s;
    }
};

inline BremsFlux ReadBremsFlux(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("damsa::io::ReadBremsFlux: cannot open " + path);

    BremsFlux flux;
    std::string line;
    while (std::getline(f, line)) {
        const std::string t = Trim(line);
        if (t.empty()) continue;

        if (t[0] == '#') {
            if (t.find("Beam mode:") != std::string::npos)
                flux.beamMode = Trim(t.substr(t.find("Beam mode:") + 10));
            else if (t.find("Beam current [A]:") != std::string::npos)
                try { flux.beamCurrent_A = std::stod(Trim(t.substr(t.find("Beam current [A]:") + 17))); }
                catch (...) {}
            continue;
        }

        std::istringstream ss(t);
        std::string a, b;
        if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) continue;
        double e = 0, r = 0;
        try { e = std::stod(a); r = std::stod(b); } catch (...) { continue; }
        // Same filter as load_geant4_brems_flux() in alp_signal_pipeline.py —
        // keep it identical or the cross-check drifts.
        if (e > 0 && r > 0) { flux.energy_MeV.push_back(e); flux.rate_per_s.push_back(r); }
    }

    if (flux.energy_MeV.empty())
        throw std::runtime_error("damsa::io::ReadBremsFlux: no usable data in " + path);
    return flux;
}

inline std::vector<CsvRowView> CsvTable::views() const
{
    std::vector<CsvRowView> v;
    v.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) v.push_back(CsvRowView{this, i});
    return v;
}

}  // namespace damsa::io

#endif  // DAMSA_IO_H
