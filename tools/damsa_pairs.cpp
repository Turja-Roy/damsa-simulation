// Correlated gamma-gamma pairs at the calorimeter face (Random-2-photons plan
// §4.4, case 1). Two forward photons are one correlated pair when they share a
// parent track AND a creation vertex: pi0/eta -> gamma gamma (proc Decay) and
// e+e- -> gamma gamma (proc annihil). Brems photons of one electron share the
// parent but not the vertex, so they are not pairs here -- they are handled as
// same-source in damsa_pileup.
//
//   ./build/damsa_pairs --library output/calo_face_particles.root \
//        --n-library-electrons 1000000 [--min-photon-E 1] [--out-csv output/pairs_correlated.csv]
//
// Needs the truth columns (parentID, vx..): libraries written before them read
// parentID = -1 and the tool refuses them.

#include "damsa_io.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace io = damsa::io;

namespace {

const char* ProcName(int p)
{
    switch (p) {
        case io::Proc::Primary: return "primary";
        case io::Proc::eBrem:   return "eBrem";
        case io::Proc::Decay:   return "Decay";
        case io::Proc::Conv:    return "conv";
        case io::Proc::Compt:   return "compt";
        case io::Proc::Annihil: return "annihil";
        default:                return "other";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    std::string library, outCsv = "output/pairs_correlated.csv";
    long long nLib = 0;
    double minE = 1.0, vtxTol_mm = 1e-3;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--library")             library = nx();
        else if (s == "--n-library-electrons") nLib = std::stoll(nx());
        else if (s == "--min-photon-E")        minE = std::stod(nx());
        else if (s == "--out-csv")             outCsv = nx();
        else {
            std::fprintf(stderr, "Usage: %s --library PATH [--n-library-electrons N] "
                                 "[--min-photon-E E] [--out-csv PATH]\n", argv[0]);
            return 1;
        }
    }
    if (library.empty()) { std::fprintf(stderr, "Error: --library is required.\n"); return 1; }

    // (eventID, parentID) -> forward photons at the face
    std::map<std::pair<int, int>, std::vector<io::ParticleRow>> byParent;
    std::size_t nPh = 0;
    {
        io::NTupleReader<io::ParticleRow> r(library);
        for (std::uint64_t i = 0; i < r.Entries(); ++i) {
            const auto& p = r.At(i);
            if (p.pdg != 22 || p.pz <= 0 || p.energy_MeV < minE) continue;
            if (p.parentID < 0) {
                std::fprintf(stderr, "Error: %s has no truth columns (parentID = -1). "
                                     "Regenerate the library.\n", library.c_str());
                return 1;
            }
            byParent[{p.eventID, p.parentID}].push_back(p);
            ++nPh;
        }
    }

    io::EnsureParentDir(outCsv);
    std::ofstream out(outCsv);
    out << "eventID,parentID,proc,primaryID,E1_MeV,E2_MeV,m_gg_MeV,theta12_rad,"
           "sep_mm,z_asym,vz_mm,scattered1,scattered2,weight\n";

    // proc -> {pairs, pairs with neither photon scattered}
    std::map<int, std::pair<long long, long long>> perProc;
    std::vector<double> masses;
    for (const auto& [key, ph] : byParent) {
        for (std::size_t i = 0; i < ph.size(); ++i)
        for (std::size_t j = i + 1; j < ph.size(); ++j) {
            const auto& a = ph[i];
            const auto& b = ph[j];
            if (std::abs(a.vx_mm - b.vx_mm) > vtxTol_mm || std::abs(a.vy_mm - b.vy_mm) > vtxTol_mm ||
                std::abs(a.vz_mm - b.vz_mm) > vtxTol_mm) continue;   // different vertex
            double c = a.px * b.px + a.py * b.py + a.pz * b.pz;
            c = std::max(-1.0, std::min(1.0, c));
            const double m = std::sqrt(std::max(2.0 * a.energy_MeV * b.energy_MeV * (1.0 - c), 0.0));
            const double sep = std::hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm);
            const double z = std::abs(a.energy_MeV - b.energy_MeV) / (a.energy_MeV + b.energy_MeV);
            out << key.first << "," << key.second << "," << ProcName(a.proc) << ","
                << a.primaryID << "," << a.energy_MeV << "," << b.energy_MeV << "," << m << ","
                << std::acos(c) << "," << sep << "," << z << "," << a.vz_mm << ","
                << a.scattered << "," << b.scattered << "," << a.weight << "\n";
            auto& pp = perProc[a.proc];
            ++pp.first;
            if (a.scattered != 1 && b.scattered != 1) ++pp.second;
        }
    }

    std::printf("[lib ] %s: %zu forward photons >= %.3g MeV, %zu (event,parent) groups\n",
                library.c_str(), nPh, minE, byParent.size());
    std::printf("%-10s %12s %14s %16s\n", "proc", "pairs", "unscattered", "pairs/electron");
    for (const auto& [proc, n] : perProc)
        std::printf("%-10s %12lld %14lld %16s\n", ProcName(proc), n.first, n.second,
                    nLib > 0 ? std::to_string(double(n.first) / double(nLib)).c_str() : "-");
    if (perProc.empty()) std::printf("  no correlated pairs reach the calo face\n");
    std::printf("  wrote %s\n", outCsv.c_str());
    return 0;
}
