// Template gun origin from the beam MC: where particles of one species that reach
// the calo face were born (library truth vz), as a weighted z table for
// /damsa/setGunOriginTable. Vertices inside the tungsten are kept as they are;
// the generator starts those at the target exit face (it knows the geometry).
//
//   ./build/damsa_gun_origin --library output/calo_face_particles.root --species neutron \
//       --energy-MeV 100 --out output/gun_origin_neutron_E100.csv
//
// Selection: forward (pz > 0), E >= --min-MeV, |pdg| of the species, face E within
// +-window of --energy-MeV. Too few rows (< --min-rows) -> drop the energy window;
// still too few for a hadron -> pool all hadrons (all are made in the target).

#include "damsa_io.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    std::string library = "output/calo_face_particles.root", species, out;
    double energy = 0, window = 0.25, minE = 5, binMm = 5;
    std::size_t minRows = 100;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--library")    library = nx();
        else if (s == "--species")    species = nx();
        else if (s == "--energy-MeV") energy = std::stod(nx());
        else if (s == "--window")     window = std::stod(nx());
        else if (s == "--min-MeV")    minE = std::stod(nx());
        else if (s == "--bin-mm")     binMm = std::stod(nx());
        else if (s == "--min-rows")   minRows = std::stoul(nx());
        else if (s == "--out")        out = nx();
        else { species.clear(); break; }   // unknown flag -> usage
    }
    const std::map<std::string, std::set<int>> kPdg = {
        {"gamma", {22}}, {"e", {11}}, {"pi", {211}}, {"proton", {2212}}, {"neutron", {2112}}};
    const std::set<int> kHadrons = {2112, 2212, 211, 321, 130, 310};
    if (!kPdg.count(species) || energy <= 0 || out.empty()) {
        std::fprintf(stderr, "usage: %s --species gamma|e|pi|proton|neutron --energy-MeV E --out PATH\n"
                             "          [--library output/calo_face_particles.root] [--window 0.25]\n"
                             "          [--min-MeV 5] [--bin-mm 5] [--min-rows 100]\n", argv[0]);
        return 1;
    }

    const auto rows = damsa::io::ReadNTuple<damsa::io::ParticleRow>(library);
    auto pick = [&](const std::set<int>& pdgs, bool inWindow) {
        std::vector<const damsa::io::ParticleRow*> v;
        for (const auto& r : rows)
            if (r.pz > 0 && r.energy_MeV >= minE && pdgs.count(std::abs(r.pdg)) &&
                (!inWindow || std::abs(r.energy_MeV - energy) <= window * energy))
                v.push_back(&r);
        return v;
    };
    std::string used = "species, E window";
    auto sel = pick(kPdg.at(species), true);
    if (sel.size() < minRows) { sel = pick(kPdg.at(species), false); used = "species, all E"; }
    if (sel.size() < minRows && kHadrons.count(*kPdg.at(species).begin())) { sel = pick(kHadrons, false); used = "all hadrons, all E"; }
    if (sel.empty()) { std::fprintf(stderr, "Error: no %s rows in %s\n", species.c_str(), library.c_str()); return 1; }

    std::map<long, double> h;   // bin index -> weight
    for (const auto* r : sel) h[static_cast<long>(std::floor(r->vz_mm / binMm))] += r->weight;
    damsa::io::EnsureParentDir(out);
    std::ofstream f(out);
    f << "# species: " << species << "\n# energy_MeV: " << energy << "\n# selection: " << used
      << "\n# rows: " << sel.size() << "\n# library: " << library << "\nz_lo_mm,z_hi_mm,weight\n";
    for (const auto& [b, w] : h) f << b * binMm << "," << (b + 1) * binMm << "," << w << "\n";
    std::printf("%s E=%g MeV: %zu library rows (%s), z %.0f..%.0f mm -> %s\n", species.c_str(), energy,
                sel.size(), used.c_str(), h.begin()->first * binMm, (h.rbegin()->first + 1) * binMm, out.c_str());
    return 0;
}
