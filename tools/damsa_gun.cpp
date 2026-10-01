// SM two-photon particle gun for validating the calo reconstruction
// (Calo-Reco-Explained §2.13): a parent of mass m and energy E along +z decays
// isotropically to gamma gamma at a vertex uniform in [z-min, z-max] (metres
// from the target centre, same convention as AlpDecayRow::decayZ_m).
//
//   ./build/damsa_gun --mass-MeV 135 --energy-MeV 1000 --n 5000 --out output/gun_pi0_E1000.root
//   ./build/damsa_alp_inject output/gun_pi0_E1000.root macros/run_alp.mac
//
// pi0 = 134.977 MeV, eta = 547.862 MeV. Defaults put the vertex in the vacuum
// decay chamber (target rear face + 1 cm .. VDC end), where a real SM pair
// would have to originate to reach the calo with both photons.

#include "damsa_io.h"
#include "decay2body.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    double mass = 134.977, energy = 1000, zMin = 0.06, zMax = 0.35;
    long n = 5000;
    std::uint64_t seed = 1;
    std::string out = "output/gun.root";
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--mass-MeV")   mass = std::stod(nx());
        else if (s == "--energy-MeV") energy = std::stod(nx());
        else if (s == "--n")          n = std::stol(nx());
        else if (s == "--z-min-m")    zMin = std::stod(nx());
        else if (s == "--z-max-m")    zMax = std::stod(nx());
        else if (s == "--seed")       seed = std::stoull(nx());
        else if (s == "--out")        out = nx();
        else {
            std::fprintf(stderr, "usage: %s [--mass-MeV 134.977] [--energy-MeV 1000] [--n 5000]\n"
                                 "          [--z-min-m 0.06] [--z-max-m 0.35] [--seed 1] [--out PATH]\n", argv[0]);
            return 1;
        }
    }
    if (energy <= mass) { std::fprintf(stderr, "Error: energy must exceed the mass.\n"); return 1; }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uz(zMin, zMax);
    const damsa::alp::FourVector parent{energy, 0, 0, std::sqrt(energy * energy - mass * mass)};

    std::vector<damsa::io::AlpDecayRow> rows;
    rows.reserve(n);
    for (long k = 0; k < n; ++k) {
        damsa::alp::FourVector g1, g2;
        damsa::alp::Decay2BodyMassless(parent, mass, rng, g1, g2);
        damsa::io::AlpDecayRow r;
        const double p1 = std::sqrt(g1.px * g1.px + g1.py * g1.py + g1.pz * g1.pz);
        const double p2 = std::sqrt(g2.px * g2.px + g2.py * g2.py + g2.pz * g2.pz);
        r.E1 = g1.e; r.px1 = g1.px / p1; r.py1 = g1.py / p1; r.pz1 = g1.pz / p1;   // unit directions
        r.E2 = g2.e; r.px2 = g2.px / p2; r.py2 = g2.py / p2; r.pz2 = g2.pz / p2;
        r.weight = 1.0;
        r.decayZ_m = uz(rng);
        rows.push_back(r);
    }
    damsa::io::EnsureParentDir(out);
    damsa::io::WriteNTuple(out, rows);
    std::printf("wrote %ld %s-like decays (m=%.3f MeV, E=%.0f MeV, z %.2f-%.2f m) -> %s\n",
                n, mass < 300 ? "pi0" : "eta", mass, energy, zMin, zMax, out.c_str());
    return 0;
}
