// Accidental gamma-gamma coincidence rate from beam pileup.
// Replaces scripts/pipeline/pileup_overlay.py.
//
//   ./build/damsa_pileup --library output/pi0_decays.root \
//        --n-library-electrons 1000000 --beam-mode lesa
//   ./build/damsa_pileup --library output/calo_face_particles.root --calo-face ...
//
// Builds synthetic readout gates by drawing library electrons with replacement,
// then counts gamma-gamma pairs from DIFFERENT sources whose invariant mass
// lands in the ALP window. Those are the fakes a real ALP search must survive.

#include "damsa_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

struct BeamSpec { double bunchCharge, kickerRate; int bunchesPerKick; double spacing_s; };

// Matches DamsaConfig::BeamSpecFor in src/config/damsa_config.h.
const std::map<std::string, BeamSpec> kBeamModes = {
    {"dark",        {0.07,     929e3, 100, 5.4e-9 }},
    {"lesa",        {4200.0,   929e3, 18,  26.9e-9}},
    {"xleap",       {167000.0, 929e3, 1,   1.08e-6}},
    {"interleaved", {6.2e8,    100.0, 1,   10e-3  }},
};

struct Photon { double E, ux, uy, uz; long long tag; };
using PhotonSet = std::vector<Photon>;

struct Args {
    std::string library, outCsv = "output/pileup_summary.csv";
    long long nLibraryElectrons = 0;
    bool caloFace = false, useGeomAccept = false, fixedOcc = false;
    double minPhotonE = 1.0, gate_ns = 1000.0;
    double mass_MeV = 100.0, massWindow_MeV = 20.0, threshold_MeV = 5.0;
    long long nTrials = 100000, maxPairs = 200000, maxPhotonsPerGate = 20000;
    std::string beamMode = "dark";
    std::uint64_t seed = 12345;
};

// pileup_overlay.py:85. Photons from pi0 decays that reached the calorimeter.
// Each pi0 is one source: its own two gammas are correlated, not accidental.
std::vector<PhotonSet> LoadPi0Library(const std::string& path, bool useGeomAccept,
                                      std::size_t& nPhotons)
{
    const auto rows = io::ReadNTuple<io::Pi0Row>(path);
    std::map<int, PhotonSet> perElectron;
    nPhotons = 0;

    for (const auto& d : rows) {
        const int a1 = useGeomAccept ? d.gamma1GeomAccept : d.gamma1AtCalo;
        const int a2 = useGeomAccept ? d.gamma2GeomAccept : d.gamma2AtCalo;
        // Any injective (eventID, pi0TrackID) key works; only equality is used.
        const long long tag = (static_cast<long long>(d.eventID) << 32)
                            ^ static_cast<long long>(d.pi0TrackID);
        auto& b = perElectron[d.eventID];
        if (a1 == 1) { b.push_back({d.e1_MeV, d.px1, d.py1, d.pz1, tag}); ++nPhotons; }
        if (a2 == 1) { b.push_back({d.e2_MeV, d.px2, d.py2, d.pz2, tag}); ++nPhotons; }
    }

    std::vector<PhotonSet> sets;
    for (auto& [evt, v] : perElectron) if (!v.empty()) sets.push_back(std::move(v));
    return sets;
}

// pileup_overlay.py:135. Every photon that crossed the calo face -- the full SM
// fake pool. All photons of one electron share its eventID as their tag: pairs
// inside a single shower are correlated, not accidental.
std::vector<PhotonSet> LoadCaloFaceLibrary(const std::string& path, double minE,
                                           std::size_t& nPhotons)
{
    const auto rows = io::ReadNTuple<io::ParticleRow>(path);
    std::map<int, PhotonSet> perElectron;
    nPhotons = 0;

    for (const auto& p : rows) {
        if (p.pdg != 22 || p.energy_MeV < minE) continue;
        perElectron[p.eventID].push_back(
            {p.energy_MeV, p.px, p.py, p.pz, static_cast<long long>(p.eventID)});
        ++nPhotons;
    }

    std::vector<PhotonSet> sets;
    for (auto& [evt, v] : perElectron) if (!v.empty()) sets.push_back(std::move(v));
    return sets;
}

// pileup_overlay.py:65, mirroring gate_sampler.h.
// n_bunches = clamp(floor(gate/spacing), 1, bunches_per_kick);
// a sum of n iid Poisson(q) is Poisson(n*q), so one draw per gate suffices.
template <class RNG>
long long SampleGateOccupancy(const BeamSpec& s, double gate_s, bool poisson, RNG& rng)
{
    int nb = static_cast<int>(std::floor(gate_s / s.spacing_s));
    nb = std::max(1, std::min(nb, s.bunchesPerKick));
    if (!poisson) return static_cast<long long>(std::llround(nb * s.bunchCharge));
    std::poisson_distribution<long long> pd(nb * s.bunchCharge);
    return pd(rng);
}

// pileup_overlay.py:181. A pair is accidental only if it came from different
// sources: different tag OR different draw. The library is sampled WITH
// replacement, so one library electron drawn twice stands for two independent
// real electrons and its photons must pair as accidentals despite equal tags.
template <class RNG>
double CountFakesInGate(const std::vector<Photon>& ph, const std::vector<int>& draw,
                        double massLo, double massHi, double esumThr,
                        long long maxPairs, RNG& rng)
{
    const std::size_t K = ph.size();
    if (K < 2) return 0.0;

    const long long nAll = static_cast<long long>(K) * (K - 1) / 2;
    double scale = 1.0;
    std::vector<std::pair<std::size_t, std::size_t>> pairs;

    if (nAll > maxPairs) {
        // Subsample pairs to bound the runtime, and scale the count back up.
        scale = double(nAll) / double(maxPairs);
        std::uniform_int_distribution<std::size_t> ui(0, K - 1);
        pairs.reserve(maxPairs);
        for (long long n = 0; n < maxPairs; ++n) {
            std::size_t i = ui(rng), j = ui(rng);
            while (j == i) j = ui(rng);
            if (i > j) std::swap(i, j);
            pairs.emplace_back(i, j);
        }
    } else {
        pairs.reserve(nAll);
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = i + 1; j < K; ++j) pairs.emplace_back(i, j);
    }

    long long passed = 0;
    for (const auto& [i, j] : pairs) {
        if (ph[i].tag == ph[j].tag && draw[i] == draw[j]) continue;   // same source
        double c = ph[i].ux * ph[j].ux + ph[i].uy * ph[j].uy + ph[i].uz * ph[j].uz;
        c = std::clamp(c, -1.0, 1.0);
        const double minv = std::sqrt(std::max(2.0 * ph[i].E * ph[j].E * (1.0 - c), 0.0));
        const double esum = ph[i].E + ph[j].E;
        if (minv >= massLo && minv <= massHi && esum >= esumThr) ++passed;
    }
    return double(passed) * scale;
}

}  // namespace

int main(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--library")              a.library = nx();
        else if (s == "--n-library-electrons")  a.nLibraryElectrons = std::stoll(nx());
        else if (s == "--calo-face")            a.caloFace = true;
        else if (s == "--min-photon-E")         a.minPhotonE = std::stod(nx());
        else if (s == "--beam-mode")            a.beamMode = nx();
        else if (s == "--gate-ns")              a.gate_ns = std::stod(nx());
        else if (s == "--n-trials")             a.nTrials = std::stoll(nx());
        else if (s == "--mass-MeV")             a.mass_MeV = std::stod(nx());
        else if (s == "--mass-window-MeV")      a.massWindow_MeV = std::stod(nx());
        else if (s == "--threshold-MeV")        a.threshold_MeV = std::stod(nx());
        else if (s == "--use-geom-accept")      a.useGeomAccept = true;
        else if (s == "--fixed")                a.fixedOcc = true;
        else if (s == "--max-pairs")            a.maxPairs = std::stoll(nx());
        else if (s == "--max-photons-per-gate") a.maxPhotonsPerGate = std::stoll(nx());
        else if (s == "--out-csv")              a.outCsv = nx();
        else if (s == "--seed")                 a.seed = std::stoull(nx());
        else {
            std::fprintf(stderr,
                "Usage: %s --library PATH --n-library-electrons N [options]\n"
                "  --calo-face            library is calo_face_particles.root\n"
                "  --beam-mode MODE       dark | lesa | xleap | interleaved\n"
                "  --gate-ns NS           readout gate (default 1000 = 1 us)\n"
                "  --n-trials N           synthetic gates (default 100000)\n"
                "  --mass-MeV M           ALP window centre (default 100)\n"
                "  --mass-window-MeV W    +/- window (default 20)\n"
                "  --threshold-MeV T      min summed pair energy (default 5)\n"
                "  --min-photon-E E       calo-face photon cut (default 1)\n"
                "  --use-geom-accept      geometric acceptance instead of scored hits\n"
                "  --fixed                fixed occupancy instead of Poisson\n"
                "  --out-csv PATH         summary CSV\n  --seed N\n", argv[0]);
            return 1;
        }
    }
    if (a.library.empty() || a.nLibraryElectrons <= 0) {
        std::fprintf(stderr, "Error: --library and --n-library-electrons are required.\n");
        return 1;
    }
    const auto it = kBeamModes.find(a.beamMode);
    if (it == kBeamModes.end()) { std::fprintf(stderr, "Unknown beam mode\n"); return 1; }
    const BeamSpec spec = it->second;

    std::size_t nPhotons = 0;
    const auto sets = a.caloFace ? LoadCaloFaceLibrary(a.library, a.minPhotonE, nPhotons)
                                 : LoadPi0Library(a.library, a.useGeomAccept, nPhotons);
    // An empty library is a real result, not an error: with the default
    // (scored-hit) acceptance no pi0 gamma survives the material to the calo
    // face, so the accidental rate from that source is exactly zero.
    if (sets.empty()) {
        std::fprintf(stderr, "[warn] no photons passed the acceptance cut in %s"
                             " -- reporting zero accidental rate.\n", a.library.c_str());
    }
    const double pHas = sets.empty() ? 0.0
                                     : double(sets.size()) / double(a.nLibraryElectrons);
    std::printf("[lib ] %s: %zu photon-bearing electrons, %zu photons, P(has)=%.4e\n",
                a.library.c_str(), sets.size(), nPhotons, pHas);
    std::printf("[beam] mode=%s gate=%.0f ns\n", a.beamMode.c_str(), a.gate_ns);

    std::mt19937_64 rng(a.seed);
    std::uniform_int_distribution<std::size_t> pickSet(0, sets.empty() ? 0 : sets.size() - 1);

    double totalFakes = 0, sumOcc = 0, sumBearers = 0;
    long long gatesWithFake = 0;
    std::size_t maxK = 0;
    bool warned = false;

    for (long long g = 0; g < a.nTrials; ++g) {
        const long long occ = SampleGateOccupancy(spec, a.gate_ns * 1e-9, !a.fixedOcc, rng);
        sumOcc += double(occ);

        // Most electrons make no calo photon; only the bearers contribute.
        std::binomial_distribution<long long> bd(occ, pHas);
        const long long m = sets.empty() ? 0 : bd(rng);
        sumBearers += double(m);
        if (m < 1) continue;

        std::vector<Photon> photons;
        std::vector<int> draw;
        for (long long k = 0; k < m; ++k) {
            const auto& s = sets[pickSet(rng)];
            for (const auto& p : s) { photons.push_back(p); draw.push_back(int(k)); }
        }
        maxK = std::max(maxK, photons.size());

        if (static_cast<long long>(photons.size()) > a.maxPhotonsPerGate) {
            if (!warned) {
                std::fprintf(stderr,
                    "[warn] gate has %zu calo photons (> --max-photons-per-gate=%lld). "
                    "This mode is in the saturated-pileup regime where the "
                    "accidental-pair model breaks down; treat results as a lower bound.\n",
                    photons.size(), a.maxPhotonsPerGate);
                warned = true;
            }
            std::vector<std::size_t> idx(photons.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            idx.resize(a.maxPhotonsPerGate);
            std::vector<Photon> ps; std::vector<int> ds;
            for (auto i : idx) { ps.push_back(photons[i]); ds.push_back(draw[i]); }
            photons.swap(ps); draw.swap(ds);
        }

        const double fakes = CountFakesInGate(photons, draw,
                                              a.mass_MeV - a.massWindow_MeV,
                                              a.mass_MeV + a.massWindow_MeV,
                                              a.threshold_MeV, a.maxPairs, rng);
        if (fakes > 0) { totalFakes += fakes; ++gatesWithFake; }
    }

    const double fracGate = double(gatesWithFake) / double(a.nTrials);
    const double rAcc = fracGate * spec.kickerRate;
    const double fakeRate = (totalFakes / double(a.nTrials)) * spec.kickerRate;

    std::printf("\n=== pileup overlay: %s ===\n", a.beamMode.c_str());
    std::printf("  mean occupancy        %.4e electrons/gate\n", sumOcc / a.nTrials);
    std::printf("  mean photon bearers   %.4e /gate\n", sumBearers / a.nTrials);
    std::printf("  max calo photons/gate %zu\n", maxK);
    std::printf("  gates with a fake     %lld / %lld  (%.4e)\n",
                gatesWithFake, a.nTrials, fracGate);
    std::printf("  total fake pairs      %.4e\n", totalFakes);
    std::printf("  R_acc                 %.4e Hz\n", rAcc);
    std::printf("  fake pair rate        %.4e Hz\n", fakeRate);

    io::EnsureParentDir(a.outCsv);
    std::ofstream out(a.outCsv);
    out << "beam_mode,gate_ns,mean_occupancy,mean_bearers,p_has,max_photons_gate,"
           "gates_with_fake,frac_gates_with_fake,total_fakes,R_acc_Hz,"
           "fake_pair_rate_Hz,kicker_rate_Hz\n";
    out << a.beamMode << "," << a.gate_ns << "," << sumOcc / a.nTrials << ","
        << sumBearers / a.nTrials << "," << pHas << "," << maxK << ","
        << gatesWithFake << "," << fracGate << "," << totalFakes << "," << rAcc << ","
        << fakeRate << "," << spec.kickerRate << "\n";
    std::printf("  wrote %s\n", a.outCsv.c_str());
    return 0;
}
