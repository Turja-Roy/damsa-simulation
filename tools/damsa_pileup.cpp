// Accidental gamma-gamma coincidence rate from beam pileup.
// Replaces scripts/pipeline/pileup_overlay.py.
//
//   ./build/damsa_pileup --library output/pi0_decays.root \
//        --n-library-electrons 1000000 --beam-mode lesa
//   ./build/damsa_pileup --library output/calo_face_particles.root --calo-face \
//        --n-library-electrons 1000000 --beam-mode dark,lesa,xleap \
//        --bunch-frac 0.1,0.2,0.4,0.5,0.6,0.8,1 --window-ns 5,10,20,30,40,50,60,70,80,90,100 \
//        --mass-MeV 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200
//
// Builds synthetic coincidence windows by drawing library electrons with
// replacement, then counts gamma-gamma pairs from DIFFERENT sources whose
// invariant mass lands in the ALP window. Those are the fakes a real ALP search
// must survive. Every list flag is a grid axis; the library is loaded once and
// one CSV row is written per (mode, bunch-frac, window, mass) cell.
//
// Intensity follows Random-2-photons plan §3.2: bunch charge = f x Table I cap,
// and the headline quantity is mu = mean electrons per window.

#include "damsa_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
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
    bool caloFace = false, useGeomAccept = false, fixedOcc = false, case3 = false;
    bool expected = false;
    double minPhotonE = 1.0, massWindow_MeV = 20.0, threshold_MeV = 5.0, zMax = 1.0;
    double kickerRateOverride = 0, targetExitZ_mm = -400.0;
    std::vector<std::string> modes = {"dark"};
    std::vector<double> bunchFracs = {1.0}, windows_ns = {1000.0}, masses_MeV = {100.0};
    long long nTrials = 100000, maxPairs = 200000, maxPhotonsPerGate = 20000;
    std::uint64_t seed = 12345;
};

template <class T>
std::vector<T> ParseList(const std::string& s)
{
    std::vector<T> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) {
            if constexpr (std::is_same_v<T, std::string>) out.push_back(tok);
            else out.push_back(static_cast<T>(std::stod(tok)));
        }
    return out;
}

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
// fake pool. One library event = one drawn electron. The source tag is
// (eventID, primaryID): photons of one primary's shower are correlated, not
// accidental. In Level A that is the same as eventID alone; files from before
// the truth columns read primaryID = -1, which still groups by event.
std::vector<PhotonSet> LoadCaloFaceLibrary(const Args& a, std::size_t& nPhotons)
{
    const auto rows = io::ReadNTuple<io::ParticleRow>(a.library);
    std::map<int, PhotonSet> perElectron;
    nPhotons = 0;

    for (const auto& p : rows) {
        // pz <= 0: backsplash leaving the calo (scored since plan step 1), not an
        // incoming photon.
        if (p.pdg != 22 || p.energy_MeV < a.minPhotonE || p.pz <= 0) continue;
        // Case 3 (plan §1): keep only photons scattered outside the target or
        // born downstream of it.
        if (a.case3 && !(p.scattered == 1 || p.vz_mm > a.targetExitZ_mm)) continue;
        const long long tag = (static_cast<long long>(p.eventID) << 32)
                            ^ static_cast<long long>(static_cast<unsigned>(p.primaryID));
        perElectron[p.eventID].push_back({p.energy_MeV, p.px, p.py, p.pz, tag});
        ++nPhotons;
    }

    std::vector<PhotonSet> sets;
    for (auto& [evt, v] : perElectron) if (!v.empty()) sets.push_back(std::move(v));
    return sets;
}

// Bunches inside one coincidence window of width W. The window is not
// phase-locked to the beam: the triggering bunch sits at a uniform position u
// inside it, so the count is 1 + floor(u/s) + floor((W-u)/s), capped at the
// bunches in one kick. Below one spacing this is exactly 1 -- all remaining
// accidentals are same-bunch (plan §3.2).
template <class RNG>
int SampleBunchesInWindow(const BeamSpec& s, double window_s, RNG& rng)
{
    std::uniform_real_distribution<double> uni(0.0, window_s);
    const double u = uni(rng);
    const int nb = 1 + static_cast<int>(std::floor(u / s.spacing_s))
                     + static_cast<int>(std::floor((window_s - u) / s.spacing_s));
    return std::min(nb, s.bunchesPerKick);
}

// A sum of n iid Poisson(q) is Poisson(n*q), so one draw per window suffices.
template <class RNG>
long long SampleOccupancy(int nBunches, double bunchCharge, bool poisson, RNG& rng)
{
    const double mean = nBunches * bunchCharge;
    if (!poisson) return static_cast<long long>(std::llround(mean));
    std::poisson_distribution<long long> pd(mean);
    return pd(rng);
}

// pileup_overlay.py:181. A pair is accidental only if it came from different
// sources: different tag OR different draw. The library is sampled WITH
// replacement, so one library electron drawn twice stands for two independent
// real electrons and its photons must pair as accidentals despite equal tags.
// All mass windows are tested per pair, so the mass axis costs one pair loop.
template <class RNG>
void CountFakesInGate(const std::vector<Photon>& ph, const std::vector<int>& draw,
                      const Args& a, std::vector<double>& fakes, RNG& rng)
{
    std::fill(fakes.begin(), fakes.end(), 0.0);
    const std::size_t K = ph.size();
    if (K < 2) return;

    const long long nAll = static_cast<long long>(K) * (K - 1) / 2;
    double scale = 1.0;
    std::vector<std::pair<std::size_t, std::size_t>> pairs;

    if (nAll > a.maxPairs) {
        // Subsample pairs to bound the runtime, and scale the count back up.
        scale = double(nAll) / double(a.maxPairs);
        std::uniform_int_distribution<std::size_t> ui(0, K - 1);
        pairs.reserve(a.maxPairs);
        for (long long n = 0; n < a.maxPairs; ++n) {
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

    for (const auto& [i, j] : pairs) {
        if (ph[i].tag == ph[j].tag && draw[i] == draw[j]) continue;   // same source
        const double esum = ph[i].E + ph[j].E;
        if (esum < a.threshold_MeV) continue;
        if (std::abs(ph[i].E - ph[j].E) / esum > a.zMax) continue;   // asymmetry, plan §7
        double c = ph[i].ux * ph[j].ux + ph[i].uy * ph[j].uy + ph[i].uz * ph[j].uz;
        c = std::clamp(c, -1.0, 1.0);
        const double minv = std::sqrt(std::max(2.0 * ph[i].E * ph[j].E * (1.0 - c), 0.0));
        for (std::size_t m = 0; m < a.masses_MeV.size(); ++m)
            if (std::abs(minv - a.masses_MeV[m]) <= a.massWindow_MeV) fakes[m] += scale;
    }
}

// ── Exact expected-value mode (--expected) ─────────────────────────────────
// The MC above cannot resolve rare fakes: a 1e-7 per-window probability is
// still ~1 Hz at 1e7 windows/s. Instead, for electrons drawn independently from
// the N-electron library,
//   E[fakes/window] = E[occ(occ-1)] * S_diff / N^2  +  E[occ] * S_same / N
// S_diff = unordered photon pairs from different library electrons passing the
// cuts; S_same = pairs within one electron but from different sources (only
// nonzero in pi0 mode). Both depend on the library and cuts only, so they are
// summed once and every (mode, f, window) cell costs nothing.
struct PairSums { std::vector<double> diff, same; std::vector<bool> valid; };

PairSums SumPairs(const std::vector<PhotonSet>& sets, const Args& a)
{
    struct P { double E, ux, uy, uz; long long tag; int set; };
    std::vector<P> ph;
    for (int k = 0; k < static_cast<int>(sets.size()); ++k)
        for (const auto& p : sets[k]) ph.push_back({p.E, p.ux, p.uy, p.uz, p.tag, k});
    std::sort(ph.begin(), ph.end(), [](const P& x, const P& y) { return x.E > y.E; });

    const std::size_t nM = a.masses_MeV.size();
    PairSums r{std::vector<double>(nM, 0.0), std::vector<double>(nM, 0.0),
               std::vector<bool>(nM, true)};
    // Pruning: m^2 = 2 E1 E2 (1-cos) <= 4 E1 E2, so a pair can reach mass lo
    // only if E1 E2 >= lo^2/4. Masses whose window reaches 0 cannot be pruned
    // (every collinear pair qualifies) and are left to the MC mode.
    double lo = 1e300;
    for (std::size_t m = 0; m < nM; ++m) {
        const double l = a.masses_MeV[m] - a.massWindow_MeV;
        if (l <= 0) { r.valid[m] = false; continue; }
        lo = std::min(lo, l);
    }
    if (lo == 1e300) return r;
    const double prodMin = lo * lo / 4.0;

    for (std::size_t i = 0; i < ph.size(); ++i) {
        const double eMin = prodMin / ph[i].E;
        for (std::size_t j = i + 1; j < ph.size() && ph[j].E >= eMin; ++j) {
            if (ph[i].tag == ph[j].tag) continue;   // same source: correlated
            const double esum = ph[i].E + ph[j].E;
            if (esum < a.threshold_MeV) continue;
            if (std::abs(ph[i].E - ph[j].E) / esum > a.zMax) continue;
            double c = ph[i].ux * ph[j].ux + ph[i].uy * ph[j].uy + ph[i].uz * ph[j].uz;
            c = std::clamp(c, -1.0, 1.0);
            const double minv = std::sqrt(std::max(2.0 * ph[i].E * ph[j].E * (1.0 - c), 0.0));
            auto& tgt = (ph[i].set == ph[j].set) ? r.same : r.diff;
            for (std::size_t m = 0; m < nM; ++m)
                if (r.valid[m] && std::abs(minv - a.masses_MeV[m]) <= a.massWindow_MeV) tgt[m] += 1.0;
        }
    }
    return r;
}

void Usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s --library PATH --n-library-electrons N [options]\n"
        "  List flags take comma-separated values; each is a grid axis.\n"
        "  --calo-face              library is calo_face_particles.root\n"
        "  --beam-mode LIST         dark | lesa | xleap | interleaved (default dark)\n"
        "  --bunch-frac LIST        fraction f of the Table I bunch charge (default 1)\n"
        "  --window-ns LIST         coincidence window (default 1000; alias --gate-ns)\n"
        "  --mass-MeV LIST          ALP window centres (default 100)\n"
        "  --mass-window-MeV W      +/- window (default 20)\n"
        "  --threshold-MeV T        min summed pair energy (default 5)\n"
        "  --z-max Z                max |E1-E2|/(E1+E2) (default 1 = no cut)\n"
        "  --kicker-rate-Hz R       override the Table I kicker rate\n"
        "  --n-trials N             synthetic windows per (mode,f,window) (default 100000)\n"
        "  --expected               exact expected fakes per window instead of MC trials;\n"
        "                           resolves rare fakes. Masses with m <= window: MC only\n"
        "  --min-photon-E E         calo-face photon cut (default 1)\n"
        "  --case3                  calo-face: only photons scattered outside / born\n"
        "                           downstream of the target (plan §1)\n"
        "  --target-exit-z-mm Z     target rear face for --case3 (default -400)\n"
        "  --use-geom-accept        pi0: geometric acceptance instead of scored hits\n"
        "  --fixed                  fixed occupancy instead of Poisson\n"
        "  --out-csv PATH           grid CSV, one row per cell\n  --seed N\n", prog);
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
        else if (s == "--case3")                a.case3 = true;
        else if (s == "--target-exit-z-mm")     a.targetExitZ_mm = std::stod(nx());
        else if (s == "--min-photon-E")         a.minPhotonE = std::stod(nx());
        else if (s == "--beam-mode")            a.modes = ParseList<std::string>(nx());
        else if (s == "--bunch-frac")           a.bunchFracs = ParseList<double>(nx());
        else if (s == "--window-ns" || s == "--gate-ns") a.windows_ns = ParseList<double>(nx());
        else if (s == "--mass-MeV")             a.masses_MeV = ParseList<double>(nx());
        else if (s == "--mass-window-MeV")      a.massWindow_MeV = std::stod(nx());
        else if (s == "--threshold-MeV")        a.threshold_MeV = std::stod(nx());
        else if (s == "--z-max")                a.zMax = std::stod(nx());
        else if (s == "--kicker-rate-Hz")       a.kickerRateOverride = std::stod(nx());
        else if (s == "--n-trials")             a.nTrials = std::stoll(nx());
        else if (s == "--use-geom-accept")      a.useGeomAccept = true;
        else if (s == "--expected")             a.expected = true;
        else if (s == "--fixed")                a.fixedOcc = true;
        else if (s == "--max-pairs")            a.maxPairs = std::stoll(nx());
        else if (s == "--max-photons-per-gate") a.maxPhotonsPerGate = std::stoll(nx());
        else if (s == "--out-csv")              a.outCsv = nx();
        else if (s == "--seed")                 a.seed = std::stoull(nx());
        else { Usage(argv[0]); return 1; }
    }
    if (a.library.empty() || a.nLibraryElectrons <= 0) {
        std::fprintf(stderr, "Error: --library and --n-library-electrons are required.\n");
        return 1;
    }
    for (const auto& m : a.modes)
        if (!kBeamModes.count(m)) { std::fprintf(stderr, "Unknown beam mode %s\n", m.c_str()); return 1; }

    std::size_t nPhotons = 0;
    const auto sets = a.caloFace ? LoadCaloFaceLibrary(a, nPhotons)
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

    io::EnsureParentDir(a.outCsv);
    std::ofstream out(a.outCsv);
    out << "beam_mode,bunch_frac,window_ns,mass_MeV,mass_window_MeV,threshold_MeV,z_max,"
           "case3,mu,mean_bunches,p_has,mean_bearers,max_photons_window,"
           "windows_with_fake,frac_windows_with_fake,fakes_per_window,"
           "windows_per_s,electrons_per_s,R_acc_Hz,fake_pair_rate_Hz,kicker_rate_Hz,method\n";

    PairSums sums;
    if (a.expected) {
        sums = SumPairs(sets, a);
        for (std::size_t m = 0; m < a.masses_MeV.size(); ++m)
            std::printf("[sums] m=%6.1f MeV  S_diff=%.6g  S_same=%.6g%s\n", a.masses_MeV[m],
                        sums.diff[m], sums.same[m], sums.valid[m] ? "" : "  (window reaches 0: use MC)");
    }

    std::mt19937_64 rng(a.seed);
    std::uniform_int_distribution<std::size_t> pickSet(0, sets.empty() ? 0 : sets.size() - 1);
    const std::size_t nM = a.masses_MeV.size();
    std::vector<double> fakes(nM);

    for (const auto& mode : a.modes)
    for (const double f : a.bunchFracs)
    for (const double w_ns : a.windows_ns) {
        BeamSpec spec = kBeamModes.at(mode);
        if (a.kickerRateOverride > 0) spec.kickerRate = a.kickerRateOverride;
        const double q = f * spec.bunchCharge;

        if (a.expected) {
            // Occupancy moments from the window/bunch model; no photons drawn.
            double sB = 0, sOcc = 0, sOcc2 = 0;   // E[nb], E[occ], E[occ(occ-1)]
            for (long long g = 0; g < a.nTrials; ++g) {
                const int nb = SampleBunchesInWindow(spec, w_ns * 1e-9, rng);
                sB += nb;
                if (a.fixedOcc) {
                    const double o = std::llround(nb * q);
                    sOcc += o; sOcc2 += o * (o - 1);
                } else {
                    sOcc += nb * q; sOcc2 += (nb * q) * (nb * q);   // Poisson: E[n(n-1)] = mean^2
                }
            }
            const double meanBunches = sB / a.nTrials, mu = sOcc / a.nTrials;
            const double occPairs = sOcc2 / a.nTrials;
            const double windowsPerS = spec.kickerRate * spec.bunchesPerKick / meanBunches;
            const double electronsPerS = q * spec.bunchesPerKick * spec.kickerRate;
            const double N = double(a.nLibraryElectrons);
            std::printf("\n=== %s f=%g window=%g ns: mu=%.4e e-/window, %.3g windows/s (expected) ===\n",
                        mode.c_str(), f, w_ns, mu, windowsPerS);
            for (std::size_t k = 0; k < nM; ++k) {
                const bool ok = sums.valid[k];
                const double perWindow = ok ? occPairs * sums.diff[k] / (N * N) + mu * sums.same[k] / N
                                            : std::nan("");
                const double frac = -std::expm1(-perWindow);   // P(>=1), Poisson in pairs
                std::printf("  m=%6.1f MeV  fakes/window %.4e  R_acc %.4e Hz\n",
                            a.masses_MeV[k], perWindow, frac * windowsPerS);
                out << mode << "," << f << "," << w_ns << "," << a.masses_MeV[k] << ","
                    << a.massWindow_MeV << "," << a.threshold_MeV << "," << a.zMax << ","
                    << (a.case3 ? 1 : 0) << "," << mu << "," << meanBunches << "," << pHas << ","
                    << mu * pHas << ",,," << frac << "," << perWindow << ","
                    << windowsPerS << "," << electronsPerS << ","
                    << frac * windowsPerS << "," << perWindow * windowsPerS << ","
                    << spec.kickerRate << ",expected\n";
            }
            continue;
        }

        double sumOcc = 0, sumBearers = 0, sumBunches = 0;
        std::vector<double> totalFakes(nM, 0.0);
        std::vector<long long> withFake(nM, 0);
        std::size_t maxK = 0;
        bool warned = false;

        for (long long g = 0; g < a.nTrials; ++g) {
            const int nb = SampleBunchesInWindow(spec, w_ns * 1e-9, rng);
            sumBunches += nb;
            const long long occ = SampleOccupancy(nb, q, !a.fixedOcc, rng);
            sumOcc += double(occ);

            // Most electrons make no calo photon; only the bearers contribute.
            std::binomial_distribution<long long> bd(occ, pHas);
            const long long m = sets.empty() ? 0 : bd(rng);
            sumBearers += double(m);
            // m == 1 still counts in pi0 mode: two pi0s of one electron differ in tag.
            if (m < 1) continue;

            std::vector<Photon> photons;
            std::vector<int> draw;
            for (long long k = 0; k < m; ++k) {
                const auto& st = sets[pickSet(rng)];
                for (const auto& p : st) { photons.push_back(p); draw.push_back(int(k)); }
            }
            maxK = std::max(maxK, photons.size());

            if (static_cast<long long>(photons.size()) > a.maxPhotonsPerGate) {
                if (!warned) {
                    std::fprintf(stderr,
                        "[warn] %s f=%g W=%gns: window has %zu calo photons "
                        "(> --max-photons-per-gate=%lld). Saturated-pileup regime: the "
                        "accidental-pair model breaks down; treat results as a lower bound.\n",
                        mode.c_str(), f, w_ns, photons.size(), a.maxPhotonsPerGate);
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

            CountFakesInGate(photons, draw, a, fakes, rng);
            for (std::size_t k = 0; k < nM; ++k)
                if (fakes[k] > 0) { totalFakes[k] += fakes[k]; ++withFake[k]; }
        }

        // Windows per second chosen so that windows/s x mu = electrons/s: each
        // kick's bunch train is tiled by windows holding mean_bunches bunches.
        // For a window >= the train this is one window per kick (the old gate).
        const double meanBunches = sumBunches / a.nTrials;
        const double mu = sumOcc / a.nTrials;
        const double windowsPerS = spec.kickerRate * spec.bunchesPerKick / meanBunches;
        const double electronsPerS = q * spec.bunchesPerKick * spec.kickerRate;

        std::printf("\n=== %s f=%g window=%g ns: mu=%.4e e-/window, %.3g windows/s, "
                    "max photons/window %zu ===\n",
                    mode.c_str(), f, w_ns, mu, windowsPerS, maxK);
        for (std::size_t k = 0; k < nM; ++k) {
            const double frac = double(withFake[k]) / double(a.nTrials);
            const double perWindow = totalFakes[k] / double(a.nTrials);
            std::printf("  m=%6.1f MeV  frac windows w/ fake %.4e  R_acc %.4e Hz\n",
                        a.masses_MeV[k], frac, frac * windowsPerS);
            out << mode << "," << f << "," << w_ns << "," << a.masses_MeV[k] << ","
                << a.massWindow_MeV << "," << a.threshold_MeV << "," << a.zMax << ","
                << (a.case3 ? 1 : 0) << "," << mu << "," << meanBunches << "," << pHas << ","
                << sumBearers / a.nTrials << "," << maxK << ","
                << withFake[k] << "," << frac << "," << perWindow << ","
                << windowsPerS << "," << electronsPerS << ","
                << frac * windowsPerS << "," << perWindow * windowsPerS << ","
                << spec.kickerRate << ",mc\n";
        }
    }
    std::printf("\n  wrote %s\n", a.outCsv.c_str());
    return 0;
}
