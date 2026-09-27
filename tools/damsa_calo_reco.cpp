// SM calorimeter reconstruction vs truth (Random-2-photons plan §5.4-§7).
//
//   ./build/damsa_calo_reco --hits output/calo_hits.root \
//        --truth output/calo_face_particles.root --n-library-electrons 1000000 \
//        --beam-mode dark,lesa --bunch-frac 0.1,1 --window-ns 10,50,100 \
//        --sigma-t-ns 1,10,100 --mass-MeV 10,20,50,100,200 --out-csv output/reco_grid.csv
//
// Per grid cell (mode, f, window, sigma_t): build readout windows by drawing
// library electrons (beam_window.h), each at its bunch time; sum their crystal
// hits; digitize (cell threshold, time smear); cluster the x and y strip
// profiles (calo_reco.h); pair clusters into photons; form 2-photon candidates
// and compare with the truth photons at the calo face of the same electrons.
//
// Candidate classes (plan §6):
//   TP        both photons match truth photons of one parent + one vertex
//             (any SM source: pi0, eta, eta', omega, annihilation, ...)
//   FP_acc    photons from different electrons (the accidental)
//   FP_shower same electron, not one decay (two pieces of one shower)
//   FP_split  both photons match the same truth particle
//   FP_ghost  a photon's x and y clusters match different truth particles
//   FP_unm    a photon matches no truth particle (charged, noise)
// Window outcome: TP if any TP candidate, else FP if any candidate, else FN if
// a genuine truth pair was in the mass/energy window, else TN. Each (mass,
// E_pair bin) is reported without and with the asymmetry cut z <= --z-cut.

#include "damsa_io.h"
#include "beam_window.h"
#include "calo_reco.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

struct Hit   { int cell; double E, t; };
struct Truth { double E, x, y, ux, uy, uz, vx, vy, vz; int pdg, parent, primary, draw; };
struct Electron { std::vector<Hit> hits; std::vector<Truth> truth; };

enum Cls { TP, FP_acc, FP_shower, FP_split, FP_ghost, FP_unm, NCls };
const char* kClsName[NCls] = {"TP", "FP_acc", "FP_shower", "FP_split", "FP_ghost", "FP_unm"};

struct Args {
    std::string hits, truth, outCsv = "output/reco_grid.csv";
    long long nLib = 0, nTrials = 2000;
    std::vector<std::string> modes = {"dark"};
    std::vector<double> fracs = {1.0}, windows_ns = {100.0}, sigmaT_ns = {1.0},
                        masses = {100.0}, eBins_GeV;   // empty eBins = no E_pair cut
    double massWindow = 20.0, esumMin = 5.0, zCut = 0.8;
    double cellThr = 0.1, seed = 0.5, clusterMin = 5.0, truthMin = 1.0, effMin = 10.0;   // seed: soft photons peak at ~1.5 MeV per strip
    double resA = 0.02, resB = 0.01, matchR = 36.0, lever_mm = 470.0;
    calo::Geometry geo;
    std::uint64_t rngSeed = 12345;
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

// Library electrons that left anything in the calo or at its face. The rest
// contribute nothing and are drawn implicitly via P(has) (as in damsa_pileup).
std::vector<Electron> LoadLibrary(const Args& a, int& nCells)
{
    std::map<int, Electron> ev;
    nCells = 0;
    for (const auto& h : io::ReadNTuple<io::CaloHitRow>(a.hits)) {
        ev[h.eventID].hits.push_back({h.cellID, h.edep_MeV, h.t_mean_ns});
        nCells = std::max(nCells, h.cellID + 1);
    }
    for (const auto& p : io::ReadNTuple<io::ParticleRow>(a.truth)) {
        if (p.pz <= 0 || p.energy_MeV < a.truthMin) continue;
        if (p.pdg != 22 && std::abs(p.pdg) != 11) continue;   // showering particles only
        if (p.parentID < 0) throw std::runtime_error(a.truth + " has no truth columns; regenerate the library");
        ev[p.eventID].truth.push_back({p.energy_MeV, p.x_mm, p.y_mm, p.px, p.py, p.pz,
                                       p.vx_mm, p.vy_mm, p.vz_mm, p.pdg, p.parentID, p.primaryID, -1});
    }
    std::vector<Electron> out;
    for (auto& [id, e] : ev) out.push_back(std::move(e));
    return out;
}

bool SameVertex(const Truth& a, const Truth& b)
{
    return std::abs(a.vx - b.vx) < 1e-3 && std::abs(a.vy - b.vy) < 1e-3 && std::abs(a.vz - b.vz) < 1e-3;
}

// Highest-energy truth particle within R of the photon, in x only, y only, or 2D.
int Match(const std::vector<Truth>& tr, double x, double y, double R, int mode)
{
    int best = -1;
    for (int i = 0; i < static_cast<int>(tr.size()); ++i) {
        const double dx = tr[i].x - x, dy = tr[i].y - y;
        const double d = mode == 0 ? std::abs(dx) : mode == 1 ? std::abs(dy) : std::hypot(dx, dy);
        if (d < R && (best < 0 || tr[i].E > tr[best].E)) best = i;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--hits")                a.hits = nx();
        else if (s == "--truth")               a.truth = nx();
        else if (s == "--n-library-electrons") a.nLib = std::stoll(nx());
        else if (s == "--beam-mode")           a.modes = ParseList<std::string>(nx());
        else if (s == "--bunch-frac")          a.fracs = ParseList<double>(nx());
        else if (s == "--window-ns")           a.windows_ns = ParseList<double>(nx());
        else if (s == "--sigma-t-ns")          a.sigmaT_ns = ParseList<double>(nx());
        else if (s == "--mass-MeV")            a.masses = ParseList<double>(nx());
        else if (s == "--e-bins-GeV")          a.eBins_GeV = ParseList<double>(nx());
        else if (s == "--mass-window-MeV")     a.massWindow = std::stod(nx());
        else if (s == "--threshold-MeV")       a.esumMin = std::stod(nx());
        else if (s == "--z-cut")               a.zCut = std::stod(nx());
        else if (s == "--cell-threshold-MeV")  a.cellThr = std::stod(nx());
        else if (s == "--seed-MeV")            a.seed = std::stod(nx());
        else if (s == "--cluster-min-MeV")     a.clusterMin = std::stod(nx());
        else if (s == "--eff-min-MeV")         a.effMin = std::stod(nx());
        else if (s == "--res-a")               a.resA = std::stod(nx());
        else if (s == "--res-b")               a.resB = std::stod(nx());
        else if (s == "--match-radius-mm")     a.matchR = std::stod(nx());
        else if (s == "--lever-arm-mm")        a.lever_mm = std::stod(nx());
        else if (s == "--calo-xy-mm")          a.geo.caloXY_mm = std::stod(nx());
        else if (s == "--n-per-layer")         a.geo.nPerLayer = std::stoi(nx());
        else if (s == "--n-trials")            a.nTrials = std::stoll(nx());
        else if (s == "--out-csv")             a.outCsv = nx();
        else if (s == "--seed")                a.rngSeed = std::stoull(nx());
        else {
            std::fprintf(stderr,
                "Usage: %s --hits calo_hits.root --truth calo_face_particles.root\n"
                "          --n-library-electrons N [grid lists: --beam-mode --bunch-frac\n"
                "          --window-ns --sigma-t-ns --mass-MeV --e-bins-GeV (centres, +/-0.5)]\n"
                "          [--mass-window-MeV 20 --threshold-MeV 5 --z-cut 0.8\n"
                "           --cell-threshold-MeV 0.1 --seed-MeV 0.5 --cluster-min-MeV 5 --eff-min-MeV 10\n"
                "           --res-a 0.02 --res-b 0.01 --match-radius-mm 36\n"
                "           --lever-arm-mm 470 --calo-xy-mm 120 --n-per-layer 12\n"
                "           --n-trials 2000 --out-csv PATH --seed N]\n", argv[0]);
            return 1;
        }
    }
    if (a.hits.empty() || a.truth.empty() || a.nLib <= 0) {
        std::fprintf(stderr, "Error: --hits, --truth and --n-library-electrons are required.\n");
        return 1;
    }
    for (const auto& m : a.modes)
        if (!kBeamModes.count(m)) { std::fprintf(stderr, "Unknown beam mode %s\n", m.c_str()); return 1; }

    int nCells = 0;
    const auto lib = LoadLibrary(a, nCells);
    if (lib.empty()) { std::fprintf(stderr, "Error: library has no calo activity.\n"); return 1; }
    const double pHas = double(lib.size()) / double(a.nLib);
    std::printf("[lib ] %zu active electrons of %lld (P=%.4e), %d cells\n",
                lib.size(), a.nLib, pHas, nCells);

    // E_pair bins in MeV; the first is always "all".
    std::vector<std::pair<double, double>> eBins = {{0.0, std::numeric_limits<double>::infinity()}};
    for (double c : a.eBins_GeV) eBins.push_back({(c - 0.5) * 1e3, (c + 0.5) * 1e3});
    const std::size_t nM = a.masses.size(), nE = eBins.size();
    auto idx = [&](std::size_t m, std::size_t e, int zc) { return (m * nE + e) * 2 + zc; };

    std::ofstream out(a.outCsv);
    out << "beam_mode,bunch_frac,window_ns,sigma_t_ns,mass_MeV,epair_lo_MeV,epair_hi_MeV,zcut,"
           "mu,windows,TP,FP,FN,TN";
    for (int c = 0; c < NCls; ++c) out << ",cand_" << kClsName[c];
    out << ",truth_photons,reco_photons,photon_eff,photon_purity,windows_per_s,FP_rate_Hz\n";

    std::mt19937_64 rng(a.rngSeed);
    std::uniform_int_distribution<std::size_t> pick(0, lib.size() - 1);
    std::normal_distribution<double> gaus(0.0, 1.0);
    const int nPer = a.geo.nPerLayer;
    const double half = a.geo.caloXY_mm / 2;

    for (const auto& mode : a.modes)
    for (const double f : a.fracs)
    for (const double w : a.windows_ns) {
        const BeamSpec& spec = kBeamModes.at(mode);
        const double q = f * spec.bunchCharge;
        // One accumulator per sigma_t: sigma_t only enters after the overlay, so
        // each overlaid window is reconstructed once per sigma_t value.
        struct Acc {
            std::vector<std::array<long long, 4>> win;   // TP FP FN TN
            std::vector<std::array<long long, NCls>> cand;
            long long nTruthPh = 0, nFound = 0, nReco = 0, nRecoMatched = 0;
        };
        std::vector<Acc> accs(a.sigmaT_ns.size());
        for (auto& ac : accs) {
            ac.win.assign(nM * nE * 2, {0, 0, 0, 0});
            ac.cand.assign(nM * nE * 2, {});
        }
        double sumOcc = 0, sumBunches = 0;

        std::vector<double> E(nCells), Et(nCells);
        for (long long g = 0; g < a.nTrials; ++g) {
            const int nb = SampleBunchesInWindow(spec, w * 1e-9, rng);
            sumBunches += nb;
            const long long occ = SampleOccupancy(nb, q, true, rng);
            sumOcc += double(occ);
            std::binomial_distribution<long long> bd(occ, pHas);
            const long long m = bd(rng);

            // ── overlay ──
            std::fill(E.begin(), E.end(), 0.0);
            std::fill(Et.begin(), Et.end(), 0.0);
            std::vector<Truth> tr;
            std::vector<std::pair<std::size_t, std::size_t>> drawRange;   // tr[first, second) per electron
            std::uniform_int_distribution<int> bunch(0, nb - 1);
            for (long long k = 0; k < m; ++k) {
                const Electron& e = lib[pick(rng)];
                const double toff = bunch(rng) * spec.spacing_s * 1e9;
                for (const auto& h : e.hits) { E[h.cell] += h.E; Et[h.cell] += h.E * (h.t + toff); }
                const std::size_t first = tr.size();
                for (auto t : e.truth) { t.draw = int(k); tr.push_back(t); }
                if (tr.size() - first >= 2) drawRange.push_back({first, tr.size()});
            }

            auto inCalo = [&](const Truth& t) { return std::abs(t.x) < half && std::abs(t.y) < half; };
            // ── genuine truth pairs present (per mass, E bin): one electron at a time ──
            std::vector<char> hasPair(nM * nE, 0);
            for (const auto& [b, e] : drawRange)
            for (std::size_t i = b; i < e; ++i)
            for (std::size_t j = i + 1; j < e; ++j) {
                const Truth &u = tr[i], &v = tr[j];
                if (u.pdg != 22 || v.pdg != 22 || u.parent != v.parent || !SameVertex(u, v)) continue;
                if (u.E < a.clusterMin || v.E < a.clusterMin || !inCalo(u) || !inCalo(v)) continue;
                const double es = u.E + v.E;
                if (es < a.esumMin) continue;
                const double mt = std::sqrt(std::max(2 * u.E * v.E * (1 - (u.ux * v.ux + u.uy * v.uy + u.uz * v.uz)), 0.0));
                for (std::size_t mi = 0; mi < nM; ++mi)
                for (std::size_t ei = 0; ei < nE; ++ei)
                    if (std::abs(mt - a.masses[mi]) <= a.massWindow && es >= eBins[ei].first && es < eBins[ei].second)
                        hasPair[mi * nE + ei] = 1;
            }

            for (std::size_t si = 0; si < a.sigmaT_ns.size(); ++si) {
                const double sigT = a.sigmaT_ns[si];
                Acc& ac = accs[si];
                auto& win = ac.win;
                auto& cand = ac.cand;
                long long& nTruthPh = ac.nTruthPh;
                long long& nFound = ac.nFound;
                long long& nReco = ac.nReco;
                long long& nRecoMatched = ac.nRecoMatched;

                // ── digitize + strip profiles (even layer -> y, odd -> x) ──
                std::vector<calo::Strip> px(nPer), py(nPer);
                for (int c = 0; c < nCells; ++c) {
                    if (E[c] < a.cellThr) continue;
                    const double t = Et[c] / E[c] + sigT * gaus(rng);
                    auto& s = ((c / nPer) % 2 == 0 ? py : px)[c % nPer];
                    s.E += E[c];
                    s.Et += E[c] * t;
                }
                // Each projection holds ~half a shower, so the photon-energy minimum
                // is applied after X/Y pairing; projections only need the seed.
                std::vector<calo::Photon> ph;
                for (auto p : calo::PairXY(calo::ClusterProjection(px, a.geo, a.seed, a.seed),
                                           calo::ClusterProjection(py, a.geo, a.seed, a.seed))) {
                    const double rel = std::hypot(a.resA / std::sqrt(p.E / 1000.0), a.resB);
                    p.E *= std::max(0.0, 1.0 + rel * gaus(rng));
                    if (p.E >= a.clusterMin) ph.push_back(p);
                }

                // ── photon-level truth comparison ──
                // Efficiency denominator: photons well above the reco threshold
                // (--eff-min-MeV), so ~90 % containment does not push them under it.
                auto effPhoton = [&](const Truth& t) { return t.pdg == 22 && t.E >= a.effMin && inCalo(t); };
                std::vector<char> found(tr.size(), 0);
                for (const auto& t : tr) if (effPhoton(t)) ++nTruthPh;
                std::vector<int> match(ph.size()), ghost(ph.size());
                for (std::size_t i = 0; i < ph.size(); ++i) {
                    match[i] = Match(tr, ph[i].x, ph[i].y, a.matchR, 2);
                    const int mx = Match(tr, ph[i].x, ph[i].y, a.matchR, 0);
                    const int my = Match(tr, ph[i].x, ph[i].y, a.matchR, 1);
                    ghost[i] = (mx >= 0 && my >= 0 && mx != my);
                    ++nReco;
                    if (match[i] >= 0) { ++nRecoMatched; found[match[i]] = 1; }
                }
                for (std::size_t i = 0; i < tr.size(); ++i)
                    if (found[i] && effPhoton(tr[i])) ++nFound;

                // ── candidates ──
                std::vector<char> pos(nM * nE * 2, 0), tp(nM * nE * 2, 0);
                for (std::size_t i = 0; i < ph.size(); ++i)
                for (std::size_t j = i + 1; j < ph.size(); ++j) {
                    const double es = ph[i].E + ph[j].E;
                    if (es < a.esumMin || std::abs(ph[i].t - ph[j].t) > w) continue;
                    const double z = std::abs(ph[i].E - ph[j].E) / es;
                    const double mgg = calo::Mgg(ph[i].E, ph[i].x, ph[i].y, ph[j].E, ph[j].x, ph[j].y, a.lever_mm);

                    Cls c;
                    const int ti = match[i], tj = match[j];
                    if (ti < 0 || tj < 0)          c = FP_unm;
                    else if (ghost[i] || ghost[j]) c = FP_ghost;
                    else if (ti == tj)             c = FP_split;
                    else {
                        const Truth &u = tr[ti], &v = tr[tj];
                        if (u.draw != v.draw)                          c = FP_acc;
                        else if (u.parent == v.parent && SameVertex(u, v)) c = TP;
                        else                                           c = FP_shower;
                    }
                    for (std::size_t mi = 0; mi < nM; ++mi) {
                        if (std::abs(mgg - a.masses[mi]) > a.massWindow) continue;
                        for (std::size_t ei = 0; ei < nE; ++ei) {
                            if (es < eBins[ei].first || es >= eBins[ei].second) continue;
                            for (int zc = 0; zc < 2; ++zc) {
                                if (zc == 1 && z > a.zCut) continue;
                                const auto k = idx(mi, ei, zc);
                                pos[k] = 1;
                                if (c == TP) tp[k] = 1;
                                ++cand[k][c];
                            }
                        }
                    }
                }
                for (std::size_t k = 0; k < win.size(); ++k) {
                    const bool pair = hasPair[k / 2];
                    win[k][tp[k] ? 0 : pos[k] ? 1 : pair ? 2 : 3]++;
                }
            }   // sigma_t
        }

        const double meanBunches = sumBunches / a.nTrials, mu = sumOcc / a.nTrials;
        const double windowsPerS = spec.kickerRate * spec.bunchesPerKick / meanBunches;
        for (std::size_t si = 0; si < a.sigmaT_ns.size(); ++si) {
            const double sigT = a.sigmaT_ns[si];
            const Acc& ac = accs[si];
            const auto& win = ac.win;
            const auto& cand = ac.cand;
            const long long nTruthPh = ac.nTruthPh, nFound = ac.nFound, nReco = ac.nReco,
                            nRecoMatched = ac.nRecoMatched;
            std::printf("=== %s f=%g W=%g ns sigma_t=%g ns: mu=%.3g, photon eff %.3f purity %.3f\n",
                        mode.c_str(), f, w, sigT, mu,
                        nTruthPh ? double(nFound) / nTruthPh : 0.0, nReco ? double(nRecoMatched) / nReco : 0.0);
            for (std::size_t mi = 0; mi < nM; ++mi)
            for (std::size_t ei = 0; ei < nE; ++ei)
            for (int zc = 0; zc < 2; ++zc) {
                const auto k = idx(mi, ei, zc);
                out << mode << "," << f << "," << w << "," << sigT << "," << a.masses[mi] << ","
                    << eBins[ei].first << "," << eBins[ei].second << "," << (zc ? std::to_string(a.zCut) : "none") << ","
                    << mu << "," << a.nTrials;
                for (auto v : win[k]) out << "," << v;
                for (auto v : cand[k]) out << "," << v;
                out << "," << nTruthPh << "," << nReco << ","
                    << (nTruthPh ? double(nFound) / nTruthPh : 0.0) << ","
                    << (nReco ? double(nRecoMatched) / nReco : 0.0) << ","
                    << windowsPerS << "," << double(win[k][1]) / a.nTrials * windowsPerS << "\n";
            }
        }   // sigma_t
    }
    std::printf("wrote %s\n", a.outCsv.c_str());
    return 0;
}
