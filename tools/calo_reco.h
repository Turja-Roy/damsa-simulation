#ifndef DAMSA_CALO_RECO_H
#define DAMSA_CALO_RECO_H

// Calorimeter reconstruction core (Random-2-photons plan §5.4-5.6), kept apart
// from the driver so tests/test_calo_reco.cpp can exercise it.
//
// Readout: crystal bars, cellID = layer*nPerLayer + strip (construction.cpp).
// Even layers measure y, odd layers measure x. Each projection is reduced to a
// 1D strip profile (energy summed over its layers), clustered around local
// maxima, and X/Y clusters are paired into photon candidates.

#include <algorithm>
#include <cmath>
#include <array>
#include <numeric>
#include <vector>

namespace calo {

struct Geometry {
    int    nPerLayer = 12;
    double caloXY_mm = 120.0;
    double pitch() const { return caloXY_mm / nPerLayer; }
    double stripPos(int s) const { return -caloXY_mm / 2 + pitch() * (s + 0.5); }
};

// Summed over the layers of one projection.
struct Strip { double E = 0, Et = 0; };   // Et = sum(E * t)

struct Cluster1D { double E = 0, pos = 0, t = 0; int lo = 1 << 30, hi = -1; };   // lo/hi: strips assigned

// Local maxima >= seed (strict on the right so a flat top yields one), each
// strip assigned to the nearest maximum. Keeps clusters >= minE.
// minSep: peaks closer than this many strips keep only the higher one (1 = no merging).
inline std::vector<Cluster1D> ClusterProjection(const std::vector<Strip>& s,
                                                const Geometry& g, double seed, double minE,
                                                int minSep = 1)
{
    const int n = static_cast<int>(s.size());
    std::vector<int> peaks;
    for (int i = 0; i < n; ++i) {
        const double l = i > 0 ? s[i - 1].E : -1, r = i + 1 < n ? s[i + 1].E : -1;
        if (s[i].E >= seed && s[i].E >= l && s[i].E > r) peaks.push_back(i);
    }
    if (minSep > 1) {
        std::vector<int> byE = peaks, kept;
        std::sort(byE.begin(), byE.end(), [&](int a, int b) { return s[a].E > s[b].E; });
        for (int pk : byE) {
            bool near = false;
            for (int k : kept) near = near || std::abs(pk - k) < minSep;
            if (!near) kept.push_back(pk);
        }
        std::sort(kept.begin(), kept.end());
        peaks.swap(kept);
    }
    std::vector<Cluster1D> out(peaks.size());
    std::vector<double> sumX(peaks.size(), 0.0);
    for (int i = 0; i < n && !peaks.empty(); ++i) {
        if (s[i].E <= 0) continue;
        std::size_t best = 0;
        for (std::size_t k = 1; k < peaks.size(); ++k)
            if (std::abs(i - peaks[k]) < std::abs(i - peaks[best])) best = k;
        out[best].E += s[i].E;
        out[best].t += s[i].Et;
        out[best].lo = std::min(out[best].lo, i);
        out[best].hi = std::max(out[best].hi, i);
        sumX[best]  += s[i].E * g.stripPos(i);
    }
    std::vector<Cluster1D> kept;
    for (std::size_t k = 0; k < out.size(); ++k) {
        if (out[k].E < minE) continue;
        out[k].pos = sumX[k] / out[k].E;
        out[k].t  /= out[k].E;
        kept.push_back(out[k]);
    }
    return kept;
}

struct Photon { double E, x, y, t; };

// X/Y pairing by energy rank (x and y halves of one shower carry
// similar energy). Fails for two photons of similar energy -- those become
// ghosts, which the truth matching counts (FP_ghost). Upgrade: depth-profile
// matching (plan §5.1) if the ghost rate matters.
inline std::vector<Photon> PairXY(std::vector<Cluster1D> xs, std::vector<Cluster1D> ys)
{
    auto byE = [](const Cluster1D& a, const Cluster1D& b) { return a.E > b.E; };
    std::sort(xs.begin(), xs.end(), byE);
    std::sort(ys.begin(), ys.end(), byE);
    std::vector<Photon> out;
    for (std::size_t k = 0; k < std::min(xs.size(), ys.size()); ++k)
        out.push_back({xs[k].E + ys[k].E, xs[k].pos, ys[k].pos,
                       (xs[k].t * xs[k].E + ys[k].t * ys[k].E) / (xs[k].E + ys[k].E)});
    return out;
}

// Invariant mass of two massless photons from one assumed vertex on the beam
// axis, at lever arm L to the calo face.
inline double Mgg(double E1, double x1, double y1, double E2, double x2, double y2, double L)
{
    const double n1 = std::sqrt(x1 * x1 + y1 * y1 + L * L);
    const double n2 = std::sqrt(x2 * x2 + y2 * y2 + L * L);
    const double c = (x1 * x2 + y1 * y2 + L * L) / (n1 * n2);
    return std::sqrt(std::max(2.0 * E1 * E2 * (1.0 - c), 0.0));
}

// ══ Depth-aware reconstruction (Calo-Reco-Explained §2.13) ═══════════════════
//
// Energy is part of every step, not a refinement:
//   1. per layer, the 12 bar energies form a 1D histogram; each peak (+ the
//      strips assigned to it) is a layer cluster with an energy-weighted
//      centroid -- finer than the 1 cm bar;
//   2. per view (x = odd layers, y = even layers) clusters are linked through
//      depth: from a seed, the next layer of the same view must hold a cluster
//      inside a road that narrows as the track grows (a single point allows
//      +-roadSeed; with >= 2 points the road is the fit prediction +- 3 sigma),
//      tolerating up to maxMiss empty layers;
//   3. each view track gets an energy-weighted straight-line fit u(z) = a + b z
//      with a covariance, so a longer track is a more certain one;
//   4. x and y tracks are paired by depth (start layer, energy-weighted depth)
//      and x/y energy balance; inconsistent pairs are vetoed. This replaces
//      the energy-rank pairing that made ghosts.
// useEnergy = false is the boolean readout (lit / not lit): centroids and fit
// weights ignore energy, and pairing uses depth only. Reported energies are
// always the real sums.

struct RecoParams {
    double seed = 0.5;          // MeV, per-layer peak seed
    int    minPeakSep = 4;      // strips: closer peaks in one layer are one shower (~1 Moliere radius)
    double trackSeed = 2.0;     // MeV: clusters below this seed only if isolated (no track within haloRadius)
    double mergeDist = 36;      // mm (~1 Moliere radius): same-view tracks this close are one shower
    double maxSlope = 0.5;      // |du/dz|: a photon from upstream cannot be steeper inside this calo
    double haloRadius = 36;     // mm (~ Moliere radius): leftover clusters add energy to the nearest track;
                                // doubled around showers (>= minShowerClusters), whose halo spreads wider at depth
    double roadSeed = 1.5;      // pitches, road half-width from a single point
    double roadMin = 1.0;       // pitches, narrowest road
    int    maxMiss = 2;         // empty same-view layers tolerated inside a track
    double maxStartGap = 3;     // layers, x/y shower-start difference
    double maxDepthGap = 40;    // mm, x/y energy-weighted depth difference
    double maxBalance = 0.8;    // |Ex-Ey|/(Ex+Ey)
    int    minShowerClusters = 3;  // the three vetoes apply only if both view tracks have this many
                                   // layer clusters: a soft photon (1-2 Compton electrons, 1-4 bars)
                                   // has no shower shape, so depth and balance say nothing about it
    double layerThick = 10;     // mm
    double frontZ = 20.1;       // mm, calo front face (world z)
    double vertexZ = -450;      // mm, assumed origin for single-layer tracks
    double critE = 11.2;        // MeV, CsI critical energy
    double X0 = 18.6;           // mm, CsI radiation length
    bool   useEnergy = true;
};

struct LayerCluster { int layer; double u, E, t; };

struct Track2D {
    std::vector<int> idx;            // into the view's LayerCluster list
    double a = 0, b = 0;             // u(z) = a + b z   (z in mm, world)
    double caa = 0, cab = 0, cbb = 0;  // fit covariance
    double E = 0, t = 0, zMean = 0;  // energy (fitted clusters), E-weighted time and depth
    double Ehalo = 0;                // leftover nearby clusters: energy only, not in the fit
    int first = 0, last = 0;         // layers
    bool fixedDir = false;           // single layer: direction from the assumed vertex
};

struct Photon3D {
    double E, x, y, t;               // at the calo front face
    double tx, ty;                   // slopes dx/dz, dy/dz
    double sx, sy, stx, sty;         // 1-sigma: position at face, slopes
    int nLayers, start, tmaxLayer;
    double balance, tmaxPull;        // energy consistency (reported)
    bool fixedDir;
};

inline double LayerZ(int layer, const RecoParams& p) { return p.frontZ + p.layerThick * (layer + 0.5); }

// Step 1. Cells -> per-layer clusters, split by view (0 = y from even layers, 1 = x from odd).
inline std::array<std::vector<LayerCluster>, 2>
LayerClusters(const std::vector<double>& E, const std::vector<double>& T, const Geometry& g,
              const RecoParams& p)
{
    std::array<std::vector<LayerCluster>, 2> out;
    const int n = g.nPerLayer, nLayers = static_cast<int>(E.size()) / n;
    for (int l = 0; l < nLayers; ++l) {
        std::vector<Strip> real(n), used(n);
        for (int s = 0; s < n; ++s) {
            const double e = E[l * n + s];
            real[s] = {e, e * T[l * n + s]};
            used[s] = p.useEnergy ? real[s] : Strip{e > 0 ? 1.0 : 0.0, e > 0 ? T[l * n + s] : 0.0};
        }
        for (const auto& c : ClusterProjection(used, g, p.useEnergy ? p.seed : 0.5, p.useEnergy ? p.seed : 0.5,
                                               p.minPeakSep)) {
            double e = 0, et = 0;
            for (int s = c.lo; s <= c.hi; ++s) { e += real[s].E; et += real[s].Et; }
            if (e <= 0) continue;
            out[l % 2].push_back({l, c.pos, e, et / e});
        }
    }
    return out;
}

// Step 3. Weighted straight-line fit, point sigma = pitch/sqrt(12)/sqrt(w).
inline void FitTrack(Track2D& tr, const std::vector<LayerCluster>& cl, const Geometry& g, const RecoParams& p)
{
    double W = 0, Wz = 0, Wzz = 0, Wu = 0, Wzu = 0, Et = 0;
    const double s0 = g.pitch() / std::sqrt(12.0);
    tr.E = 0;
    for (int i : tr.idx) tr.E += cl[i].E;
    for (int i : tr.idx) {
        const double z = LayerZ(cl[i].layer, p), u = cl[i].u;
        const double w = p.useEnergy ? cl[i].E / tr.E * tr.idx.size() : 1.0;   // mean weight 1
        W += w; Wz += w * z; Wzz += w * z * z; Wu += w * u; Wzu += w * z * u;
        Et += cl[i].E * cl[i].t;
    }
    tr.t = Et / tr.E;
    tr.zMean = 0;
    for (int i : tr.idx) tr.zMean += cl[i].E * LayerZ(cl[i].layer, p);
    tr.zMean /= tr.E;
    tr.first = cl[tr.idx.front()].layer;
    tr.last  = cl[tr.idx.back()].layer;

    const double D = W * Wzz - Wz * Wz;
    if (tr.idx.size() < 2 || D <= 0) {
        // One layer: no direction from the data. Point back to the assumed vertex
        // with a slope error of one pitch over the calo depth.
        const double z = LayerZ(cl[tr.idx[0]].layer, p), u = cl[tr.idx[0]].u;
        tr.b = u / (z - p.vertexZ);
        tr.a = u - tr.b * z;
        tr.fixedDir = true;
        const double sb = g.pitch() / (z - p.vertexZ);
        tr.cbb = sb * sb; tr.cab = -z * tr.cbb; tr.caa = s0 * s0 + z * z * tr.cbb;
        return;
    }
    tr.b = (W * Wzu - Wz * Wu) / D;
    tr.a = (Wu - tr.b * Wz) / W;
    double chi2 = 0;
    for (int i : tr.idx) {
        const double z = LayerZ(cl[i].layer, p), r = cl[i].u - tr.a - tr.b * z;
        const double w = p.useEnergy ? cl[i].E / tr.E * tr.idx.size() : 1.0;
        chi2 += w * r * r / (s0 * s0);
    }
    const int ndf = static_cast<int>(tr.idx.size()) - 2;
    const double scale = s0 * s0 * (ndf > 0 ? std::max(1.0, chi2 / ndf) : 1.0);
    tr.caa = scale * Wzz / D; tr.cab = -scale * Wz / D; tr.cbb = scale * W / D;
}

// Step 2. Road following through the layers of one view.
// Seeds are the most energetic clusters first (the shower core), and each
// track is extended both forward and backward in depth. A weak cluster
// (< trackSeed) seeds only if it is isolated -- no existing track within
// haloRadius at its depth -- so soft photons still get a track, while the
// fluctuating halo around a shower becomes that shower's energy instead.
inline std::vector<Track2D> FindTracks(const std::vector<LayerCluster>& cl, const Geometry& g,
                                       const RecoParams& p)
{
    std::vector<int> order(cl.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) { return cl[i].E > cl[j].E; });
    std::vector<char> used(cl.size(), 0);
    std::vector<Track2D> tracks;
    const double seedE = p.useEnergy ? p.trackSeed : 0.0;   // boolean readout: any lit cluster seeds

    auto extend = [&](Track2D& tr, int dir) {
        int layer = dir > 0 ? cl[tr.idx.back()].layer : cl[tr.idx.front()].layer, miss = 0;
        while (miss <= p.maxMiss && layer >= -2 && layer < 1000) {
            layer += 2 * dir;   // next layer of the same view
            const int endIdx = dir > 0 ? tr.idx.back() : tr.idx.front();
            double pred = cl[endIdx].u, road = p.roadSeed * g.pitch();
            if (tr.idx.size() >= 2) {
                FitTrack(tr, cl, g, p);
                const double z = LayerZ(layer, p);
                pred = tr.a + tr.b * z;
                const double var = tr.caa + 2 * z * tr.cab + z * z * tr.cbb;
                road = std::max(p.roadMin * g.pitch(), 3.0 * std::sqrt(std::max(var, 0.0)));
            }
            int best = -1;
            for (std::size_t i = 0; i < cl.size(); ++i)
                if (!used[i] && cl[i].layer == layer && std::abs(cl[i].u - pred) < road &&
                    (best < 0 || std::abs(cl[i].u - pred) < std::abs(cl[best].u - pred)))
                    best = static_cast<int>(i);
            if (best >= 0) {
                if (dir > 0) tr.idx.push_back(best); else tr.idx.insert(tr.idx.begin(), best);
                FitTrack(tr, cl, g, p);
                if (std::abs(tr.b) > p.maxSlope) {   // zig-zag through fragments: refuse it
                    if (dir > 0) tr.idx.pop_back(); else tr.idx.erase(tr.idx.begin());
                    best = -1;
                }
            }
            if (best >= 0) { used[best] = 1; miss = 0; } else ++miss;
        }
    };

    auto haloR = [&](const Track2D& t) {
        return t.idx.size() >= static_cast<std::size_t>(p.minShowerClusters) ? 2 * p.haloRadius : p.haloRadius;
    };
    for (int s : order) {
        if (used[s]) continue;
        if (cl[s].E < seedE) {
            const double z = LayerZ(cl[s].layer, p);
            bool nearTrack = false;
            for (const auto& t : tracks) nearTrack = nearTrack || std::abs(cl[s].u - (t.a + t.b * z)) < haloR(t);
            if (nearTrack) continue;
        }
        Track2D tr;
        tr.idx = {s};
        used[s] = 1;
        extend(tr, +1);
        extend(tr, -1);
        FitTrack(tr, cl, g, p);
        tracks.push_back(std::move(tr));
    }

    // Merge fragments of one shower: two tracks whose lines are within mergeDist
    // at the energy-weighted depth of the weaker one.
    for (bool merged = true; merged;) {
        merged = false;
        for (std::size_t i = 0; i < tracks.size() && !merged; ++i)
            for (std::size_t j = i + 1; j < tracks.size() && !merged; ++j) {
                const Track2D& w = tracks[i].E < tracks[j].E ? tracks[i] : tracks[j];
                const double z = w.zMean;
                if (std::abs((tracks[i].a + tracks[i].b * z) - (tracks[j].a + tracks[j].b * z)) > p.mergeDist) continue;
                Track2D m = tracks[i];
                m.idx.insert(m.idx.end(), tracks[j].idx.begin(), tracks[j].idx.end());
                std::sort(m.idx.begin(), m.idx.end(), [&](int a, int b) { return cl[a].layer < cl[b].layer; });
                FitTrack(m, cl, g, p);
                tracks[i] = std::move(m);
                tracks.erase(tracks.begin() + j);
                merged = true;
            }
    }

    // Halo: clusters in no track add their energy to the nearest track within haloRadius.
    for (std::size_t i = 0; i < cl.size(); ++i) {
        if (used[i]) continue;
        const double z = LayerZ(cl[i].layer, p);
        int best = -1; double bd = 1e9;
        for (std::size_t t = 0; t < tracks.size(); ++t) {
            const double d = std::abs(cl[i].u - (tracks[t].a + tracks[t].b * z));
            if (d < haloR(tracks[t]) && d < bd) { bd = d; best = static_cast<int>(t); }
        }
        if (best >= 0) tracks[best].Ehalo += cl[i].E;
    }
    return tracks;
}

// Step 4. x/y pairing by depth and energy balance (min total cost).
// ponytail: brute force over permutations up to 7 tracks per view, greedy above.
inline std::vector<Photon3D> MatchViews(const std::vector<Track2D>& xs, const std::vector<Track2D>& ys,
                                        const std::vector<LayerCluster>& xcl,
                                        const std::vector<LayerCluster>& ycl, const RecoParams& p)
{
    auto cost = [&](const Track2D& x, const Track2D& y) {
        const double startGap = std::abs(x.first - y.first);
        const double depthGap = std::abs(x.zMean - y.zMean);
        const double ex = x.E + x.Ehalo, ey = y.E + y.Ehalo;
        const double bal = std::abs(ex - ey) / (ex + ey);
        const bool showers = std::min(x.idx.size(), y.idx.size()) >= static_cast<std::size_t>(p.minShowerClusters);
        if (showers && (startGap > p.maxStartGap || depthGap > p.maxDepthGap || (p.useEnergy && bal > p.maxBalance)))
            return 1e9;   // vetoed
        return startGap / p.maxStartGap + depthGap / p.maxDepthGap + (p.useEnergy ? bal / p.maxBalance : 0.0);
    };
    std::vector<std::pair<int, int>> pairs;
    const bool swapViews = xs.size() > ys.size();
    const auto& A = swapViews ? ys : xs;
    const auto& B = swapViews ? xs : ys;
    if (B.size() <= 7) {
        std::vector<int> perm(B.size());
        std::iota(perm.begin(), perm.end(), 0);
        double best = 1e18;
        std::vector<int> bestPerm;
        do {
            double c = 0;
            for (std::size_t i = 0; i < A.size(); ++i) {
                const double ci = swapViews ? cost(B[perm[i]], A[i]) : cost(A[i], B[perm[i]]);
                c += std::min(ci, 50.0);   // a vetoed pair costs 50: leaving it unpaired beats it
            }
            if (c < best) { best = c; bestPerm = perm; }
        } while (std::next_permutation(perm.begin(), perm.end()));
        for (std::size_t i = 0; i < A.size() && !bestPerm.empty(); ++i) pairs.push_back({int(i), bestPerm[i]});
    } else {
        std::vector<char> taken(B.size(), 0);
        for (std::size_t i = 0; i < A.size(); ++i) {
            int bj = -1; double bc = 1e9;
            for (std::size_t j = 0; j < B.size(); ++j) {
                if (taken[j]) continue;
                const double c = swapViews ? cost(B[j], A[i]) : cost(A[i], B[j]);
                if (c < bc) { bc = c; bj = static_cast<int>(j); }
            }
            if (bj >= 0) { taken[bj] = 1; pairs.push_back({int(i), bj}); }
        }
    }

    std::vector<Photon3D> out;
    for (auto [ia, ib] : pairs) {
        const Track2D& x = swapViews ? B[ib] : A[ia];
        const Track2D& y = swapViews ? A[ia] : B[ib];
        if (cost(x, y) >= 1e9) continue;
        Photon3D ph{};
        ph.E = x.E + x.Ehalo + y.E + y.Ehalo;
        ph.tx = x.b; ph.ty = y.b;
        ph.x = x.a + x.b * p.frontZ;
        ph.y = y.a + y.b * p.frontZ;
        ph.t = (x.t * x.E + y.t * y.E) / (x.E + y.E);
        ph.sx = std::sqrt(std::max(0.0, x.caa + 2 * p.frontZ * x.cab + p.frontZ * p.frontZ * x.cbb));
        ph.sy = std::sqrt(std::max(0.0, y.caa + 2 * p.frontZ * y.cab + p.frontZ * p.frontZ * y.cbb));
        ph.stx = std::sqrt(x.cbb); ph.sty = std::sqrt(y.cbb);
        ph.nLayers = static_cast<int>(x.idx.size() + y.idx.size());
        ph.start = std::min(x.first, y.first);
        ph.fixedDir = x.fixedDir || y.fixedDir;
        ph.balance = std::abs(x.E + x.Ehalo - y.E - y.Ehalo) / ph.E;
        // Longitudinal consistency: depth of the most energetic layer vs the
        // shower-maximum expectation t_max = ln(E/E_c) - 0.5 radiation lengths.
        double emax = -1;
        for (int i : x.idx) if (xcl[i].E > emax) { emax = xcl[i].E; ph.tmaxLayer = xcl[i].layer; }
        for (int i : y.idx) if (ycl[i].E > emax) { emax = ycl[i].E; ph.tmaxLayer = ycl[i].layer; }
        const double tMeas = (ph.tmaxLayer + 0.5) * p.layerThick / p.X0;
        const double tExp = std::max(0.0, std::log(ph.E / p.critE) - 0.5);
        ph.tmaxPull = tMeas - tExp;   // in X0
        out.push_back(ph);
    }
    return out;
}

// Full chain for one event / overlay: cells (E, T per cellID) -> photons.
inline std::vector<Photon3D> ReconstructDepth(const std::vector<double>& E, const std::vector<double>& T,
                                              const Geometry& g, const RecoParams& p)
{
    const auto cl = LayerClusters(E, T, g, p);
    const auto ys = FindTracks(cl[0], g, p), xs = FindTracks(cl[1], g, p);
    return MatchViews(xs, ys, cl[1], cl[0], p);
}

// Closest approach of two straight lines (point + slopes at z): returns the
// midpoint and the distance between the lines there.
struct Vertex { double x, y, z, dist; };
inline Vertex ClosestApproach(double x1, double y1, double tx1, double ty1,
                              double x2, double y2, double tx2, double ty2, double z0)
{
    // Line i: P_i + s (tx_i, ty_i, 1), P_i at z = z0.
    const double d1[3] = {tx1, ty1, 1}, d2[3] = {tx2, ty2, 1};
    const double w[3] = {x1 - x2, y1 - y2, 0};
    auto dot = [](const double* u, const double* v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; };
    const double a = dot(d1, d1), b = dot(d1, d2), c = dot(d2, d2), d = dot(d1, w), e = dot(d2, w);
    const double den = a * c - b * b;
    const double s = den > 1e-15 ? (b * e - c * d) / den : 0.0;
    const double t = den > 1e-15 ? (a * e - b * d) / den : e / c;
    const double p1[3] = {x1 + s * d1[0], y1 + s * d1[1], z0 + s};
    const double p2[3] = {x2 + t * d2[0], y2 + t * d2[1], z0 + t};
    return {(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2,
            std::sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                      (p1[2] - p2[2]) * (p1[2] - p2[2]))};
}

}  // namespace calo

#endif
