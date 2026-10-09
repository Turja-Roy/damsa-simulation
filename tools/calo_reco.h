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

struct RecoParams {
    double seed = 0.5;          // MeV, per-layer peak seed
    int    minPeakSep = 4;      // strips: closer peaks in one layer are one shower (~1 Moliere radius)
    double trackSeed = 2.0;     // MeV: clusters below this seed only if isolated (no track within haloRadius)
    double mergeDist = 35.3;    // mm = 1 Moliere radius (CsI): same-view tracks this close are one shower
    double maxSlope = 0.5;      // |du/dz|: a photon from upstream cannot be steeper inside this calo
    double haloRadius = 35.3;   // mm = 1 R_M (CsI R_M = 3.53 cm): Moliere mask of a short track; showers
                                // (>= minShowerClusters) mask 2 R_M (95 % of their energy). Clusters inside
                                // a mask add energy to that object; outside every mask they form their own
                                // object, which must pass the photon ID to count as a photon.
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

    // Photon ID (Calo-Reco-Explained §2.14). Defaults from CsI physics; tuned on
    // single-particle templates (jobs/09_templates.sbatch).
    // Shower start: soft photons convert later (longer attenuation length near
    // the pair threshold, and the first cluster must pass the seed), so the cut
    // grows as E falls: start <= emMaxStartLayer + emStartPerDecade * log10(1 GeV / E).
    // Local gamma templates, 95 % of starts: layer 7 (1 GeV), 9 (100 MeV), 14 (20 MeV).
    int    emMaxStartLayer = 6;   // at E >= 1 GeV: 7 cm = 3 conversion lengths (95 % of photons)
    double emStartPerDecade = 4.0;
    double emMaxTmaxPull = 3.0;   // |measured - expected shower-max depth| in X0, showers only
    // Satellite (secondary) showers: an object starting >= satStartGap layers
    // after a more energetic object's start, within satRadius of it and with
    // < satEfrac of its energy is that shower's offspring, not a photon. Without
    // this, the energy-dependent start cut let ~0.15 of them per 1 GeV photon through.
    double satRadius = 106.0;     // mm = 3 R_M
    int    satStartGap = 3;
    double satEfrac = 0.15;
    int    mipMinLayers = 8;      // a MIP crosses many layers ...
    double mipMaxBars = 1.6;      // ... about one bar wide ...
    double mipEmin = 2.0, mipEmax = 15.0;   // ... with ~5.6 MeV per 1 cm bar (median per layer) ...
    double mipMaxPeak = 3.0;      // ... and no shower maximum (max / median per-layer energy)

    // Si charged-particle veto: tracker geometry (construction.cpp) and rule.
    int    nTrackers = 6;
    double magnetZ0 = -100, magnetLen = 120, trackerThick = 2;   // mm, world
    double trackerHalf = 49;      // mm, Si planes are 9.8 x 9.8 cm
    double siRoad = 3;            // mm: Si hit to track line (0.2 cm Si, little scattering)
    double chargedMask = 70.6;    // mm = 2 R_M: calo objects entering this close to a Si track's
                                  // extrapolation are that particle or its spray, not photons
    int    minVetoPlanes = 4;     // a charged particle from upstream crosses all 6 planes; calo
                                  // backsplash hits mostly the last 2-3 (measured on 1 GeV photons)
};

struct LayerCluster { int layer; double u, E, t; int nb; double rms; };   // nb: bars lit, rms: lateral [mm]

struct Track2D {
    std::vector<int> idx;            // into the view's LayerCluster list
    double a = 0, b = 0;             // u(z) = a + b z   (z in mm, world)
    double caa = 0, cab = 0, cbb = 0;  // fit covariance
    double E = 0, t = 0, zMean = 0;  // energy (fitted clusters), E-weighted time and depth
    double Ehalo = 0;                // leftover nearby clusters: energy only, not in the fit
    int first = 0, last = 0;         // layers
    bool fixedDir = false;           // single layer: direction from the assumed vertex
};

enum ObjClass { kEM = 0, kMIP = 1, kHadronic = 2 };

struct Photon3D {                    // a reconstructed calo object (photon candidate if `photon`)
    double E, x, y, t;               // at the calo front face
    double tx, ty;                   // slopes dx/dz, dy/dz
    double sx, sy, stx, sty;         // 1-sigma: position at face, slopes
    int nLayers, start, tmaxLayer;
    double balance, tmaxPull;        // energy consistency
    bool fixedDir;
    // Per-layer energy profile (the per-layer dE/dx) and lateral shape:
    int last = 0, nLit = 0;          // last layer, layers with a cluster
    double medLayerE = 0;            // median energy per lit layer [MeV]
    double layerEcv = 0;             // spread of per-layer energy (std / mean)
    double peakRatio = 0;            // max / median per-layer energy (shower max vs flat MIP)
    double meanBars = 0;             // mean bars lit per layer
    double widthEarly = 0, widthLate = 0;   // mean lateral RMS, first / second half of lit layers [mm]
    int cls = kEM;                   // ObjClass
    int vetoPlanes = 0;              // charged: Si planes on its track
    bool charged = false;            // nearest object to a Si track's extrapolation
    double siX = 0, siY = 0, siTx = 0, siTy = 0;   // charged: its Si track line (at the face)
    bool vetoInfo = false;           // inside the tracker acceptance (entry or line at the first plane)
    bool photon = false;             // EM and not vetoed as charged
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
        std::vector<Strip> real(n);
        for (int s = 0; s < n; ++s) {
            const double e = E[l * n + s];
            real[s] = {e, e * T[l * n + s]};
        }
        for (const auto& c : ClusterProjection(real, g, p.seed, p.seed, p.minPeakSep)) {
            double e = 0, et = 0, var = 0;
            int nb = 0;
            for (int s = c.lo; s <= c.hi; ++s) {
                e += real[s].E; et += real[s].Et;
                nb += real[s].E > 0;
                var += real[s].E * (g.stripPos(s) - c.pos) * (g.stripPos(s) - c.pos);
            }
            if (e <= 0) continue;
            out[l % 2].push_back({l, c.pos, e, et / e, nb, std::sqrt(var / e)});
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
        const double w = cl[i].E / tr.E * tr.idx.size();   // energy weight, mean 1
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
        const double w = cl[i].E / tr.E * tr.idx.size();
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
    const double seedE = p.trackSeed;

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

// Photon ID on the per-layer profile (Calo-Reco-Explained §2.14):
//   MIP track  -- many layers, ~1 bar wide, flat ~5.6 MeV per layer, no maximum
//                (mu, and pi/p crossing without a nuclear interaction);
//   EM shower  -- starts early enough for its energy, and if it is a real shower its
//                most energetic layer sits where t_max = ln(E/E_c) - 0.5 predicts;
//   hadronic   -- everything else (deep or irregular deposits, e.g. n / pi / p interactions).
inline int Classify(const Photon3D& o, const RecoParams& p)
{
    if (o.nLit >= p.mipMinLayers && o.meanBars <= p.mipMaxBars && o.medLayerE >= p.mipEmin &&
        o.medLayerE <= p.mipEmax && o.peakRatio <= p.mipMaxPeak)
        return kMIP;
    const bool shower = o.nLit >= p.minShowerClusters;
    const double maxStart = p.emMaxStartLayer + p.emStartPerDecade * std::max(0.0, std::log10(1000.0 / std::max(o.E, 1.0)));
    if (o.start <= maxStart && (!shower || std::abs(o.tmaxPull) <= p.emMaxTmaxPull)) return kEM;
    return kHadronic;
}

// Si plane centre z (world, mm): N planes in N+1 equal gaps across the magnet region.
inline double TrackerZ(int i, const RecoParams& p)
{
    const double gap = (p.magnetLen - p.nTrackers * p.trackerThick) / (p.nTrackers + 1);
    return p.magnetZ0 + (i + 1) * gap + i * p.trackerThick + p.trackerThick / 2;
}

struct TrackerHit { int plane; double x, y; };

struct SiTrack { double x, y, tx, ty; int n; };   // straight line, position at the calo face

// Standalone Si tracking (no magnetic field in the geometry, so straight lines):
// seed with one hit in the first half of the planes and one in the second, keep
// the line if >= minVetoPlanes planes have a hit within siRoad, refit those hits
// (least squares, x(z) and y(z)), and remove them from the pool.
// ponytail: O(n^3) in hits per window; fine for single events and moderate mu,
// replace by a road search per plane if Si occupancy gets large.
inline std::vector<SiTrack> FindSiTracks(std::vector<TrackerHit> hits, const RecoParams& p)
{
    std::vector<SiTrack> out;
    auto zf = [&](int plane) { return TrackerZ(plane, p) - p.frontZ; };
    for (bool found = true; found;) {
        found = false;
        for (std::size_t a = 0; a < hits.size() && !found; ++a)
        for (std::size_t b = 0; b < hits.size() && !found; ++b) {
            const auto &h1 = hits[a], &h2 = hits[b];
            if (h1.plane >= p.nTrackers / 2 || h2.plane < p.nTrackers / 2) continue;
            const double tx = (h2.x - h1.x) / (zf(h2.plane) - zf(h1.plane)), ty = (h2.y - h1.y) / (zf(h2.plane) - zf(h1.plane));
            std::vector<std::size_t> on;   // nearest hit within siRoad, per plane
            for (int i = 0; i < p.nTrackers; ++i) {
                const double xi = h1.x + tx * (zf(i) - zf(h1.plane)), yi = h1.y + ty * (zf(i) - zf(h1.plane));
                std::size_t best = hits.size();
                double dBest = p.siRoad;
                for (std::size_t k = 0; k < hits.size(); ++k)
                    if (hits[k].plane == i && std::hypot(hits[k].x - xi, hits[k].y - yi) < dBest)
                        { dBest = std::hypot(hits[k].x - xi, hits[k].y - yi); best = k; }
                if (best < hits.size()) on.push_back(best);
            }
            if (static_cast<int>(on.size()) < p.minVetoPlanes) continue;
            double n = on.size(), sz = 0, szz = 0, sx = 0, szx = 0, sy = 0, szy = 0;
            for (auto k : on) {
                const double z = zf(hits[k].plane);
                sz += z; szz += z * z; sx += hits[k].x; szx += z * hits[k].x; sy += hits[k].y; szy += z * hits[k].y;
            }
            const double den = n * szz - sz * sz;
            SiTrack t;
            t.tx = (n * szx - sz * sx) / den; t.x = (sx - t.tx * sz) / n;
            t.ty = (n * szy - sz * sy) / den; t.y = (sy - t.ty * sz) / n;
            t.n = static_cast<int>(on.size());
            out.push_back(t);
            std::sort(on.rbegin(), on.rend());
            for (auto k : on) hits.erase(hits.begin() + static_cast<long>(k));
            found = true;
        }
    }
    return out;
}

// Charged veto: every calo object entering within chargedMask (2 R_M) of a Si
// track's extrapolation to the face is that particle or its spray -> not a
// photon. The nearest one is the charged particle itself (charged, Si line
// stored). Measured on 1 GeV pi+: matching from the calo side instead (a road
// around each object's line) missed the pion whenever its hadronic shower's
// centroid sat > 2 cm off. Calo backsplash lights mostly the last 2 planes, so
// it does not make a track. Objects outside the 9.8 cm planes get
// vetoInfo = false ("no information", not "neutral").
inline void ApplyVeto(std::vector<Photon3D>& objs, const std::vector<SiTrack>& tracks, const RecoParams& p)
{
    const double z0 = TrackerZ(0, p) - p.frontZ;
    for (auto& o : objs)
        o.vetoInfo = (std::abs(o.x) <= p.trackerHalf && std::abs(o.y) <= p.trackerHalf) ||
                     (std::abs(o.x + o.tx * z0) <= p.trackerHalf && std::abs(o.y + o.ty * z0) <= p.trackerHalf);
    for (const auto& t : tracks) {
        Photon3D* near = nullptr;
        for (auto& o : objs) {
            const double d = std::hypot(o.x - t.x, o.y - t.y);
            if (d >= p.chargedMask) continue;
            o.photon = false;
            if (!near || d < std::hypot(near->x - t.x, near->y - t.y)) near = &o;
        }
        if (!near) continue;
        near->charged = true;
        near->vetoPlanes = t.n;
        near->siX = t.x; near->siY = t.y; near->siTx = t.tx; near->siTy = t.ty;
    }
}
inline void ApplyVeto(std::vector<Photon3D>& objs, const std::vector<TrackerHit>& hits, const RecoParams& p)
{
    ApplyVeto(objs, FindSiTracks(hits, p), p);
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
        if (showers && (startGap > p.maxStartGap || depthGap > p.maxDepthGap || bal > p.maxBalance))
            return 1e9;   // vetoed
        return startGap / p.maxStartGap + depthGap / p.maxDepthGap + bal / p.maxBalance;
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
        // From the shower start (the conversion point), not the calo front.
        const double tMeas = (ph.tmaxLayer - ph.start + 0.5) * p.layerThick / p.X0;
        const double tExp = std::max(0.0, std::log(ph.E / p.critE) - 0.5);
        ph.tmaxPull = tMeas - tExp;   // in X0

        // Per-layer profile from both views' clusters (one cluster per layer per track).
        std::vector<std::pair<int, const LayerCluster*>> lc;
        for (int i : x.idx) lc.push_back({xcl[i].layer, &xcl[i]});
        for (int i : y.idx) lc.push_back({ycl[i].layer, &ycl[i]});
        std::sort(lc.begin(), lc.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<double> el;
        double bars = 0, wE = 0, wL = 0; int nE = 0, nL = 0;
        for (std::size_t k = 0; k < lc.size(); ++k) {
            el.push_back(lc[k].second->E);
            bars += lc[k].second->nb;
            if (k < lc.size() / 2 + lc.size() % 2) { wE += lc[k].second->rms; ++nE; } else { wL += lc[k].second->rms; ++nL; }
        }
        ph.nLit = static_cast<int>(lc.size());
        ph.last = lc.back().first;
        ph.meanBars = bars / ph.nLit;
        ph.widthEarly = nE ? wE / nE : 0; ph.widthLate = nL ? wL / nL : 0;
        std::vector<double> sorted = el;
        std::sort(sorted.begin(), sorted.end());
        ph.medLayerE = sorted[sorted.size() / 2];
        double mean = 0, m2 = 0;
        for (double e : el) { mean += e; m2 += e * e; }
        mean /= el.size();
        ph.layerEcv = mean > 0 ? std::sqrt(std::max(0.0, m2 / el.size() - mean * mean)) / mean : 0;
        ph.peakRatio = ph.medLayerE > 0 ? sorted.back() / ph.medLayerE : 0;
        ph.cls = Classify(ph, p);
        ph.photon = ph.cls == kEM;   // the Si veto (ApplyVeto) may still clear it
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
    auto objs = MatchViews(xs, ys, cl[1], cl[0], p);
    for (auto& o : objs)
        for (const auto& b : objs)
            if (&b != &o && o.photon && o.E < p.satEfrac * b.E && o.start >= b.start + p.satStartGap &&
                std::hypot(o.x - b.x, o.y - b.y) < p.satRadius) o.photon = false;
    return objs;
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
