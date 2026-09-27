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

struct Cluster1D { double E = 0, pos = 0, t = 0; };

// Local maxima >= seed (strict on the right so a flat top yields one), each
// strip assigned to the nearest maximum. Keeps clusters >= minE.
inline std::vector<Cluster1D> ClusterProjection(const std::vector<Strip>& s,
                                                const Geometry& g, double seed, double minE)
{
    const int n = static_cast<int>(s.size());
    std::vector<int> peaks;
    for (int i = 0; i < n; ++i) {
        const double l = i > 0 ? s[i - 1].E : -1, r = i + 1 < n ? s[i + 1].E : -1;
        if (s[i].E >= seed && s[i].E >= l && s[i].E > r) peaks.push_back(i);
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

}  // namespace calo

#endif
