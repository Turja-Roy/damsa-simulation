// Self-check for tools/calo_reco.h. Build: calo_reco_test target in CMakeLists.

#include "../tools/calo_reco.h"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace calo;

int main()
{
    const Geometry g;   // 12 strips x 10 mm, centres -55 .. +55 mm

    // Two separated showers -> two clusters, centroids at their peaks.
    std::vector<Strip> s(12);
    auto put = [&](int i, double E, double t) { s[i].E += E; s[i].Et += E * t; };
    put(1, 20, 1.0); put(2, 100, 1.0); put(3, 20, 1.0);
    put(8, 10, 5.0); put(9, 50, 5.0); put(10, 10, 5.0);
    auto c = ClusterProjection(s, g, 5.0, 5.0);
    assert(c.size() == 2);
    assert(std::abs(c[0].pos - g.stripPos(2)) < 1e-9 && std::abs(c[0].E - 140) < 1e-9);
    assert(std::abs(c[1].pos - g.stripPos(9)) < 1e-9 && std::abs(c[1].t - 5.0) < 1e-9);

    // Flat-topped single shower -> one cluster; below-seed profile -> none.
    std::vector<Strip> f(12);
    f[5].E = f[6].E = 30;
    assert(ClusterProjection(f, g, 5.0, 5.0).size() == 1);
    assert(ClusterProjection(f, g, 50.0, 5.0).empty());

    // X/Y pairing by energy rank.
    auto p = PairXY({{10, -30, 0}, {80, 20, 0}}, {{75, 5, 0}, {12, -40, 0}});
    assert(p.size() == 2 && std::abs(p[0].E - 155) < 1e-9 && p[0].x == 20 && p[0].y == 5);

    // Symmetric pair: m = 2 E sin(theta/2), tan(theta/2) = d / L.
    const double E = 1000, d = 50, L = 470;
    const double m = Mgg(E, d, 0, E, -d, 0, L);
    assert(std::abs(m - 2 * E * std::sin(std::atan(d / L))) < 1e-9);

    // ── Depth-aware reconstruction ──
    // Synthetic shower: centre (x0 + tx z, y0 + ty z), energy shared over
    // neighbouring bars with a 4 mm Gaussian, starting at layer `start`.
    auto addShower = [&](std::vector<double>& Ec, std::vector<double>& Tc, double x0, double y0,
                         double tx, double ty, int start, double Elayer, const RecoParams& rp) {
        for (int l = start; l < 44; ++l) {
            const double z = LayerZ(l, rp);
            const double c = (l % 2 == 0) ? y0 + ty * z : x0 + tx * z;   // even layers measure y
            for (int st = 0; st < 12; ++st) {
                const double d = g.stripPos(st) - c;
                const double e = Elayer * std::exp(-d * d / (2 * 16.0));
                if (e > 0.05) { Ec[l * 12 + st] += e; Tc[l * 12 + st] = 1.0; }
            }
        }
    };
    RecoParams rp;
    {
        std::vector<double> Ec(528, 0.0), Tc(528, 0.0);
        addShower(Ec, Tc, 10, -10, 0.05, -0.03, 0, 20, rp);
        auto ph = ReconstructDepth(Ec, Tc, g, rp);
        assert(ph.size() == 1);
        assert(std::abs(ph[0].tx - 0.05) < 0.005 && std::abs(ph[0].ty + 0.03) < 0.005);
        assert(std::abs(ph[0].x - (10 + 0.05 * rp.frontZ)) < 2.0);
        assert(!ph[0].fixedDir && ph[0].nLayers == 44);
    }
    // Two equal-energy showers starting at different depths: rank pairing has no
    // way to tell which x goes with which y; depth pairing must get both right.
    {
        std::vector<double> Ec(528, 0.0), Tc(528, 0.0);
        addShower(Ec, Tc, -35, 35, 0, 0, 1, 20, rp);
        addShower(Ec, Tc, 35, -35, 0, 0, 22, 20, rp);
        auto ph = ReconstructDepth(Ec, Tc, g, rp);
        assert(ph.size() == 2);
        for (const auto& q : ph) assert(std::abs(q.x + q.y) < 4.0);   // (-35,35) or (35,-35), never a ghost
    }

    // Photon ID. A shower: energy rising to a maximum near t_max(E), several bars wide.
    auto addShowerProfile = [&](std::vector<double>& Ec, double x0, double y0, int start, double A = 120) {
        for (int l = start; l < 44; ++l) {
            const double t = (l - start + 0.5) * rp.layerThick / rp.X0;          // depth in X0
            const double Elay = A * std::pow(t, 2.0) * std::exp(-t / 1.2);      // peaks ~2.4 X0 in
            const double c = (l % 2 == 0) ? y0 : x0;
            for (int st = 0; st < 12; ++st) {
                const double d = g.stripPos(st) - c;
                const double e = Elay * std::exp(-d * d / (2 * 64.0));
                if (e > 0.05) Ec[l * 12 + st] += e;
            }
        }
    };
    {   // EM shower from the front -> EM, a photon
        std::vector<double> Ec(528, 0.0), Tc(528, 1.0);
        addShowerProfile(Ec, 5, -5, 1);
        auto ph = ReconstructDepth(Ec, Tc, g, rp);
        assert(ph.size() == 1 && ph[0].cls == kEM && ph[0].photon);
    }
    {   // MIP: 5.6 MeV in one bar per layer through all 44 layers -> MIP, not a photon
        std::vector<double> Ec(528, 0.0), Tc(528, 1.0);
        for (int l = 0; l < 44; ++l) Ec[l * 12 + 6] = 5.6 * (1 + 0.1 * ((l * 7) % 5 - 2) / 2.0);
        auto ph = ReconstructDepth(Ec, Tc, g, rp);
        assert(ph.size() == 1 && ph[0].cls == kMIP && !ph[0].photon);
        assert(ph[0].meanBars < 1.1 && std::abs(ph[0].medLayerE - 5.6) < 0.6);
    }
    {   // Shower starting at layer 20 (14 cm deep: a neutron-like interaction) -> hadronic
        std::vector<double> Ec(528, 0.0), Tc(528, 1.0);
        addShowerProfile(Ec, -20, 20, 20);
        auto ph = ReconstructDepth(Ec, Tc, g, rp);
        assert(ph.size() == 1 && ph[0].cls == kHadronic && !ph[0].photon);
    }
    {   // Si veto: hits on the line in all 6 planes -> charged; backsplash in planes 4-5 only -> not
        std::vector<double> Ec(528, 0.0), Tc(528, 1.0);
        addShowerProfile(Ec, 5, -5, 1);
        auto ph = ReconstructDepth(Ec, Tc, g, rp);
        std::vector<TrackerHit> through, splash;
        for (int i = 0; i < 6; ++i) through.push_back({i, ph[0].x, ph[0].y});
        splash = {{4, ph[0].x, ph[0].y}, {5, ph[0].x + 3, ph[0].y}};
        auto a1 = ph, a2 = ph;
        ApplyVeto(a1, through, rp);
        ApplyVeto(a2, splash, rp);
        assert(a1[0].vetoInfo && a1[0].charged && a1[0].vetoPlanes == 6 && !a1[0].photon);
        assert(!a2[0].charged && a2[0].photon);
        // Standalone Si track recovers the slope.
        std::vector<TrackerHit> slanted;
        for (int i = 0; i < 6; ++i) {
            const double z = TrackerZ(i, rp) - rp.frontZ;
            slanted.push_back({i, ph[0].x + 0.01 * z, ph[0].y - 0.02 * z});
        }
        auto a3 = ph;
        ApplyVeto(a3, slanted, rp);
        assert(a3[0].charged && std::abs(a3[0].siTx - 0.01) < 1e-9 && std::abs(a3[0].siTy + 0.02) < 1e-9);
    }
    {   // Satellite: a 5 % shower starting 7 layers after a big one, 10 cm away: EM on its own
        // (start 8 is within the cut at 75 MeV), dropped as the big shower's offspring
        std::vector<double> Ec(528, 0.0), Tc(528, 1.0);
        addShowerProfile(Ec, -35, -35, 1);
        addShowerProfile(Ec, 35, 35, 8, 6);
        auto ph = ReconstructDepth(Ec, Tc, g, rp);
        int nPh = 0;
        for (const auto& q : ph) nPh += q.photon;
        assert(ph.size() == 2 && nPh == 1);
    }

    // Closest approach: two lines from (5, -3, -450) meet there.
    {
        const double zv = -450, z0 = 20;
        const double t1x = 0.02, t1y = 0.01, t2x = -0.03, t2y = 0.04;
        const auto v = ClosestApproach(5 + t1x * (z0 - zv), -3 + t1y * (z0 - zv), t1x, t1y,
                                       5 + t2x * (z0 - zv), -3 + t2y * (z0 - zv), t2x, t2y, z0);
        assert(std::abs(v.z - zv) < 1e-6 && std::abs(v.x - 5) < 1e-6 && v.dist < 1e-6);
    }

    std::printf("calo_reco self-check: all passed\n");
    return 0;
}
