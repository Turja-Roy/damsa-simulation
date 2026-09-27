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

    std::printf("calo_reco self-check: all passed\n");
    return 0;
}
