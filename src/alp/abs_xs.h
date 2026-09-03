#ifndef DAMSA_ALP_ABS_XS_H
#define DAMSA_ALP_ABS_XS_H

// Photon absorption cross section, ported from alplib/photon_xs.py:18
// (AbsCrossSection). Reads alplib's own data table in place — no copy, so it
// stays in sync if alplib is ever updated.

#include "constants.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace damsa::alp {

class AbsCrossSection {
public:
    // xsDim: barns -> cm^2. photon_xs.py overrides this only for compound
    // materials (NaI, CsI, CH2, N2, O2, TeO2); elemental W keeps 1e-24.
    explicit AbsCrossSection(const std::string& dataPath, double xsDim = 1e-24)
    : fXsDim(xsDim)
    {
        std::ifstream f(dataPath);
        if (!f.is_open())
            throw std::runtime_error("AbsCrossSection: cannot open " + dataPath);

        // np.genfromtxt(..., skip_header=3)
        std::string line;
        for (int i = 0; i < 3 && std::getline(f, line); ++i) {}

        // cleanPEData(): np.unique(col0, return_index=True)[1] keeps the FIRST
        // occurrence of each energy and returns them sorted by energy. The file
        // has duplicate energies at absorption edges (105 lines -> 94 rows), so
        // this picks the low-side value at each edge. std::map does both.
        std::map<double, double> unique;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            double e = 0, xs = 0;
            if (!(ss >> e >> xs)) continue;
            unique.emplace(e, xs);      // emplace keeps the first insertion
        }
        if (unique.empty())
            throw std::runtime_error("AbsCrossSection: no data in " + dataPath);

        for (const auto& [e, xs] : unique) {
            fLogE.push_back(std::log10(e));
            fLogXs.push_back(std::log10(fXsDim * xs));
        }
    }

    // photon_xs.py:45
    //   10**interp(log10(E), log10(E_i), log10(xs_dim*xs_i), left=0.0, right=0.0)
    //
    // The left/right fills are applied in LOG space, so out-of-range returns
    // 10**0 == 1.0 cm^2, not zero. That is alplib's behaviour, reproduced here.
    // In practice the table spans 1e-3..1e5 MeV so bremsstrahlung never leaves it.
    double SigmaCm2(double E) const
    {
        return std::pow(10.0, Interp(std::log10(E)));
    }

    double SigmaMeV(double E) const { return SigmaCm2(E) / kMeV2Cm2; }

    std::size_t size() const { return fLogE.size(); }

private:
    // np.interp: constant fill outside, linear between.
    double Interp(double x) const
    {
        if (x < fLogE.front() || x > fLogE.back()) return 0.0;   // left / right
        const auto it = std::upper_bound(fLogE.begin(), fLogE.end(), x);
        if (it == fLogE.end()) return fLogXs.back();
        const std::size_t i = std::distance(fLogE.begin(), it);
        if (i == 0) return fLogXs.front();
        const double x0 = fLogE[i - 1], x1 = fLogE[i];
        const double y0 = fLogXs[i - 1], y1 = fLogXs[i];
        if (x1 == x0) return y0;
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }

    double fXsDim;
    std::vector<double> fLogE, fLogXs;
};

}  // namespace damsa::alp

#endif
