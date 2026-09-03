#ifndef DAMSA_ALP_DECAY2BODY_H
#define DAMSA_ALP_DECAY2BODY_H

// Isotropic two-body decay with a boost to the lab frame.
// Ports alplib/cross_section_mc.py:495 (Decay2Body.decay) and :669
// (lorentz_boost).

#include <cmath>
#include <random>

namespace damsa::alp {

struct FourVector {
    double e = 0, px = 0, py = 0, pz = 0;
};

// cross_section_mc.py:669. v is the velocity of the NEW frame; Decay2Body
// passes the negated parent 3-velocity, so the sign convention matters.
//
// Faithful to alplib, including its precision limit: gamma = 1/sqrt(1 - beta^2)
// cancels catastrophically as beta -> 1. For a 1 MeV ALP at 8 GeV (gamma 8000)
// 1 - beta^2 ~ 1.6e-8, so gamma keeps only ~8 digits and energy conservation in
// the boosted pair degrades to ~5e-9 relative (measured; see tests/test_alp.cpp).
// Physically irrelevant against a calorimeter resolving ~2%/sqrt(E).
// ponytail: gamma = E_parent/m_parent is exact and would remove this, but it
// would also diverge from alplib and loosen the cross-check. Revisit only if a
// Phase 3 result is sensitive at the 1e-9 level.
inline FourVector LorentzBoost(const FourVector& p, double vx, double vy, double vz)
{
    const double beta = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (beta == 0.0) return p;

    const double nx = vx / beta, ny = vy / beta, nz = vz / beta;
    const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
    const double gb = gamma * beta;
    const double gm1 = gamma - 1.0;
    const double ndotp = nx * p.px + ny * p.py + nz * p.pz;

    FourVector o;
    o.e  = gamma * p.e - gb * ndotp;
    o.px = -gb * nx * p.e + p.px + gm1 * nx * ndotp;
    o.py = -gb * ny * p.e + p.py + gm1 * ny * ndotp;
    o.pz = -gb * nz * p.e + p.pz + gm1 * nz * ndotp;
    return o;
}

// a -> gamma gamma, both daughters massless (m1 = m2 = 0), so p_cm = mp/2.
// The parent is built forward along +z by the caller, as alplib does: its
// simulate_decay_4vectors uses theta = 0 when no ALP angles were simulated.
// Deterministic core: the decay for a given point on the 2-sphere. Split out
// from the sampling so it can be cross-checked against alplib exactly rather
// than only in distribution.
inline void Decay2BodyMasslessAt(const FourVector& parent, double mp,
                                 double phi, double theta,
                                 FourVector& out1, FourVector& out2)
{
    const double pCm = mp / 2.0;                 // (mp^2 - 0)(mp^2 - 0) ^1/2 / (2 mp)
    const double eCm = pCm;                      // massless daughters
    const double st = std::sin(theta), ct = std::cos(theta);

    const FourVector p1cm{eCm,  pCm * std::cos(phi) * st,  pCm * std::sin(phi) * st,  pCm * ct};
    const FourVector p2cm{eCm, -pCm * std::cos(phi) * st, -pCm * std::sin(phi) * st, -pCm * ct};

    // v_in = -parent.get_3velocity()
    const double vx = -parent.px / parent.e;
    const double vy = -parent.py / parent.e;
    const double vz = -parent.pz / parent.e;

    out1 = LorentzBoost(p1cm, vx, vy, vz);
    out2 = LorentzBoost(p2cm, vx, vy, vz);
}

// Sampling wrapper. cross_section_mc.py:529 draws phi uniform on [0, 2pi) and
// theta = arccos(1 - 2u), i.e. uniform on the sphere.
template <class RNG>
inline void Decay2BodyMassless(const FourVector& parent, double mp, RNG& rng,
                               FourVector& out1, FourVector& out2)
{
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double phi   = 2.0 * M_PI * uni(rng);
    const double theta = std::acos(1.0 - 2.0 * uni(rng));
    Decay2BodyMasslessAt(parent, mp, phi, theta, out1, out2);
}

}  // namespace damsa::alp

#endif
