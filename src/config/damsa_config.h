#ifndef DAMSA_CONFIG_H
#define DAMSA_CONFIG_H

// Runtime configuration shared across DamsaActionInitialization,
// DamsaRunAction, and the executables (damsa.cpp, damsa_alp_inject.cpp …).
//
// Two run modes are supported:
//
//   ElectronBeam : standard 8 GeV e- on tungsten dump (used by damsa.cpp,
//                  damsa_opt.cpp).  Brems-flux CSVs are
//                  written for downstream alplib processing.
//
//   ALPInject    : a-> gamma gamma decay photons are read from a CSV exported
//                  by scripts/alp_signal_pipeline.py and fired into the
//                  geometry as primary particles.  Used by damsa_alp_inject.cpp
//                  to obtain the calorimeter response to the signal.

#include <string>

namespace DamsaConfig {

enum class RunMode { ElectronBeam, ALPInject };

// ── Beam pulse structure ────────────────────────────────────────────────────
// LESA Table I operating modes (Fermilab-PUB-26-0039).  The flux normalization
// uses the DELIVERED current, derived from the bunch parameters below — NOT the
// "max current" spec cap quoted in the paper.  See plan.md §1-2.
constexpr double kEcharge_C = 1.602176634e-19;   // electron charge [C]

enum class BeamMode { DarkCurrent, LESALaser, XLEAP, Interleaved };

struct BeamSpec {
    double bunchCharge_e;    // electrons per bunch
    double kickerRate_Hz;    // kicker fire rate
    int    bunchesPerKick;   // bunches swept per kick
    double bunchSpacing_s;   // time between bunches

    // Delivered DC-equivalent current [A]. Reproduces the Table I power column.
    double current_A() const {
        return bunchCharge_e * bunchesPerKick * kickerRate_Hz * kEcharge_C;
    }
};

inline BeamSpec BeamSpecFor(BeamMode m) {
    switch (m) {
      //                          q_bunch   rate    n/kick  spacing
      case BeamMode::DarkCurrent: return {0.07,   929e3,  100, 5.4e-9 };
      case BeamMode::LESALaser:   return {4200,   929e3,  18,  26.9e-9};
      case BeamMode::XLEAP:       return {167000, 929e3,  1,   1.08e-6};
      case BeamMode::Interleaved: return {6.2e8,  100,    1,   10e-3  };
    }
    return {4200, 929e3, 18, 26.9e-9};   // = LESALaser
}

inline const char* BeamModeName(BeamMode m) {
    switch (m) {
      case BeamMode::DarkCurrent: return "DarkCurrent";
      case BeamMode::LESALaser:   return "LESALaser";
      case BeamMode::XLEAP:       return "XLEAP";
      case BeamMode::Interleaved: return "Interleaved";
    }
    return "LESALaser";
}

// Accepts short names (dark|lesa|xleap|interleaved). Returns false if unknown.
inline bool ParseBeamMode(const std::string& s, BeamMode& out) {
    if      (s == "dark")        out = BeamMode::DarkCurrent;
    else if (s == "lesa")        out = BeamMode::LESALaser;
    else if (s == "xleap")       out = BeamMode::XLEAP;
    else if (s == "interleaved") out = BeamMode::Interleaved;
    else return false;
    return true;
}

// All inline so multiple translation units link cleanly without a .cpp.
inline RunMode    gRunMode        = RunMode::ElectronBeam;
inline BeamMode   gBeamMode       = BeamMode::LESALaser;  // flux normalization mode
inline std::string gALPDecayCSV   = "";     // path to alp_decay_photons_maX.csv
inline std::string gOutputPrefix  = "";     // prepended to every CSV/ROOT file in run.h
inline double     gALPVertexZ_cm  = -45.0; // ALP production vertex z (target centre, cm)
inline int        gALPRefireFactor = 1;    // number of times each ALP row is re-fired

}  // namespace DamsaConfig

#endif  // DAMSA_CONFIG_H
