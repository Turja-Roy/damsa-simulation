#ifndef DAMSA_CONFIG_H
#define DAMSA_CONFIG_H

// Runtime configuration shared across DamsaActionInitialization,
// DamsaRunAction, and the executables (damsa.cpp, damsa_alp_inject.cpp …).
//
// Two run modes are supported:
//
//   ElectronBeam : standard 8 GeV e- on tungsten dump (used by damsa.cpp,
//                  damsa_gap_scan.cpp, damsa_opt.cpp).  Brems-flux CSVs are
//                  written for downstream alplib processing.
//
//   ALPInject    : a-> gamma gamma decay photons are read from a CSV exported
//                  by scripts/alp_signal_pipeline.py and fired into the
//                  geometry as primary particles.  Used by damsa_alp_inject.cpp
//                  to obtain the calorimeter response to the signal.

#include <string>

namespace DamsaConfig {

enum class RunMode { ElectronBeam, ALPInject };

// All inline so multiple translation units link cleanly without a .cpp.
inline RunMode    gRunMode       = RunMode::ElectronBeam;
inline std::string gALPDecayCSV  = "";   // path to alp_decay_photons_maX.csv
inline std::string gOutputPrefix = "";   // prepended to every CSV/ROOT file in run.h

}  // namespace DamsaConfig

#endif  // DAMSA_CONFIG_H
