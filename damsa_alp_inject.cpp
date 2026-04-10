// damsa_alp_inject.cpp — Geant4 propagation of ALP→γγ decay photons.
//
// Reads a CSV produced by scripts/alp_signal_pipeline.py
// (alp_decay_photons_maX.csv) and fires the γγ pairs as primary particles
// at the tungsten target centre, then propagates them through the VDC
// (vacuum decay chamber), magnet region, and CsI calorimeter for full
// detector response simulation.
//
// Usage:
//     ./damsa_alp_inject <decay_csv> [macro.mac] [refire_factor]
//
// Examples:
//     ./damsa_alp_inject output/alp_decay_photons_ma100MeV.csv run_alp.mac
//     ./damsa_alp_inject output/alp_decay_photons_ma50MeV.csv run_alp.mac 5
//     ./damsa_alp_inject output/alp_decay_photons_ma50MeV.csv
//
// If no macro is given, an interactive UI session opens.
// The refire_factor controls how many times each input row is re-fired
// (default: 1). Row weights are divided by this factor. The total beamOn
// count is auto-computed as (n_rows_loaded * refire_factor) and issued
// from here after the macro performs /run/initialize.
//
// The output files written by run.h are prefixed with the basename of the
// input CSV (e.g. "alp_decay_photons_ma100MeV_") so that successive mass
// points do not overwrite each other.

#include <iostream>
#include <string>
#include <cstdlib>

#include "G4MTRunManager.hh"
#include "G4Threading.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "G4UIExecutive.hh"

#include "construction.h"
#include "physics.h"
#include "damsa_config.h"
#include "action.h"
#include "analysis.h"
#include "alp_generator.h"

namespace {
std::string basenameNoExt(const std::string& path) {
    auto slash = path.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    auto dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base;
}
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <alp_decay_photons.csv> [macro.mac] [refire_factor]\n"
                  << "  refire_factor: number of times each row is re-fired (default: 1)\n";
        return 1;
    }

    const std::string csvPath = argv[1];

    // Parse optional refire factor (positional arg 3, or after macro)
    int refireFactor = 1;  // default
    if (argc >= 4) {
        refireFactor = std::atoi(argv[3]);
        if (refireFactor <= 0) {
            std::cerr << "Warning: invalid refire_factor '" << argv[3] 
                      << "', using default 100\n";
            refireFactor = 100;
        }
    }

    // Configure the run BEFORE constructing the action initialization, so
    // that DamsaActionInitialization::Build() picks the ALP generator.
    DamsaConfig::gRunMode       = DamsaConfig::RunMode::ALPInject;
    DamsaConfig::gALPDecayCSV   = csvPath;
    DamsaConfig::gOutputPrefix  = basenameNoExt(csvPath) + "_";
    DamsaConfig::gALPRefireFactor = refireFactor;

    G4cout << "[damsa_alp_inject] CSV          : " << csvPath << G4endl;
    G4cout << "[damsa_alp_inject] Output pfx   : " << DamsaConfig::gOutputPrefix << G4endl;
    G4cout << "[damsa_alp_inject] Refire factor: " << refireFactor << G4endl;

    // Query target centre Z from the detector before handing it to the run manager,
    // so the ALP generator fires from the correct vertex (target centre follows
    // target dimensions automatically via GetTargetCentreZ()).
    auto* detector = new DamsaDetectorConstruction();
    DamsaConfig::gALPVertexZ_cm = detector->GetTargetCentreZ() / cm;
    G4cout << "[damsa_alp_inject] ALP vertex Z: " << DamsaConfig::gALPVertexZ_cm << " cm" << G4endl;

    G4MTRunManager* runManager = new G4MTRunManager();
    runManager->SetNumberOfThreads(G4Threading::G4GetNumberOfCores());
    runManager->SetUserInitialization(detector);
    runManager->SetUserInitialization(new DamsaPhysicsList());
    runManager->SetUserInitialization(new DamsaActionInitialization());
    runManager->Initialize();

    G4UIExecutive* ui = nullptr;
    if (argc < 3) ui = new G4UIExecutive(argc, argv);

    G4VisManager* visManager = new G4VisExecutive();
    visManager->Initialize();

    G4UImanager* UIManager = G4UImanager::GetUIpointer();

    if (ui) {
        UIManager->ApplyCommand("/control/execute vis.mac");
        ui->SessionStart();
    } else {
        G4String command  = "/control/execute ";
        G4String fileName = argv[2];
        UIManager->ApplyCommand(command + fileName);

        // Auto-size beamOn from the number of rows actually loaded by the
        // generator (zero-weight rows have already been dropped in
        // DamsaALPDecayGenerator::LoadEvents). This guarantees every row is
        // fired exactly `refireFactor` times regardless of input CSV size.
        auto* gen = DamsaALPDecayGenerator::Instance();
        if (gen) {
            G4int nBeamOn = gen->GetNEvents() * refireFactor;
            G4cout << "[damsa_alp_inject] Issuing /run/beamOn " << nBeamOn
                   << "  (" << gen->GetNEvents() << " rows × " << refireFactor
                   << " refires)" << G4endl;
            UIManager->ApplyCommand("/run/beamOn " + std::to_string(nBeamOn));
        } else {
            G4cerr << "[damsa_alp_inject] ERROR: ALP generator instance is null "
                   << "after macro execution; no beamOn issued." << G4endl;
        }
    }

    // Delete UI and vis manager first so that G4cout is unregistered from the
    // Qt stream buffer before PrintSummary flushes output to the terminal.
    delete ui;
    delete visManager;

    DamsaAnalysis::Instance()->PrintSummary();

    delete runManager;
    return 0;
}
