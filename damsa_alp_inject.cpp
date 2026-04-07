// damsa_alp_inject.cpp — Geant4 propagation of ALP→γγ decay photons.
//
// Reads a CSV produced by scripts/alp_signal_pipeline.py
// (alp_decay_photons_maX.csv) and fires the γγ pairs as primary particles
// at the tungsten target centre, then propagates them through the magnet,
// air gap, and CsI calorimeter for full detector response simulation.
//
// Usage:
//     ./damsa_alp_inject <decay_csv> [macro.mac]
//
// Examples:
//     ./damsa_alp_inject output/alp_decay_photons_ma100MeV.csv run.mac
//     ./damsa_alp_inject output/alp_decay_photons_ma50MeV.csv
//
// If no macro is given, an interactive UI session opens.
//
// The output files written by run.h are prefixed with the basename of the
// input CSV (e.g. "alp_decay_photons_ma100MeV_") so that successive mass
// points do not overwrite each other.

#include <iostream>
#include <string>

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "G4UIExecutive.hh"

#include "construction.h"
#include "physics.h"
#include "damsa_config.h"
#include "action.h"
#include "analysis.h"

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
                  << " <alp_decay_photons.csv> [macro.mac]\n";
        return 1;
    }

    const std::string csvPath = argv[1];

    // Configure the run BEFORE constructing the action initialization, so
    // that DamsaActionInitialization::Build() picks the ALP generator.
    DamsaConfig::gRunMode      = DamsaConfig::RunMode::ALPInject;
    DamsaConfig::gALPDecayCSV  = csvPath;
    DamsaConfig::gOutputPrefix = basenameNoExt(csvPath) + "_";

    G4cout << "[damsa_alp_inject] CSV         : " << csvPath << G4endl;
    G4cout << "[damsa_alp_inject] Output pfx  : " << DamsaConfig::gOutputPrefix << G4endl;

    G4RunManager* runManager = new G4RunManager();
    runManager->SetUserInitialization(new DamsaDetectorConstruction());
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
    }

    DamsaAnalysis::Instance()->PrintSummary();

    delete ui;
    delete visManager;
    delete runManager;
    return 0;
}
