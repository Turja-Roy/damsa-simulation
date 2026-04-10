#include <iostream>

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "G4UIExecutive.hh"
#include "G4AnalysisManager.hh"

#include "construction.h"
#include "physics.h"
#include "action.h"
#include "analysis.h"

int main (int argc, char *argv[])
{
    G4RunManager *runManager = new G4RunManager();
    
    runManager->SetUserInitialization(new DamsaDetectorConstruction());
    runManager->SetUserInitialization(new DamsaPhysicsList());
    runManager->SetUserInitialization(new DamsaActionInitialization());

    G4UIExecutive *ui = 0;
    if (argc == 1) ui = new G4UIExecutive(argc, argv);


    G4VisManager *visManager = new G4VisExecutive();
    visManager->Initialize();

    G4UImanager *UIManager = G4UImanager::GetUIpointer();

    if (ui) {
        UIManager->ApplyCommand("/control/execute vis.mac");
        ui->SessionStart();
    }
    else {
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UIManager->ApplyCommand(command+fileName);
    }
    
    // Delete UI and vis manager first so that G4cout is unregistered from the
    // Qt stream buffer before PrintSummary flushes output to the terminal.
    delete ui;
    delete visManager;

    DamsaAnalysis::Instance()->PrintSummary();

    delete runManager;

    return 0;
}
