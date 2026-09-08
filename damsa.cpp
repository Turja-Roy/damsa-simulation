#include <cstdlib>
#include <iostream>

#include "G4MTRunManager.hh"
#include "G4Threading.hh"
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
    // Thread count: honour the batch allocation before falling back to the
    // whole machine. G4GetNumberOfCores() reports every core on the node, which
    // on a shared cluster oversubscribes a partial allocation badly.
    // SLURM_CPUS_PER_TASK is set by sbatch -c; G4FORCENUMBEROFTHREADS is
    // Geant4's own override and wins if both are present.
    G4int nThreads = G4Threading::G4GetNumberOfCores();
    if (const char* env = std::getenv("SLURM_CPUS_PER_TASK")) {
        const int n = std::atoi(env);
        if (n > 0) nThreads = n;
    }
    if (const char* env = std::getenv("G4FORCENUMBEROFTHREADS")) {
        const int n = std::atoi(env);
        if (n > 0) nThreads = n;
    }

    G4MTRunManager *runManager = new G4MTRunManager();
    runManager->SetNumberOfThreads(nThreads);
    G4cout << "[damsa] worker threads: " << nThreads << G4endl;
    
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
