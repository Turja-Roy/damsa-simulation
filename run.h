#ifndef RUN_H
#define RUN_H

#include "G4UserRunAction.hh"
#include "G4Run.hh"

#include "analysis.h"
#include "FluxData.h"

class DamsaRunAction : public G4UserRunAction {
public:
    DamsaRunAction() {}
    virtual ~DamsaRunAction() {}

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
};

void DamsaRunAction::BeginOfRunAction(const G4Run*) {
    // Reset flux collector at start of each run
    DamsaFluxCollector::Instance()->Reset();
}

void DamsaRunAction::EndOfRunAction(const G4Run* run) {
    // Write analysis histograms (particle spectra at scoring planes)
    DamsaAnalysis::Instance()->WriteROOTHistograms("particle_spectra.root");
    
    // Write flux data for alplib integration
    G4int nEvents = run->GetNumberOfEvent();
    
    // Write detailed CSV files
    DamsaFluxCollector::Instance()->WritePhotonFluxCSV("photon_flux_target_exit.csv");
    DamsaFluxCollector::Instance()->WriteBackgroundCSV("background_target_exit.csv");
    DamsaFluxCollector::Instance()->WriteCSV("all_particles_target_exit.csv");
    
    // Write alplib-compatible flux file (binned spectrum with rate conversion)
    // Default beam current: 62.5 μA
    DamsaFluxCollector::Instance()->WriteAlplibFlux("alplib_photon_flux.csv", nEvents, 62.5e-6);
    
    // Print summary
    G4cout << "\n=== Flux Collection Summary ===" << G4endl;
    G4cout << "Primary events: " << nEvents << G4endl;
    G4cout << "Photons at target exit: " << DamsaFluxCollector::Instance()->GetPhotonCount() << G4endl;
    G4cout << "Neutrons at target exit: " << DamsaFluxCollector::Instance()->GetNeutronCount() << G4endl;
    G4cout << "Total particles recorded: " << DamsaFluxCollector::Instance()->GetTotalParticleCount() << G4endl;
    G4cout << "================================\n" << G4endl;
}

#endif
