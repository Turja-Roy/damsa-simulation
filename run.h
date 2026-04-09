#ifndef RUN_H
#define RUN_H

#include "G4UserRunAction.hh"
#include "G4Run.hh"

#include "analysis.h"
#include "FluxData.h"
#include "damsa_config.h"

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
    const std::string& prefix = DamsaConfig::gOutputPrefix;
    const bool isALPInject = (DamsaConfig::gRunMode == DamsaConfig::RunMode::ALPInject);

    // Write analysis histograms (particle spectra at scoring planes)
    DamsaAnalysis::Instance()->WriteROOTHistograms(prefix + "particle_spectra.root");

    // Write flux data for alplib integration
    G4int nEvents = run->GetNumberOfEvent();

    // Write detailed CSV files (exit photons — for background studies)
    DamsaFluxCollector::Instance()->WritePhotonFluxCSV(prefix + "photon_flux_target_exit.csv");
    DamsaFluxCollector::Instance()->WriteBackgroundCSV(prefix + "background_target_exit.csv");
    DamsaFluxCollector::Instance()->WriteCSV(prefix + "all_particles_target_exit.csv");
    
    // Write calo-face particle CSV for SNR analysis
    DamsaFluxCollector::Instance()->WriteCaloFaceCSV(prefix + "calo_face_particles.csv");

    // Brems-flux files only make sense for the electron-beam mode.  In ALP
    // injection mode there are no electrons → these would be empty/zero and
    // would clobber the real flux files used as input to alp_signal_pipeline.py.
    if (!isALPInject) {
        DamsaFluxCollector::Instance()->WriteBremsPhotonFluxCSV(prefix + "brems_photon_flux_target.csv");
        DamsaFluxCollector::Instance()->WriteAlplibFlux(prefix + "alplib_photon_flux_exit.csv", nEvents, 62.5e-6);
        DamsaFluxCollector::Instance()->WriteAlplibBremsFlux(prefix + "alplib_brems_flux.csv", nEvents, 62.5e-6);
    }
    
    // Print summary
    G4cout << "\n=== Flux Collection Summary ===" << G4endl;
    G4cout << "Primary events: " << nEvents << G4endl;
    G4cout << "Photons at target exit: " << DamsaFluxCollector::Instance()->GetPhotonCount() << G4endl;
    G4cout << "Brems photons inside target: " << DamsaFluxCollector::Instance()->GetBremsPhotonCount() << G4endl;
    G4cout << "Neutrons at target exit: " << DamsaFluxCollector::Instance()->GetNeutronCount() << G4endl;
    G4cout << "Total particles at target exit: " << DamsaFluxCollector::Instance()->GetTotalParticleCount() << G4endl;
    G4cout << "Particles at calo face: " << DamsaFluxCollector::Instance()->GetCaloFaceCount() << G4endl;
    G4cout << "================================\n" << G4endl;
}

#endif
