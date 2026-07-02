#ifndef GENERATOR_H
#define GENERATOR_H

#include "G4VUserPrimaryGeneratorAction.hh"

#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "Randomize.hh"

#include "damsa_config.h"
#include "gate_sampler.h"

class DamsaPrimaryGenerator : public G4VUserPrimaryGeneratorAction {
public:
    DamsaPrimaryGenerator();
    virtual ~DamsaPrimaryGenerator();

    virtual void GeneratePrimaries(G4Event*);

    void SetBeamEnergy(G4double energy);
    G4double GetBeamEnergy() const { return fCurrentEnergy; }

private:
    G4ThreeVector SampleBeamSpot() const;   // beam origin, optional transverse spread

    G4ParticleGun* fParticleGun;
    G4double fCurrentEnergy;
    G4double fBeamOriginZ;
};

DamsaPrimaryGenerator::DamsaPrimaryGenerator ()
: fParticleGun(nullptr), fCurrentEnergy(8.*GeV), fBeamOriginZ(-60.*cm) {
    fParticleGun = new G4ParticleGun(1);

    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4String particleName = "e-";
    G4ParticleDefinition* particle = particleTable->FindParticle(particleName);

    G4ThreeVector pos(0., 0., fBeamOriginZ);
    G4ThreeVector mom(0., 0., 1.);

    fParticleGun->SetParticlePosition(pos);
    fParticleGun->SetParticleMomentumDirection(mom);
    fParticleGun->SetParticleEnergy(fCurrentEnergy);
    fParticleGun->SetParticleDefinition(particle);
}
DamsaPrimaryGenerator::~DamsaPrimaryGenerator () {
    delete fParticleGun;
}

G4ThreeVector DamsaPrimaryGenerator::SampleBeamSpot() const {
    const G4double sigma = DamsaConfig::gBeamSpotSigma_mm * mm;
    if (sigma <= 0.) return G4ThreeVector(0., 0., fBeamOriginZ);   // pencil beam
    return G4ThreeVector(G4RandGauss::shoot(0., sigma),
                         G4RandGauss::shoot(0., sigma),
                         fBeamOriginZ);
}

void DamsaPrimaryGenerator::GeneratePrimaries(G4Event* anEvent) {
    // Level A: one electron per event.
    if (!DamsaConfig::gPulsedBeam) {
        DamsaConfig::gElectronsFired += 1;
        fParticleGun->GeneratePrimaryVertex(anEvent);
        return;
    }

    // Level B: one event = one readout gate. Fire every electron in the gate
    // into THIS event so their showers pile up (plan.md §3.2). Each electron
    // carries its bunch's arrival time (b * bunchSpacing) so global times in
    // the scoring output reflect the real intra-gate structure.
    const DamsaConfig::BeamSpec spec = DamsaConfig::BeamSpecFor(DamsaConfig::gBeamMode);
    const std::vector<int> occ = DamsaConfig::SampleGateBunchOccupancies(
        spec, DamsaConfig::gReadoutGate_s, DamsaConfig::gPoissonOccupancy);

    long n = 0;
    for (int nb : occ) n += nb;

    if (n > DamsaConfig::gMaxDirectElectrons) {
        G4ExceptionDescription msg;
        msg << "Gate occupancy " << n << " exceeds gMaxDirectElectrons ("
            << DamsaConfig::gMaxDirectElectrons << ") for beam mode "
            << DamsaConfig::BeamModeName(DamsaConfig::gBeamMode)
            << ". Direct multi-vertex shower simulation is intractable here; use "
               "the overlay/shower-library path instead (plan.md §3.3 Strategy 2).";
        G4Exception("DamsaPrimaryGenerator::GeneratePrimaries", "GateTooDense",
                    FatalException, msg);
    }

    if (n == 0) {
        // Empty gate (e.g. dark current, P ≈ 1e-3). Geant4 cannot process an
        // event with no primary vertex, so fire a single geantino: it does not
        // interact or deposit energy, and the stepping action ignores it. The
        // gate still counts as an event (denominator of the gate rate) while
        // gElectronsFired stays untouched — no occupancy bias.
        fParticleGun->SetParticleDefinition(
            G4ParticleTable::GetParticleTable()->FindParticle("geantino"));
        fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., fBeamOriginZ));
        fParticleGun->SetParticleTime(0.);
        fParticleGun->SetParticleEnergy(1.*keV);
        fParticleGun->GeneratePrimaryVertex(anEvent);
        fParticleGun->SetParticleDefinition(
            G4ParticleTable::GetParticleTable()->FindParticle("e-"));
        fParticleGun->SetParticleEnergy(fCurrentEnergy);
        return;
    }

    DamsaConfig::gElectronsFired += n;
    for (std::size_t b = 0; b < occ.size(); ++b) {
        const G4double bunchTime = b * spec.bunchSpacing_s * second;
        for (int i = 0; i < occ[b]; ++i) {
            fParticleGun->SetParticlePosition(SampleBeamSpot());
            fParticleGun->SetParticleTime(bunchTime);
            fParticleGun->SetParticleEnergy(fCurrentEnergy);
            fParticleGun->GeneratePrimaryVertex(anEvent);
        }
    }
}

void DamsaPrimaryGenerator::SetBeamEnergy(G4double energy) {
    fCurrentEnergy = energy;
    fParticleGun->SetParticleEnergy(energy);
    G4cout << "Beam energy set to: " << energy/MeV << " MeV" << G4endl;
}

#endif
