#ifndef STEPPING_H
#define STEPPING_H

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepStatus.hh"
#include "G4RunManager.hh"
#include "G4VProcess.hh"
#include "analysis.h"
#include "FluxData.h"

class DamsaSteppingAction : public G4UserSteppingAction
{
public:
    DamsaSteppingAction();
    virtual ~DamsaSteppingAction();
    
    virtual void UserSteppingAction(const G4Step* step);
};

DamsaSteppingAction::DamsaSteppingAction()
: G4UserSteppingAction()
{}

DamsaSteppingAction::~DamsaSteppingAction()
{}

void DamsaSteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    G4String particleName = track->GetDefinition()->GetParticleName();
    G4int trackID = track->GetTrackID();
    G4int eventID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    // ─── Bremsstrahlung photon scoring INSIDE the target ────────────────────
    // MUST be before the fGeomBoundary early-return below.
    // Score every photon at its creation point (step 1) inside physTungsten,
    // regardless of whether that step crosses a geometry boundary.
    // These hard photons (up to 8 GeV) are the correct alplib Primakoff input.
    if(particleName == "gamma" && track->GetCurrentStepNumber() == 1) {
        G4VPhysicalVolume* birthVolume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
        if(birthVolume && birthVolume->GetName() == "physTungsten") {
            G4double brems_energy = track->GetKineticEnergy();
            if(brems_energy > 1.0*MeV) {
                const G4VProcess* creatorProcess = track->GetCreatorProcess();
                G4String processName = creatorProcess ? creatorProcess->GetProcessName() : "primary";
                if(processName == "eBrem" || processName == "annihil" || processName == "conv") {
                    G4ThreeVector bpos = step->GetPreStepPoint()->GetPosition();
                    G4ThreeVector bmom = track->GetMomentumDirection();
                    G4double btime    = step->GetPreStepPoint()->GetGlobalTime();
                    DamsaFluxCollector::Instance()->RecordBremsPhoton(
                        brems_energy, btime,
                        bpos.x(), bpos.y(), bpos.z(),
                        bmom.x(), bmom.y(), bmom.z(),
                        trackID, eventID);
                }
            }
        }
    }

    // ─── Boundary crossing checks (scoring planes) ──────────────────────────
    G4StepPoint* postStepPoint = step->GetPostStepPoint();
    if(!postStepPoint) return;

    G4VPhysicalVolume* volume = postStepPoint->GetTouchableHandle()->GetVolume();
    if(!volume) return;

    G4StepStatus stepStatus = postStepPoint->GetStepStatus();

    // Only record particles when they cross a geometry boundary
    if(stepStatus != fGeomBoundary) return;

    // Get particle properties
    G4double energy = track->GetKineticEnergy();
    G4bool isPrimary = (track->GetParentID() == 0);
    G4int pdgCode = track->GetDefinition()->GetPDGEncoding();

    G4ThreeVector momentum = track->GetMomentumDirection();
    G4double cosTheta = momentum.z();
    if(cosTheta < 0) return;

    G4double angle = momentum.angle(G4ThreeVector(0, 0, 1));
    G4ThreeVector position = postStepPoint->GetPosition();
    G4double time = postStepPoint->GetGlobalTime();

    G4String volumeName = volume->GetName();

    // Target exit scoring plane - records particles exiting the target
    if(volumeName == "physScoringVolumeTarget") {
        if(!DamsaAnalysis::Instance()->WasTrackRecorded(trackID, "TargetExit")) {
            DamsaAnalysis::Instance()->RecordParticle(particleName, energy, "TargetExit", angle, trackID, isPrimary);
            if(particleName == "gamma") {
                DamsaFluxCollector::Instance()->RecordPhoton(
                    energy, time,
                    position.x(), position.y(), position.z(),
                    momentum.x(), momentum.y(), momentum.z(),
                    trackID, eventID);
            }
            DamsaFluxCollector::Instance()->RecordParticle(
                pdgCode, energy, time,
                position.x(), position.y(), position.z(),
                momentum.x(), momentum.y(), momentum.z(),
                trackID, eventID);
        }
    }
    // Mid-target scoring plane
    else if(volumeName == "physScoringTargetMid") {
        if(!DamsaAnalysis::Instance()->WasTrackRecorded(trackID, "TargetMid")) {
            DamsaAnalysis::Instance()->RecordParticle(particleName, energy, "TargetMid", angle, trackID, isPrimary);
            
            // Also record photons at mid-target for alplib
            if(particleName == "gamma") {
                DamsaFluxCollector::Instance()->RecordPhoton(
                    energy, time,
                    position.x(), position.y(), position.z(),
                    momentum.x(), momentum.y(), momentum.z(),
                    trackID, eventID);
            }
        }
    }
    // Magnet entrance scoring plane
    else if(volumeName == "physScoringMagnetEntrance") {
        if(!DamsaAnalysis::Instance()->WasTrackRecorded(trackID, "MagnetEntrance")) {
            DamsaAnalysis::Instance()->RecordParticle(particleName, energy, "MagnetEntrance", angle, trackID, isPrimary);
        }
    }
    // Calorimeter entrance scoring plane - DETECTOR face for background scoring
    else if(volumeName == "physScoringCaloEntrance") {
        if(!DamsaAnalysis::Instance()->WasTrackRecorded(trackID, "CaloEntrance")) {
            DamsaAnalysis::Instance()->RecordParticle(particleName, energy, "CaloEntrance", angle, trackID, isPrimary);
        }
    }
    // Calorimeter exit scoring plane
    else if(volumeName == "physScoringCaloExit") {
        if(!DamsaAnalysis::Instance()->WasTrackRecorded(trackID, "CaloExit")) {
            DamsaAnalysis::Instance()->RecordParticle(particleName, energy, "CaloExit", angle, trackID, isPrimary);
        }
    }
}

#endif
