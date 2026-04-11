#ifndef DETECTOR_H
#define DETECTOR_H

#include "G4VSensitiveDetector.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"

class DamsaSensitiveDetector : public G4VSensitiveDetector {
public:
    DamsaSensitiveDetector(G4String);
    ~DamsaSensitiveDetector();

private:
    virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);
};

inline DamsaSensitiveDetector::DamsaSensitiveDetector(G4String name) : G4VSensitiveDetector(name) {}
inline DamsaSensitiveDetector::~DamsaSensitiveDetector() {}

inline G4bool DamsaSensitiveDetector::ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) {
    G4Track* track = aStep->GetTrack();

    track->SetTrackStatus(fStopAndKill);

    G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
    G4StepPoint* postStepPoint = aStep->GetPostStepPoint();

    G4ThreeVector posPhoton = preStepPoint->GetPosition();

    const G4VTouchable* touchable = aStep->GetPreStepPoint()->GetTouchable();
    G4int copyNo = touchable->GetCopyNumber(0);

    G4VPhysicalVolume* physVol = touchable->GetVolume();
    G4ThreeVector posDetector = physVol->GetTranslation();

    G4cout << "Detector hit at position: " << posDetector << G4endl;

    G4int evt = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleIColumn(0, evt);
    man->FillNtupleDColumn(1, posDetector[0]);
    man->FillNtupleDColumn(2, posDetector[1]);
    man->FillNtupleDColumn(3, posDetector[2]);
    man->AddNtupleRow(0);

    return true;
}

#endif
