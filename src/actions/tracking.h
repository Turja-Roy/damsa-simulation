#ifndef TRACKING_H
#define TRACKING_H

#include "G4UserTrackingAction.hh"
#include "G4Track.hh"
#include "G4RunManager.hh"
#include "pi0DecayData.h"

class DamsaTrackingAction : public G4UserTrackingAction {
public:
    DamsaTrackingAction() {}
    virtual ~DamsaTrackingAction() {}
    virtual void PreUserTrackingAction(const G4Track* track);
};

void DamsaTrackingAction::PreUserTrackingAction(const G4Track* track)
{
    G4int trackID  = track->GetTrackID();
    G4int parentID = track->GetParentID();
    G4String name  = track->GetDefinition()->GetParticleName();

    DamsaPi0TrackingState* state = DamsaPi0TrackingState::Instance();

    if (name == "pi0") {
        // Register pi0 before any step fires — more reliable than step-1 check.
        state->pi0TrackIDs.insert(trackID);
        DamsaPi0Collector::Instance()->IncrementPi0Produced();
        state->trackToPi0Ancestor[trackID] = trackID;
    } else {
        // Propagate pi0 ancestry to ALL descendants (gammas, e+/e-, hadrons, etc.)
        // so that calo energy from pair-converted pi0 gammas is correctly attributed.
        auto it = state->trackToPi0Ancestor.find(parentID);
        if (it != state->trackToPi0Ancestor.end()) {
            state->trackToPi0Ancestor[trackID] = it->second;
        }
    }
}

#endif
