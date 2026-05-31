#ifndef EVENT_H
#define EVENT_H

#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include "analysis.h"
#include "pi0DecayData.h"

class DamsaEventAction : public G4UserEventAction
{
public:
    DamsaEventAction() {}
    virtual ~DamsaEventAction() {}
    
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);
};

void DamsaEventAction::BeginOfEventAction(const G4Event*)
{
    // Reset per-event track ID sets to prevent cross-event contamination
    DamsaAnalysis::Instance()->ResetEventTracking();
    // Reset per-thread pi0 tracking state
    DamsaPi0TrackingState::Instance()->Reset();
}

void DamsaEventAction::EndOfEventAction(const G4Event* event)
{
    // Transfer per-thread pi0 calo energy accumulation to global collector.
    // Must happen after all tracking for this event is complete (ordering guarantee).
    G4int evtID = event->GetEventID();
    DamsaPi0TrackingState* state = DamsaPi0TrackingState::Instance();
    for (const auto& kv : state->pi0CaloEnergy_MeV) {
        DamsaPi0Collector::Instance()->AddCaloEnergy(evtID, kv.first, kv.second);
    }
}

#endif
