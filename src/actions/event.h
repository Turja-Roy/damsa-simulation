#ifndef EVENT_H
#define EVENT_H

#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include "analysis.h"

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
}

void DamsaEventAction::EndOfEventAction(const G4Event*)
{
    // Particle spectra analysis is handled by DamsaAnalysis via stepping action
    // No additional processing needed at end of event
}

#endif
