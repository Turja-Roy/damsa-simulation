#ifndef ACTION_H
#define ACTION_H

#include "G4VUserActionInitialization.hh"

#include "generator.h"
#include "alp_generator.h"
#include "damsa_config.h"
#include "run.h"
#include "event.h"
#include "stepping.h"

class DamsaActionInitialization : public G4VUserActionInitialization {
public:
    DamsaActionInitialization();
    virtual ~DamsaActionInitialization();

    virtual void Build() const;
};

DamsaActionInitialization::DamsaActionInitialization () {}
DamsaActionInitialization::~DamsaActionInitialization () {}

void DamsaActionInitialization::Build () const {
    if (DamsaConfig::gRunMode == DamsaConfig::RunMode::ALPInject) {
        if (DamsaConfig::gALPDecayCSV.empty()) {
            G4Exception("DamsaActionInitialization::Build", "ALPInjectNoCSV",
                        FatalException,
                        "ALPInject mode requested but DamsaConfig::gALPDecayCSV is empty.");
        }
        SetUserAction(new DamsaALPDecayGenerator(DamsaConfig::gALPDecayCSV));
    } else {
        SetUserAction(new DamsaPrimaryGenerator());
    }

    DamsaRunAction* runAction = new DamsaRunAction();
    SetUserAction(runAction);

    DamsaEventAction* eventAction = new DamsaEventAction();
    SetUserAction(eventAction);
    
    DamsaSteppingAction* steppingAction = new DamsaSteppingAction();
    SetUserAction(steppingAction);
}

#endif
