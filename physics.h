#ifndef PHYSICS_H
#define PHYSICS_H

#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4DecayPhysics.hh"
#include "G4HadronPhysicsFTFP_BERT.hh"

class DamsaPhysicsList : public G4VModularPhysicsList {
public:
    DamsaPhysicsList();
    virtual ~DamsaPhysicsList();
};

DamsaPhysicsList::DamsaPhysicsList() {
    RegisterPhysics (new G4EmStandardPhysics_option4());
    RegisterPhysics (new G4DecayPhysics());
    RegisterPhysics (new G4HadronPhysicsFTFP_BERT());
}

DamsaPhysicsList::~DamsaPhysicsList() {}

#endif
