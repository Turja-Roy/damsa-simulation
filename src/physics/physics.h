#ifndef PHYSICS_H
#define PHYSICS_H

// #include "G4VModularPhysicsList.hh"
// #include "G4EmStandardPhysics_option4.hh"
#include "G4EmExtraPhysics.hh"
// #include "G4DecayPhysics.hh"
#include "G4HadronElasticPhysics.hh"
// #include "G4HadronPhysicsFTFP_BERT.hh"
#include "G4EmParameters.hh"
#include "G4IonPhysics.hh"


#include "FTFP_BERT.hh"
// #include "QBBC.hh"
class DamsaPhysicsList : public FTFP_BERT {
// class DamsaPhysicsList : public QBBC {
public:
    DamsaPhysicsList() : FTFP_BERT() {
        G4EmExtraPhysics* emExtra = new G4EmExtraPhysics();
        emExtra->GammaNuclear(true);       // gamma + N -> pi0 + X
        emExtra->ElectroNuclear(true);     // e- + N -> e- + pi0 + X
        emExtra->MuonNuclear(true);        // minor contribution, good to have
        // FTFP_BERT already registers a G4EmExtraPhysics; RegisterPhysics()
        // rejects a second constructor of the same physics type, silently
        // dropping the flags above. ReplacePhysics swaps the existing one.
        ReplacePhysics(emExtra);
    }
    // DamsaPhysicsList() : QBBC() {}
    virtual ~DamsaPhysicsList() {}
};

// class DamsaPhysicsList : public G4VModularPhysicsList {
// public:
//     DamsaPhysicsList();
//     virtual ~DamsaPhysicsList();
// };
//
// DamsaPhysicsList::DamsaPhysicsList() {
//     // RegisterPhysics(new G4EmStandardPhysics_option4());
//     // RegisterPhysics(new G4EmExtraPhysics());       // photo-nuclear & electro-nuclear
//     // RegisterPhysics(new G4DecayPhysics());
//     // RegisterPhysics(new G4HadronElasticPhysics()); // neutron/hadron elastic scattering
//     // RegisterPhysics(new G4HadronPhysicsFTFP_BERT());
//     G4VUserPhysicsList::RegisterPhysics(new FTFP_BERT());
// }
//
// DamsaPhysicsList::~DamsaPhysicsList() {}

#endif
