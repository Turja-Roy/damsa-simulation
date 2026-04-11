#ifndef CONSTRUCTION_MESSENGER_H
#define CONSTRUCTION_MESSENGER_H

// Geant4 UI messenger for DamsaDetectorConstruction.
//
// Provides macro commands:
//   /damsa/setVDCLength    <value> <unit>   — vacuum decay chamber length
//   /damsa/setTargetLength <value> <unit>   — tungsten target length along beam
//   /damsa/setCaloSizeXY   <value> <unit>   — calorimeter transverse (square) size
//
// All commands must be issued BEFORE /run/initialize so that Construct()
// picks up the new values.
//
// Example usage in a macro:
//   /damsa/setTargetLength 15 cm
//   /damsa/setVDCLength 35 cm
//   /damsa/setCaloSizeXY 16 cm
//   /run/initialize
//   /run/beamOn 100000

#include "G4UImessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "damsa_config.h"

// Forward declaration avoids circular include with construction.h.
// construction.cpp includes both headers so the full type is available there.
class DamsaDetectorConstruction;

class DamsaDetectorMessenger : public G4UImessenger
{
public:
    explicit DamsaDetectorMessenger(DamsaDetectorConstruction* det);
    ~DamsaDetectorMessenger() override;

    void SetNewValue(G4UIcommand* cmd, G4String val) override;

private:
    DamsaDetectorConstruction* fDetector;
    G4UIdirectory*             fDetDir;
    G4UIcmdWithADoubleAndUnit* fSetVDCLengthCmd;
    G4UIcmdWithADoubleAndUnit* fSetTargetLengthCmd;
    G4UIcmdWithADoubleAndUnit* fSetCaloSizeXYCmd;
    G4UIcmdWithAString*        fSetOutputPrefixCmd;
};

// ── Inline implementation ─────────────────────────────────────────────────────
// Included after the class declaration so that construction.h (which includes
// this file from construction.cpp) sees the full DamsaDetectorConstruction type
// before these methods use it.

#include "construction.h"

inline DamsaDetectorMessenger::DamsaDetectorMessenger(DamsaDetectorConstruction* det)
: fDetector(det), fDetDir(nullptr),
  fSetVDCLengthCmd(nullptr), fSetTargetLengthCmd(nullptr), fSetCaloSizeXYCmd(nullptr),
  fSetOutputPrefixCmd(nullptr)
{
    fDetDir = new G4UIdirectory("/damsa/");
    fDetDir->SetGuidance("DAMSA detector control commands.");

    fSetVDCLengthCmd = new G4UIcmdWithADoubleAndUnit("/damsa/setVDCLength", this);
    fSetVDCLengthCmd->SetGuidance("Set the vacuum decay chamber (VDC) length.");
    fSetVDCLengthCmd->SetGuidance("Must be called BEFORE /run/initialize.");
    fSetVDCLengthCmd->SetParameterName("Length", false);
    fSetVDCLengthCmd->SetRange("Length>0.");
    fSetVDCLengthCmd->SetDefaultUnit("cm");
    fSetVDCLengthCmd->SetUnitCandidates("mm cm m");
    fSetVDCLengthCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fSetTargetLengthCmd = new G4UIcmdWithADoubleAndUnit("/damsa/setTargetLength", this);
    fSetTargetLengthCmd->SetGuidance("Set the tungsten target length along the beam axis.");
    fSetTargetLengthCmd->SetGuidance("Must be called BEFORE /run/initialize.");
    fSetTargetLengthCmd->SetParameterName("Length", false);
    fSetTargetLengthCmd->SetRange("Length>0.");
    fSetTargetLengthCmd->SetDefaultUnit("cm");
    fSetTargetLengthCmd->SetUnitCandidates("mm cm m");
    fSetTargetLengthCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fSetCaloSizeXYCmd = new G4UIcmdWithADoubleAndUnit("/damsa/setCaloSizeXY", this);
    fSetCaloSizeXYCmd->SetGuidance("Set the calorimeter transverse (square) size.");
    fSetCaloSizeXYCmd->SetGuidance("Must be called BEFORE /run/initialize.");
    fSetCaloSizeXYCmd->SetParameterName("Size", false);
    fSetCaloSizeXYCmd->SetRange("Size>0.");
    fSetCaloSizeXYCmd->SetDefaultUnit("cm");
    fSetCaloSizeXYCmd->SetUnitCandidates("mm cm m");
    fSetCaloSizeXYCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fSetOutputPrefixCmd = new G4UIcmdWithAString("/damsa/setOutputPrefix", this);
    fSetOutputPrefixCmd->SetGuidance("Set output file prefix (path + basename prefix).");
    fSetOutputPrefixCmd->SetGuidance("Example: /damsa/setOutputPrefix output/Tz14/");
    fSetOutputPrefixCmd->SetParameterName("Prefix", false);
    fSetOutputPrefixCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
}

inline DamsaDetectorMessenger::~DamsaDetectorMessenger()
{
    delete fSetVDCLengthCmd;
    delete fSetTargetLengthCmd;
    delete fSetCaloSizeXYCmd;
    delete fSetOutputPrefixCmd;
    delete fDetDir;
}

inline void DamsaDetectorMessenger::SetNewValue(G4UIcommand* cmd, G4String val)
{
    if (cmd == fSetVDCLengthCmd) {
        fDetector->SetVDCLength(fSetVDCLengthCmd->GetNewDoubleValue(val));
    } else if (cmd == fSetTargetLengthCmd) {
        fDetector->SetTargetLength(fSetTargetLengthCmd->GetNewDoubleValue(val));
    } else if (cmd == fSetCaloSizeXYCmd) {
        fDetector->SetCaloSizeXY(fSetCaloSizeXYCmd->GetNewDoubleValue(val));
    } else if (cmd == fSetOutputPrefixCmd) {
        DamsaConfig::gOutputPrefix = std::string(val);
        G4cout << "[Config] Output prefix set to: " << val << G4endl;
    }
}

#endif  // CONSTRUCTION_MESSENGER_H
