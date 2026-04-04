#ifndef CONSTRUCTION_H
#define CONSTRUCTION_H

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4SubtractionSolid.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4PSNofSecondary.hh"
#include "G4MagneticField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4ChordFinder.hh"

class DamsaDetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DamsaDetectorConstruction();
    virtual ~DamsaDetectorConstruction();

    virtual G4VPhysicalVolume* Construct() override;

    void SetGapDistance(G4double gap);
    void SetTargetLength(G4double length);
    void SetTargetTransverse(G4double size);

    G4double GetTargetExitZ() const { return fTargetExitZ; }
    G4double GetCaloEntranceZ() const { return fCaloEntranceZ; }

private:
    virtual void ConstructSDandField() override;
    
    void DefineMaterials();
    void BuildTarget(G4LogicalVolume* worldLV, G4double& zPos);
    void BuildVacuumChamber(G4LogicalVolume* worldLV, G4double& zPos);
    void BuildMagnetAndTrackerRegion(G4LogicalVolume* worldLV, G4double& zPos);
    void BuildCalorimeter(G4LogicalVolume* worldLV, G4double& zPos);

    G4double fWorldSize;
    G4double fTargetX, fTargetY, fTargetZ;
    G4double fTargetExitZ;
    G4double fGapDistance;
    G4double fCaloEntranceZ;
    G4double fChamberInnerRadius, fChamberWallThickness, fChamberLength;
    G4double fMagnetOuterSizeXY, fMagnetOuterSizeZ, fMagnetHollowSizeXY, fMagnetHollowSizeZ;
    G4double fTrackerSizeXY, fTrackerThickness;
    G4int fNumTrackers;
    G4double fCaloSizeXY, fLayerThickness;
    G4int fNumCaloLayers, fNumCrystalsPerLayer;

    G4LogicalVolume* fLogicSiTracker;
    G4LogicalVolume* fLogicCrystal;
    G4LogicalVolume* fLogicMagnetHollow;
    G4LogicalVolume* fLogicScoringMagnetEntrance;
    G4LogicalVolume* fLogicScoringCaloEntrance;
    G4LogicalVolume* fLogicScoringCaloExit;
    G4LogicalVolume* fLogicScoringTargetMid;
    G4MagneticField* fMagField;
    
    G4Material* fMatAir;
    G4Material* fMatVacuum;
    G4Material* fMatTungsten;
    G4Material* fMatStainlessSteel;
    G4Material* fMatSilicon;
    G4Material* fMatCsI;
    G4Material* fMatNeodymium;
};

#endif

