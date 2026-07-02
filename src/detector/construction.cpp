#include "construction_messenger.h"
#include "construction.h"
#include "pi0DecayData.h"

#include <algorithm>

#include "G4Material.hh"
#include "G4Element.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4RotationMatrix.hh"
#include "MagneticField.h"

DamsaDetectorConstruction::DamsaDetectorConstruction()
    : fMessenger(nullptr),
    fLogicSiTracker(nullptr), fLogicCrystal(nullptr), fLogicECAL(nullptr), fLogicMagnetHollow(nullptr), fMagField(nullptr),
    fMatAir(nullptr), fMatVacuum(nullptr), fMatTungsten(nullptr), fMatStainlessSteel(nullptr),
    fMatSilicon(nullptr), fMatCsI(nullptr), fMatNeodymium(nullptr)
{
    fWorldSize = 0.4*m;

    fTargetX = 5.0*cm;
    fTargetY = 5.0*cm;
    fTargetZ = 10.0*cm;
    fTargetExitZ = 0.0*cm;    // Computed dynamically in BuildTarget; do not hardcode
    fVDCLength = 30.0*cm;     // Default VDC length; scan range is 30–60 cm.
    fCaloEntranceZ = 0.0*cm;  // Calculated in BuildCalorimeter()

    fMessenger = new DamsaDetectorMessenger(this);

    fChamberInnerRadius = 10.0*cm;
    fChamberWallThickness = 0.5*cm;

    fMagnetOuterSizeXY = 20.0*cm;
    fMagnetOuterSizeZ = 12.0*cm;
    fMagnetHollowSizeXY = 10.0*cm;
    fMagnetHollowSizeZ = fMagnetOuterSizeZ;

    fTrackerSizeXY = 9.8*cm;
    fTrackerThickness = 0.2*cm;
    fNumTrackers = 6;

    fCaloSizeXY = 12.0*cm;
    fLayerThickness = 1.0*cm;
    fNumCaloLayers = 44;  // Alternating X/Y orientation
    fNumCrystalsPerLayer = 12;
}

DamsaDetectorConstruction::~DamsaDetectorConstruction()
{
    delete fMessenger;
}

void DamsaDetectorConstruction::SetVDCLength(G4double length)
{
    fVDCLength = length;
    G4cout << "[Construction] VDC length set to: " << length/cm << " cm" << G4endl;
}

void DamsaDetectorConstruction::SetTargetLength(G4double length)
{
    fTargetZ = length;
    G4cout << "[Construction] Target length set to: " << length/cm << " cm" << G4endl;
}

void DamsaDetectorConstruction::SetTargetTransverse(G4double size)
{
    fTargetX = size;
    fTargetY = size;
    G4cout << "[Construction] Target transverse size set to: " << size/cm << " cm" << G4endl;
}

void DamsaDetectorConstruction::SetCaloSizeXY(G4double xy)
{
    fCaloSizeXY = xy;
    G4cout << "[Construction] Calo XY size set to: " << xy/cm << " cm" << G4endl;
}

G4VPhysicalVolume* DamsaDetectorConstruction::Construct()
{
    DefineMaterials();

    // World must contain target, VDC, magnet, and ECAL — always use 2 m half-length.
    G4double worldSizeZ = 2.0*m;
    // Transverse half-size follows the widest runtime-tunable element (the calo
    // via /damsa/setCaloSizeXY) so large scans cannot poke through the world.
    G4double worldHalfXY = std::max(0.2*m, fCaloSizeXY/2. + 5.*cm);
    auto* solidWorld = new G4Box("solidWorld", worldHalfXY, worldHalfXY, worldSizeZ);
    auto* logicWorld = new G4LogicalVolume(solidWorld, fMatAir, "logicWorld");
    auto* physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, true);

    G4double zPos = -50.*cm;

    BuildTarget(logicWorld, zPos);
    BuildVacuumChamber(logicWorld, zPos);
    BuildMagnetAndTrackerRegion(logicWorld, zPos);
    BuildCalorimeter(logicWorld, zPos);

    // Publish calo geometry to pi0 collector for geometric acceptance computation.
    // Called before any events run — no mutex needed.
    DamsaPi0Collector::Instance()->SetGeometry(fCaloEntranceZ, fCaloSizeXY / 2.0);

    return physWorld;
}

void DamsaDetectorConstruction::DefineMaterials()
{
    auto* nist = G4NistManager::Instance();

    fMatAir = nist->FindOrBuildMaterial("G4_AIR");
    fMatVacuum = nist->FindOrBuildMaterial("G4_Galactic");
    fMatTungsten = nist->FindOrBuildMaterial("G4_W");
    fMatStainlessSteel = nist->FindOrBuildMaterial("G4_STAINLESS-STEEL");
    fMatSilicon = nist->FindOrBuildMaterial("G4_Si");
    fMatCsI = nist->FindOrBuildMaterial("G4_CESIUM_IODIDE");

    // Neodymium
    fMatNeodymium = new G4Material("Neodymium", 7.01*g/cm3, 1);
    fMatNeodymium->AddElement(nist->FindOrBuildElement("Nd"), 1);
}

void DamsaDetectorConstruction::BuildTarget(G4LogicalVolume* worldLV, G4double& zPos)
{
    auto* solidTungsten = new G4Box("solidTungsten", fTargetX/2.0, fTargetY/2.0, fTargetZ/2.0);
    auto* logicTungsten = new G4LogicalVolume(solidTungsten, fMatTungsten, "logicTungsten");
    new G4PVPlacement(0, G4ThreeVector(0., 0., zPos+fTargetZ/2.0), logicTungsten, "physTungsten", worldLV, false, 0, true);

    auto* tungstenVis = new G4VisAttributes(G4Colour(0.3, 0.3, 0.3, 1.0));
    tungstenVis->SetForceSolid(true);
    logicTungsten->SetVisAttributes(tungstenVis);

    zPos += fTargetZ;
    fTargetExitZ = zPos;  // Target exit = start-of-build (-50 cm) + fTargetZ
    G4cout << "[Construction] Target length: " << fTargetZ/cm << " cm"
           << "  exit Z: " << fTargetExitZ/cm << " cm" << G4endl;
}

void DamsaDetectorConstruction::BuildVacuumChamber(G4LogicalVolume* worldLV, G4double& zPos)
{
    G4double chamberOuterRadius = fChamberInnerRadius + fChamberWallThickness;

    // Chamber wall (cylindrical tube)
    auto* solidChamberOuter = new G4Tubs("solidChamberOuter", 0., chamberOuterRadius, fVDCLength/2., 0., 360.*deg);
    auto* solidChamberInner = new G4Tubs("solidChamberInner", 0., fChamberInnerRadius, fVDCLength/2., 0., 360.*deg);

    auto* solidChamberWall = new G4SubtractionSolid("solidChamberWall", solidChamberOuter, solidChamberInner);
    auto* logicChamberWall = new G4LogicalVolume(solidChamberWall, fMatStainlessSteel, "logicChamberWall");

    auto* chamberWallVis = new G4VisAttributes(G4Colour(1.0, 1.0, 0.0, 0.7));
    chamberWallVis->SetForceSolid(true);
    logicChamberWall->SetVisAttributes(chamberWallVis);

    // End caps: solid discs placed INSIDE the chamber (as part of the vacuum volume geometry)
    // They seal the chamber at both ends. End cap thickness reduces the vacuum region.
    auto* solidEndCap = new G4Tubs("solidEndCap", 0., fChamberInnerRadius, fChamberWallThickness/2., 0., 360.*deg);
    
    auto* logicEndCapFront = new G4LogicalVolume(solidEndCap, fMatStainlessSteel, "logicEndCapFront");
    logicEndCapFront->SetVisAttributes(chamberWallVis);

    auto* logicEndCapBack = new G4LogicalVolume(solidEndCap, fMatStainlessSteel, "logicEndCapBack");
    logicEndCapBack->SetVisAttributes(chamberWallVis);

    // Vacuum region is the full chamber inner volume (end caps will be placed as daughters)
    auto* logicChamberVacuum = new G4LogicalVolume(solidChamberInner, fMatVacuum, "logicChamberVacuum");
    logicChamberVacuum->SetVisAttributes(G4VisAttributes::GetInvisible());

    // Place end caps inside the vacuum volume at the front and back
    G4double endCapLocalFrontZ = -fVDCLength/2. + fChamberWallThickness/2.;
    G4double endCapLocalBackZ = fVDCLength/2. - fChamberWallThickness/2.;
    new G4PVPlacement(0, G4ThreeVector(0., 0., endCapLocalFrontZ), logicEndCapFront, "physEndCapFront", logicChamberVacuum, false, 0, true);
    new G4PVPlacement(0, G4ThreeVector(0., 0., endCapLocalBackZ), logicEndCapBack, "physEndCapBack", logicChamberVacuum, false, 1, true);

    // Target exit scoring volume (placed inside vacuum chamber, after front end cap)
    // Target rear face is at zPos (current value), scoring plane just after front end cap.
    // Offset by the plane's own half-thickness (0.1 mm) so it does not overlap
    // the end cap (its front face is flush with the cap's back face).
    G4double scoringZ_local = endCapLocalFrontZ + fChamberWallThickness/2. + 0.1*mm;
    
    // Circular scoring plane matching chamber inner radius to capture all particles entering decay chamber
    auto* solidScoringTarget = new G4Tubs("solidScoringTarget", 0., fChamberInnerRadius, 0.1*mm, 0., 360.*deg);
    auto* logicScoringTarget = new G4LogicalVolume(solidScoringTarget, 
                                                   fMatVacuum, 
                                                   "logicScoringTarget");
    auto* scoringVis = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0, 1.0));
    scoringVis->SetForceSolid(true);
    logicScoringTarget->SetVisAttributes(scoringVis);
    // logicScoringTarget->SetVisAttributes(G4VisAttributes::GetInvisible());
    
    // Place scoring plane inside vacuum chamber volume
    new G4PVPlacement(0, G4ThreeVector(0., 0., scoringZ_local), 
                      logicScoringTarget, "physScoringVolumeTarget", logicChamberVacuum, false, 0, true);

    zPos += fVDCLength/2.;
    new G4PVPlacement(0, G4ThreeVector(0., 0., zPos), logicChamberWall, "physChamberWall", worldLV, false, 0, true);
    new G4PVPlacement(0, G4ThreeVector(0., 0., zPos), logicChamberVacuum, "physChamberVacuum", worldLV, false, 0, true);

    zPos += fVDCLength/2.;
}

void DamsaDetectorConstruction::BuildMagnetAndTrackerRegion(G4LogicalVolume* worldLV, G4double& zPos)
{
    // Air-filled region placed directly in world to avoid subtraction solid navigation issues.
    // Neodymium magnet frame disabled (B=0).

    auto* solidMagnetHollow = new G4Box("solidMagnetHollow", fMagnetHollowSizeXY/2., fMagnetHollowSizeXY/2., fMagnetHollowSizeZ/2.);
    fLogicMagnetHollow = new G4LogicalVolume(solidMagnetHollow, fMatAir, "logicMagnetHollow");
    fLogicMagnetHollow->SetVisAttributes(G4VisAttributes::GetInvisible());

    zPos += fMagnetOuterSizeZ/2.;
    new G4PVPlacement(0, G4ThreeVector(0., 0., zPos), fLogicMagnetHollow, "physMagnetHollow", worldLV, false, 0, true);

    // Neodymium magnet (disabled)
    // auto* solidMagnetOuter = new G4Box("solidMagnetOuter", fMagnetOuterSizeXY/2., fMagnetOuterSizeXY/2., fMagnetOuterSizeZ/2.);
    // auto* solidMagnet = new G4SubtractionSolid("solidMagnet", solidMagnetOuter, solidMagnetHollow);
    // auto* logicMagnet = new G4LogicalVolume(solidMagnet, fMatNeodymium, "logicMagnet");
    // auto* magnetVis = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.3));
    // magnetVis->SetForceSolid(true);
    // logicMagnet->SetVisAttributes(magnetVis);
    // new G4PVPlacement(0, G4ThreeVector(0., 0., zPos), logicMagnet, "physMagnet", worldLV, false, 0, true);

    // Magnet entrance scoring volume
    G4double hollowHalfZ = fMagnetOuterSizeZ/2.;

    G4double scoringHalfThickness = 0.05*mm;  // Match reference implementation (0.1mm total thickness)
    auto* solidScoringMagnetEntrance = new G4Box("solidScoringMagnetEntrance", fTrackerSizeXY/2., fTrackerSizeXY/2., scoringHalfThickness);
    fLogicScoringMagnetEntrance = new G4LogicalVolume(solidScoringMagnetEntrance,
                                                      fMatVacuum,
                                                      "logicScoringMagnetEntrance");
    auto* scoringVis = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0, 1.0));
    scoringVis->SetForceSolid(true);
    fLogicScoringMagnetEntrance->SetVisAttributes(scoringVis);
    // fLogicScoringMagnetEntrance->SetVisAttributes(G4VisAttributes::GetInvisible());

    G4double scoringZ = -hollowHalfZ + scoringHalfThickness + 0.1*mm;  // 0.1mm clearance
    new G4PVPlacement(0, G4ThreeVector(0., 0., scoringZ),
                      fLogicScoringMagnetEntrance, "physScoringMagnetEntrance", 
                      fLogicMagnetHollow, false, 0, true);

    // With N trackers, there are (N+1) equal gaps
    G4double totalLength = 2.*hollowHalfZ;
    G4double totalTrackerLength = fNumTrackers * fTrackerThickness;
    G4double numGaps = fNumTrackers + 1;
    G4double gapSize = (totalLength - totalTrackerLength) / numGaps;

    G4double trackerFitSize = fTrackerSizeXY - 0.2*mm;
    auto* solidSiTracker = new G4Box("solidSiTracker", trackerFitSize/2., trackerFitSize/2., fTrackerThickness/2.);
    fLogicSiTracker = new G4LogicalVolume(solidSiTracker, fMatSilicon, "logicSiTracker");

    auto* trackerVis = new G4VisAttributes(G4Colour(0.0, 1.0, 1.0, 0.8));
    trackerVis->SetForceSolid(true);
    fLogicSiTracker->SetVisAttributes(trackerVis);

    for (G4int i = 0; i < fNumTrackers; i++) {
        // Position: -halfZ + (i+1)*gap + i*thickness + thickness/2
        G4double localZ = -hollowHalfZ + (i + 1) * gapSize + i * fTrackerThickness + fTrackerThickness/2.;
        new G4PVPlacement(0, G4ThreeVector(0., 0., localZ), fLogicSiTracker, "physSiTracker", fLogicMagnetHollow, false, i, true);
    }

    zPos += fMagnetOuterSizeZ/2.;
}

void DamsaDetectorConstruction::BuildCalorimeter(G4LogicalVolume* worldLV, G4double& zPos)
{
    G4double ecalDepth = fNumCaloLayers * fLayerThickness;
    G4double scoringHalfThickness = 0.05*mm;

    // Calo entrance = target exit + VDC + magnet.
    // The VDC (vacuum chamber) physically resizes with fVDCLength; the magnet
    // follows at the VDC exit, and the calo is placed immediately after the magnet.
    fCaloEntranceZ = fTargetExitZ + fVDCLength + fMagnetOuterSizeZ;

    G4cout << "\n=== CALORIMETER POSITION ===" << G4endl;
    G4cout << "Target exit Z:   " << fTargetExitZ/cm       << " cm" << G4endl;
    G4cout << "VDC length:      " << fVDCLength/cm         << " cm" << G4endl;
    G4cout << "Magnet length:   " << fMagnetOuterSizeZ/cm  << " cm" << G4endl;
    G4cout << "Calo entrance Z: " << fCaloEntranceZ/cm     << " cm" << G4endl;

    // Scoring plane at calorimeter entrance
    auto* solidScoringCaloEntrance = new G4Box("solidScoringCaloEntrance",
                                               fCaloSizeXY/2., fCaloSizeXY/2., scoringHalfThickness);
    fLogicScoringCaloEntrance = new G4LogicalVolume(solidScoringCaloEntrance,
                                                    fMatVacuum,
                                                    "logicScoringCaloEntrance");
    auto* scoringEntranceVis = new G4VisAttributes(G4Colour(0.0, 1.0, 0.0, 1.0));
    scoringEntranceVis->SetForceSolid(true);
    fLogicScoringCaloEntrance->SetVisAttributes(scoringEntranceVis);

    // Centre the plane half a thickness downstream of fCaloEntranceZ: the magnet
    // hollow ends exactly at fCaloEntranceZ, so centring the plane there would
    // overlap the hollow by scoringHalfThickness.
    new G4PVPlacement(0, G4ThreeVector(0., 0., fCaloEntranceZ + scoringHalfThickness),
                      fLogicScoringCaloEntrance, "physScoringCaloEntrance",
                      worldLV, false, 0, true);

    G4double ecalFrontZ  = fCaloEntranceZ + 2.*scoringHalfThickness;
    G4double ecalCenterZ = ecalFrontZ + ecalDepth / 2.;

    G4cout << "ECAL front face absolute Z: " << ecalFrontZ/cm  << " cm" << G4endl;
    G4cout << "ECAL center absolute Z:     " << ecalCenterZ/cm << " cm" << G4endl;

    // ── Simplified monolithic CsI calorimeter (temporary, for geometry optimisation) ──
    // The detailed crystal bar geometry is preserved below (commented out) and should
    // be restored once the optimal geometry is determined.
    G4cout << "[Calorimeter] Using simplified monolithic CsI box ("
           << fCaloSizeXY/cm << " x " << fCaloSizeXY/cm
           << " x " << ecalDepth/cm << " cm)" << G4endl;

    auto* solidECAL = new G4Box("solidECAL", fCaloSizeXY/2., fCaloSizeXY/2., ecalDepth/2.);
    fLogicECAL = new G4LogicalVolume(solidECAL, fMatCsI, "logicECAL");

    auto* caloVis = new G4VisAttributes(G4Colour(1.0, 0.0, 1.0, 0.5));
    caloVis->SetForceSolid(true);
    fLogicECAL->SetVisAttributes(caloVis);

    new G4PVPlacement(0, G4ThreeVector(0., 0., ecalCenterZ), fLogicECAL, "physECAL",
                      worldLV, false, 0, true);

    // ── Original crystal bar geometry (commented out for geometry optimisation) ──
    // Restore when reverting to full simulation after optimisation is complete.
    //
    // auto* logicECAL_bars = new G4LogicalVolume(solidECAL, fMatAir, "logicECAL");
    // logicECAL_bars->SetVisAttributes(G4VisAttributes::GetInvisible());
    //
    // auto* solidCrystal = new G4Box("solidCrystal", 6.0*cm, 0.5*cm, 0.5*cm);
    // fLogicCrystal = new G4LogicalVolume(solidCrystal, fMatCsI, "logicCrystal");
    // auto* crystalVis = new G4VisAttributes(G4Colour(1.0, 0.0, 1.0, 0.5));
    // crystalVis->SetForceSolid(true);
    // fLogicCrystal->SetVisAttributes(crystalVis);
    //
    // for (G4int layer = 0; layer < fNumCaloLayers; layer++) {
    //     G4double localZ    = -ecalDepth/2. + fLayerThickness/2. + layer * fLayerThickness;
    //     G4bool isXOriented = (layer % 2 == 0);
    //     G4RotationMatrix* rot = nullptr;
    //     if (!isXOriented) { rot = new G4RotationMatrix(); rot->rotateZ(90.*deg); }
    //     for (G4int crystal = 0; crystal < fNumCrystalsPerLayer; crystal++) {
    //         G4double offset = -fCaloSizeXY/2. + 0.5*cm + crystal * 1.0*cm;
    //         G4ThreeVector position = isXOriented
    //             ? G4ThreeVector(0., offset, localZ)
    //             : G4ThreeVector(offset, 0., localZ);
    //         G4int copyNo = layer * fNumCrystalsPerLayer + crystal;
    //         new G4PVPlacement(rot, position, fLogicCrystal, "physCalorimeter",
    //                           logicECAL_bars, false, copyNo, true);
    //     }
    // }
    // new G4PVPlacement(0, G4ThreeVector(0., 0., ecalCenterZ), logicECAL_bars, "physECAL",
    //                   worldLV, false, 0, true);

    // Scoring plane at calorimeter exit
    auto* solidScoringCaloExit = new G4Box("solidScoringCaloExit",
                                           fCaloSizeXY/2., fCaloSizeXY/2., scoringHalfThickness);
    fLogicScoringCaloExit = new G4LogicalVolume(solidScoringCaloExit, fMatVacuum, "logicScoringCaloExit");
    auto* scoringExitVis = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 1.0));
    scoringExitVis->SetForceSolid(true);
    fLogicScoringCaloExit->SetVisAttributes(scoringExitVis);

    G4double ecalBackZ = ecalFrontZ + ecalDepth;
    G4double caloExitZ = ecalBackZ + 0.1*mm + scoringHalfThickness;
    new G4PVPlacement(0, G4ThreeVector(0., 0., caloExitZ),
                      fLogicScoringCaloExit, "physScoringCaloExit",
                      worldLV, false, 0, true);

    G4cout << "ECAL back face absolute Z:      " << ecalBackZ/cm
           << " cm" << G4endl;
    G4cout << "Calo exit scoring absolute Z:   " << caloExitZ/cm << " cm" << G4endl;
}

void DamsaDetectorConstruction::ConstructSDandField()
{
    auto* sdManager = G4SDManager::GetSDMpointer();

    auto* trackerSD = new G4MultiFunctionalDetector("TrackerSD");
    sdManager->AddNewDetector(trackerSD);

    auto* energyDep = new G4PSEnergyDeposit("EnergyDeposit");
    trackerSD->RegisterPrimitive(energyDep);

    auto* nofSecondary = new G4PSNofSecondary("NofSecondary");
    trackerSD->RegisterPrimitive(nofSecondary);

    fLogicSiTracker->SetSensitiveDetector(trackerSD);

    auto* calorimeterSD = new G4MultiFunctionalDetector("CalorimeterSD");
    sdManager->AddNewDetector(calorimeterSD);

    auto* caloEnergyDep = new G4PSEnergyDeposit("EnergyDeposit");
    calorimeterSD->RegisterPrimitive(caloEnergyDep);

    auto* caloNofSecondary = new G4PSNofSecondary("NofSecondary");
    calorimeterSD->RegisterPrimitive(caloNofSecondary);

    // Simplified geometry: register monolithic CsI box as sensitive detector.
    // When reverting to crystal bar geometry, replace fLogicECAL with fLogicCrystal.
    fLogicECAL->SetSensitiveDetector(calorimeterSD);
    // fLogicCrystal->SetSensitiveDetector(calorimeterSD);  // crystal bar geometry

    // Scoring volume sensitive detectors
    auto* scoringMagnetEntranceSD = new G4MultiFunctionalDetector("ScoringMagnetEntranceSD");
    sdManager->AddNewDetector(scoringMagnetEntranceSD);
    
    auto* scoringMagnetEntranceEnergyDep = new G4PSEnergyDeposit("EnergyDeposit");
    scoringMagnetEntranceSD->RegisterPrimitive(scoringMagnetEntranceEnergyDep);
    
    auto* scoringMagnetEntranceNofSecondary = new G4PSNofSecondary("NofSecondary");
    scoringMagnetEntranceSD->RegisterPrimitive(scoringMagnetEntranceNofSecondary);
    
    fLogicScoringMagnetEntrance->SetSensitiveDetector(scoringMagnetEntranceSD);

    auto* scoringCaloEntranceSD = new G4MultiFunctionalDetector("ScoringCaloEntranceSD");
    sdManager->AddNewDetector(scoringCaloEntranceSD);
    
    auto* scoringCaloEntranceEnergyDep = new G4PSEnergyDeposit("EnergyDeposit");
    scoringCaloEntranceSD->RegisterPrimitive(scoringCaloEntranceEnergyDep);
    
    auto* scoringCaloEntranceNofSecondary = new G4PSNofSecondary("NofSecondary");
    scoringCaloEntranceSD->RegisterPrimitive(scoringCaloEntranceNofSecondary);
    
    fLogicScoringCaloEntrance->SetSensitiveDetector(scoringCaloEntranceSD);

    auto* scoringCaloExitSD = new G4MultiFunctionalDetector("ScoringCaloExitSD");
    sdManager->AddNewDetector(scoringCaloExitSD);
    
    auto* scoringCaloExitEnergyDep = new G4PSEnergyDeposit("EnergyDeposit");
    scoringCaloExitSD->RegisterPrimitive(scoringCaloExitEnergyDep);
    
    auto* scoringCaloExitNofSecondary = new G4PSNofSecondary("NofSecondary");
    scoringCaloExitSD->RegisterPrimitive(scoringCaloExitNofSecondary);
    
    fLogicScoringCaloExit->SetSensitiveDetector(scoringCaloExitSD);

    // fMagField = new MagneticField();
    // auto* fieldMgr = new G4FieldManager();
    // fieldMgr->SetDetectorField(fMagField);
    // fieldMgr->CreateChordFinder(fMagField);
    // fLogicMagnetHollow->SetFieldManager(fieldMgr, true);
}
