#include "construction.h"

#include "G4Material.hh"
#include "G4Element.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4RotationMatrix.hh"
#include "MagneticField.h"

DamsaDetectorConstruction::DamsaDetectorConstruction()
    : fLogicSiTracker(nullptr), fLogicCrystal(nullptr), fLogicMagnetHollow(nullptr), fMagField(nullptr),
    fMatAir(nullptr), fMatVacuum(nullptr), fMatTungsten(nullptr), fMatStainlessSteel(nullptr),
    fMatSilicon(nullptr), fMatCsI(nullptr), fMatNeodymium(nullptr)
{
    fWorldSize = 0.4*m;

    fTargetX = 5.0*cm;
    fTargetY = 5.0*cm;
    fTargetZ = 10.0*cm;
    fTargetExitZ = -40.0*cm;  // Target rear face position
    fGapDistance = 0.0*cm;  // Default gap (no gap - calorimeter right after magnet)
    fCaloEntranceZ = 0.0*cm; // Will be calculated

    fChamberInnerRadius = 10.0*cm;
    fChamberWallThickness = 0.5*cm;
    fChamberLength = 30.0*cm;

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

DamsaDetectorConstruction::~DamsaDetectorConstruction(){}

void DamsaDetectorConstruction::SetGapDistance(G4double gap)
{
    fGapDistance = gap;
    G4cout << "Gap distance set to: " << gap/cm << " cm" << G4endl;
}

void DamsaDetectorConstruction::SetTargetLength(G4double length)
{
    fTargetZ = length;
    G4cout << "Target length set to: " << length/cm << " cm" << G4endl;
}

void DamsaDetectorConstruction::SetTargetTransverse(G4double size)
{
    fTargetX = size;
    fTargetY = size;
    G4cout << "Target transverse size set to: " << size/cm << " cm" << G4endl;
}

G4VPhysicalVolume* DamsaDetectorConstruction::Construct()
{
    DefineMaterials();

    // Increase world size for gap scan mode to accommodate larger distances
    G4double worldSizeZ = (fGapDistance > 0) ? 2.0*m : 0.6*m;
    auto* solidWorld = new G4Box("solidWorld", 0.2*m, 0.2*m, worldSizeZ);
    auto* logicWorld = new G4LogicalVolume(solidWorld, fMatAir, "logicWorld");
    auto* physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, true);

    G4double zPos = -50.*cm;

    BuildTarget(logicWorld, zPos);
    BuildVacuumChamber(logicWorld, zPos);
    BuildMagnetAndTrackerRegion(logicWorld, zPos);
    
    // Store target exit Z for gap calculations (target rear face)
    fTargetExitZ = -40.0*cm;  // Fixed target exit position
    BuildCalorimeter(logicWorld, zPos);

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
    
    // Mid-target scoring plane (at center of target)
    G4double scoringHalfThickness = 0.05*mm;
    auto* solidScoringTargetMid = new G4Box("solidScoringTargetMid", fTargetX/2., fTargetY/2., scoringHalfThickness);
    fLogicScoringTargetMid = new G4LogicalVolume(solidScoringTargetMid, fMatVacuum, "logicScoringTargetMid");
    
    auto* midScoringVis = new G4VisAttributes(G4Colour(1.0, 0.5, 0.5, 1.0));
    midScoringVis->SetForceSolid(true);
    fLogicScoringTargetMid->SetVisAttributes(midScoringVis);
    
    G4double midZ = zPos + fTargetZ/2.0;
    new G4PVPlacement(0, G4ThreeVector(0., 0., midZ), fLogicScoringTargetMid, "physScoringTargetMid", worldLV, false, 0, true);
    
    zPos += fTargetZ;
}

void DamsaDetectorConstruction::BuildVacuumChamber(G4LogicalVolume* worldLV, G4double& zPos)
{
    G4double chamberOuterRadius = fChamberInnerRadius + fChamberWallThickness;

    // Chamber wall (cylindrical tube)
    auto* solidChamberOuter = new G4Tubs("solidChamberOuter", 0., chamberOuterRadius, fChamberLength/2., 0., 360.*deg);
    auto* solidChamberInner = new G4Tubs("solidChamberInner", 0., fChamberInnerRadius, fChamberLength/2., 0., 360.*deg);

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
    G4double endCapLocalFrontZ = -fChamberLength/2. + fChamberWallThickness/2.;
    G4double endCapLocalBackZ = fChamberLength/2. - fChamberWallThickness/2.;
    new G4PVPlacement(0, G4ThreeVector(0., 0., endCapLocalFrontZ), logicEndCapFront, "physEndCapFront", logicChamberVacuum, false, 0, true);
    new G4PVPlacement(0, G4ThreeVector(0., 0., endCapLocalBackZ), logicEndCapBack, "physEndCapBack", logicChamberVacuum, false, 1, true);

    // Target exit scoring volume (placed inside vacuum chamber, after front end cap)
    // Target rear face is at zPos (current value), scoring plane just after front end cap
    G4double scoringZ_local = endCapLocalFrontZ + fChamberWallThickness/2.;  // Just after front end cap
    
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

    zPos += fChamberLength/2.;
    new G4PVPlacement(0, G4ThreeVector(0., 0., zPos), logicChamberWall, "physChamberWall", worldLV, false, 0, true);
    new G4PVPlacement(0, G4ThreeVector(0., 0., zPos), logicChamberVacuum, "physChamberVacuum", worldLV, false, 0, true);

    zPos += fChamberLength/2.;
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
    
    if (fGapDistance > 0) {
        // Gap scan mode: position calorimeter based on gap distance
        fCaloEntranceZ = fTargetExitZ + fGapDistance;
        
        G4cout << "\n=== CALORIMETER POSITION (Gap Scan Mode) ===" << G4endl;
        G4cout << "Target exit Z: " << fTargetExitZ/cm << " cm" << G4endl;
        G4cout << "Gap distance: " << fGapDistance/cm << " cm" << G4endl;
        G4cout << "Calo entrance Z: " << fCaloEntranceZ/cm << " cm" << G4endl;
        
        // Place scoring plane at calorimeter entrance
        auto* solidScoringCaloEntrance = new G4Box("solidScoringCaloEntrance", fCaloSizeXY/2., fCaloSizeXY/2., scoringHalfThickness);
        fLogicScoringCaloEntrance = new G4LogicalVolume(solidScoringCaloEntrance,
                                                        fMatVacuum,
                                                        "logicScoringCaloEntrance");
        auto* scoringEntranceVis = new G4VisAttributes(G4Colour(0.0, 1.0, 0.0, 1.0));
        scoringEntranceVis->SetForceSolid(true);
        fLogicScoringCaloEntrance->SetVisAttributes(scoringEntranceVis);
        
        new G4PVPlacement(0, G4ThreeVector(0., 0., fCaloEntranceZ),
                          fLogicScoringCaloEntrance, "physScoringCaloEntrance",
                          worldLV, false, 0, true);
        
        G4double ecalFrontZ = fCaloEntranceZ + scoringHalfThickness;
        G4double ecalCenterZ = ecalFrontZ + ecalDepth / 2.;
        
        G4cout << "ECAL front face absolute Z: " << ecalFrontZ/cm << " cm" << G4endl;
        G4cout << "ECAL center absolute Z: " << ecalCenterZ/cm << " cm" << G4endl;
        
        // Create ECAL container
        auto* solidECAL = new G4Box("solidECAL", fCaloSizeXY/2., fCaloSizeXY/2., ecalDepth/2.);
        auto* logicECAL = new G4LogicalVolume(solidECAL, fMatAir, "logicECAL");
        logicECAL->SetVisAttributes(G4VisAttributes::GetInvisible());

        auto* solidCrystal = new G4Box("solidCrystal", 6.0*cm, 0.5*cm, 0.5*cm);
        fLogicCrystal = new G4LogicalVolume(solidCrystal, fMatCsI, "logicCrystal");

        auto* caloVis = new G4VisAttributes(G4Colour(1.0, 0.0, 1.0, 0.5));
        caloVis->SetForceSolid(true);
        fLogicCrystal->SetVisAttributes(caloVis);

        for(G4int layer = 0; layer < fNumCaloLayers; layer++) {
            G4double localZ = -ecalDepth/2. + fLayerThickness/2. + layer * fLayerThickness;
            G4bool isXOriented = (layer % 2 == 0);

            G4RotationMatrix* rot = nullptr;
            if(!isXOriented) {
                rot = new G4RotationMatrix();
                rot->rotateZ(90.*deg);
            }

            for(G4int crystal = 0; crystal < fNumCrystalsPerLayer; crystal++) {
                G4double offset = -fCaloSizeXY/2. + 0.5*cm + crystal * 1.0*cm;
                G4ThreeVector position;

                if(isXOriented) {
                    position = G4ThreeVector(0., offset, localZ);
                } else {
                    position = G4ThreeVector(offset, 0., localZ);
                }

                G4int copyNo = layer * fNumCrystalsPerLayer + crystal;
                new G4PVPlacement(rot, position, fLogicCrystal, "physCalorimeter", logicECAL, false, copyNo, true);
            }
        }

        // Scoring plane at calorimeter exit
        auto* solidScoringCaloExit = new G4Box("solidScoringCaloExit", fCaloSizeXY/2., fCaloSizeXY/2., scoringHalfThickness);
        fLogicScoringCaloExit = new G4LogicalVolume(solidScoringCaloExit,
                                                    fMatVacuum,
                                                    "logicScoringCaloExit");
        auto* scoringExitVis = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 1.0));
        scoringExitVis->SetForceSolid(true);
        fLogicScoringCaloExit->SetVisAttributes(scoringExitVis);
        
        new G4PVPlacement(0, G4ThreeVector(0., 0., ecalCenterZ), logicECAL, "physECAL", worldLV, false, 0, true);
        
        G4double caloExitZ = fCaloEntranceZ + scoringHalfThickness + ecalDepth + 0.1*mm + scoringHalfThickness;
        new G4PVPlacement(0, G4ThreeVector(0., 0., caloExitZ),
                          fLogicScoringCaloExit, "physScoringCaloExit",
                          worldLV, false, 0, true);
        
        G4cout << "ECAL back face absolute Z: " << (fCaloEntranceZ + scoringHalfThickness + ecalDepth)/cm << " cm" << G4endl;
        G4cout << "Calo exit scoring absolute Z: " << caloExitZ/cm << " cm" << G4endl;
        
    } else {
        // Normal mode: position calorimeter after magnet (original behavior)
        G4cout << "\n=== CALORIMETER ENTRANCE SCORING GEOMETRY (Normal Mode) ===" << G4endl;
        G4cout << "ECAL depth: " << ecalDepth/cm << " cm" << G4endl;
        G4cout << "Magnet region ends at Z: " << zPos/cm << " cm" << G4endl;
        
        // Scoring plane at calorimeter entrance
        auto* solidScoringCaloEntrance = new G4Box("solidScoringCaloEntrance", fCaloSizeXY/2., fCaloSizeXY/2., scoringHalfThickness);
        fLogicScoringCaloEntrance = new G4LogicalVolume(solidScoringCaloEntrance,
                                                        fMatVacuum,
                                                        "logicScoringCaloEntrance");
        auto* scoringEntranceVis = new G4VisAttributes(G4Colour(0.0, 1.0, 0.0, 1.0));
        scoringEntranceVis->SetForceSolid(true);
        fLogicScoringCaloEntrance->SetVisAttributes(scoringEntranceVis);
        
        G4double caloEntranceZ = zPos + 1.0*mm + scoringHalfThickness;
        new G4PVPlacement(0, G4ThreeVector(0., 0., caloEntranceZ),
                          fLogicScoringCaloEntrance, "physScoringCaloEntrance",
                          worldLV, false, 0, true);
        
        G4cout << "Calo entrance scoring absolute Z: " << caloEntranceZ/cm << " cm" << G4endl;
        
        G4double gapForScoring = 2.0*mm + 2*scoringHalfThickness;
        zPos += gapForScoring;
        
        G4cout << "ECAL front face absolute Z: " << zPos/cm << " cm" << G4endl;
        G4cout << "ECAL will be centered at absolute Z: " << (zPos + ecalDepth/2.)/cm << " cm" << G4endl;
        
        fCaloEntranceZ = caloEntranceZ;

        // Create ECAL container
        auto* solidECAL = new G4Box("solidECAL", fCaloSizeXY/2., fCaloSizeXY/2., ecalDepth/2.);
        auto* logicECAL = new G4LogicalVolume(solidECAL, fMatAir, "logicECAL");
        logicECAL->SetVisAttributes(G4VisAttributes::GetInvisible());

        auto* solidCrystal = new G4Box("solidCrystal", 6.0*cm, 0.5*cm, 0.5*cm);
        fLogicCrystal = new G4LogicalVolume(solidCrystal, fMatCsI, "logicCrystal");

        auto* caloVis = new G4VisAttributes(G4Colour(1.0, 0.0, 1.0, 0.5));
        caloVis->SetForceSolid(true);
        fLogicCrystal->SetVisAttributes(caloVis);

        for(G4int layer = 0; layer < fNumCaloLayers; layer++) {
            G4double localZ = -ecalDepth/2. + fLayerThickness/2. + layer * fLayerThickness;
            G4bool isXOriented = (layer % 2 == 0);

            G4RotationMatrix* rot = nullptr;
            if(!isXOriented) {
                rot = new G4RotationMatrix();
                rot->rotateZ(90.*deg);
            }

            for(G4int crystal = 0; crystal < fNumCrystalsPerLayer; crystal++) {
                G4double offset = -fCaloSizeXY/2. + 0.5*cm + crystal * 1.0*cm;
                G4ThreeVector position;

                if(isXOriented) {
                    position = G4ThreeVector(0., offset, localZ);
                } else {
                    position = G4ThreeVector(offset, 0., localZ);
                }

                G4int copyNo = layer * fNumCrystalsPerLayer + crystal;
                new G4PVPlacement(rot, position, fLogicCrystal, "physCalorimeter", logicECAL, false, copyNo, true);
            }
        }

        // Scoring plane at calorimeter exit
        auto* solidScoringCaloExit = new G4Box("solidScoringCaloExit", fCaloSizeXY/2., fCaloSizeXY/2., scoringHalfThickness);
        fLogicScoringCaloExit = new G4LogicalVolume(solidScoringCaloExit,
                                                    fMatVacuum,
                                                    "logicScoringCaloExit");
        auto* scoringExitVis = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 1.0));
        scoringExitVis->SetForceSolid(true);
        fLogicScoringCaloExit->SetVisAttributes(scoringExitVis);
        
        zPos += ecalDepth/2.;
        new G4PVPlacement(0, G4ThreeVector(0., 0., zPos), logicECAL, "physECAL", worldLV, false, 0, true);
        zPos += ecalDepth/2.;
        
        G4double caloExitZ = zPos + 0.1*mm + scoringHalfThickness;
        new G4PVPlacement(0, G4ThreeVector(0., 0., caloExitZ),
                          fLogicScoringCaloExit, "physScoringCaloExit",
                          worldLV, false, 0, true);
        
        G4cout << "ECAL back face absolute Z: " << zPos/cm << " cm" << G4endl;
        G4cout << "Calo exit scoring absolute Z: " << caloExitZ/cm << " cm" << G4endl;
    }
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

    fLogicCrystal->SetSensitiveDetector(calorimeterSD);

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

    // Mid-target scoring plane
    auto* scoringTargetMidSD = new G4MultiFunctionalDetector("ScoringTargetMidSD");
    sdManager->AddNewDetector(scoringTargetMidSD);
    
    auto* scoringTargetMidEnergyDep = new G4PSEnergyDeposit("EnergyDeposit");
    scoringTargetMidSD->RegisterPrimitive(scoringTargetMidEnergyDep);
    
    auto* scoringTargetMidNofSecondary = new G4PSNofSecondary("NofSecondary");
    scoringTargetMidSD->RegisterPrimitive(scoringTargetMidNofSecondary);
    
    fLogicScoringTargetMid->SetSensitiveDetector(scoringTargetMidSD);

    // fMagField = new MagneticField();
    // auto* fieldMgr = new G4FieldManager();
    // fieldMgr->SetDetectorField(fMagField);
    // fieldMgr->CreateChordFinder(fMagField);
    // fLogicMagnetHollow->SetFieldManager(fieldMgr, true);
}
