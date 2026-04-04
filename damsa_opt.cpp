/**
 * @file damsa_opt.cpp
 * @brief DAMSA optimization batch runner
 * 
 * Runs a single simulation with geometry parameters specified via command line
 * and outputs flux data for Python optimization scripts.
 * 
 * Usage:
 *   ./damsa_opt --target-z 10 --target-xy 5 --gap 50 --n-events 1000 --output-dir ./output
 * 
 * Output files (in output-dir):
 *   - photon_flux_target_exit.csv  : Photon flux at target exit
 *   - background_target_exit.csv   : Background particles at target exit
 *   - simulation_info.json         : Metadata about the run
 * 
 * Author: DAMSA Collaboration
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4SystemOfUnits.hh"
#include "QBBC.hh"

#include "construction.h"
#include "action.h"
#include "analysis.h"
#include "FluxData.h"

// Simple command-line argument parser
struct OptConfig {
    G4double targetZ = 10.0;      // cm
    G4double targetXY = 5.0;      // cm (square target)
    G4double gap = 0.0;           // cm (no gap by default)
    G4int nEvents = 1000;
    std::string outputDir = "output";
    bool verbose = true;
    bool optimizeMode = false;    // Use config-specific filenames
    
    bool parse(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "--target-z" && i + 1 < argc) {
                targetZ = std::atof(argv[++i]);
            }
            else if (arg == "--target-xy" && i + 1 < argc) {
                targetXY = std::atof(argv[++i]);
            }
            else if (arg == "--gap" && i + 1 < argc) {
                gap = std::atof(argv[++i]);
            }
            else if (arg == "--n-events" && i + 1 < argc) {
                nEvents = std::atoi(argv[++i]);
            }
            else if (arg == "--output-dir" && i + 1 < argc) {
                outputDir = argv[++i];
            }
            else if (arg == "--quiet") {
                verbose = false;
            }
            else if (arg == "--optimize") {
                optimizeMode = true;
            }
            else if (arg == "--help" || arg == "-h") {
                printHelp();
                return false;
            }
        }
        return true;
    }
    
    void printHelp() {
        G4cout << "DAMSA Optimization Batch Runner\n"
               << "\nUsage: damsa_opt [options]\n"
               << "\nOptions:\n"
               << "  --target-z <cm>      Target thickness (default: 10.0)\n"
               << "  --target-xy <cm>     Target transverse size (default: 5.0)\n"
               << "  --gap <cm>           Gap distance (default: 47.0)\n"
               << "  --n-events <N>       Number of events (default: 1000)\n"
               << "  --output-dir <dir>   Output directory (default: output)\n"
               << "  --optimize           Use config-specific filenames (e.g., Tz10_xy5_G50_)\n"
               << "  --quiet              Suppress verbose output\n"
               << "  --help, -h           Show this help\n"
               << G4endl;
    }
    
    void print() const {
        G4cout << "=== DAMSA Optimization Run ===" << G4endl;
        G4cout << "Target Z:     " << targetZ << " cm" << G4endl;
        G4cout << "Target XY:    " << targetXY << " cm" << G4endl;
        G4cout << "Gap:          " << gap << " cm" << G4endl;
        G4cout << "Events:       " << nEvents << G4endl;
        G4cout << "Output dir:   " << outputDir << G4endl;
        G4cout << "Optimize:     " << (optimizeMode ? "yes" : "no") << G4endl;
        G4cout << "===============================" << G4endl;
    }
    
    // Generate config prefix string for filenames (e.g., "Tz10_xy5_G50_")
    std::string getConfigPrefix() const {
        if (!optimizeMode) return "";
        char buf[64];
        snprintf(buf, sizeof(buf), "Tz%.0f_xy%.0f_G%.0f_", targetZ, targetXY, gap);
        return std::string(buf);
    }
};

void writeSimulationInfo(const OptConfig& config, const std::string& outputDir,
                         G4int exitPhotons, G4int exitNeutrons,
                         G4int caloPhotons, G4int caloNeutrons,
                         G4double runTime) {
    std::string prefix = config.getConfigPrefix();
    std::string filepath = outputDir + "/" + prefix + "simulation_info.json";
    std::ofstream outFile(filepath);
    
    if (!outFile.is_open()) {
        G4cerr << "ERROR: Could not write " << filepath << G4endl;
        return;
    }
    
    outFile << "{\n";
    outFile << "  \"target_z_cm\": " << config.targetZ << ",\n";
    outFile << "  \"target_xy_cm\": " << config.targetXY << ",\n";
    outFile << "  \"gap_cm\": " << config.gap << ",\n";
    outFile << "  \"n_events\": " << config.nEvents << ",\n";
    outFile << "  \"exit_photons\": " << exitPhotons << ",\n";
    outFile << "  \"exit_neutrons\": " << exitNeutrons << ",\n";
    outFile << "  \"calo_entrance_photons\": " << caloPhotons << ",\n";
    outFile << "  \"calo_entrance_neutrons\": " << caloNeutrons << ",\n";
    outFile << "  \"run_time_seconds\": " << runTime << "\n";
    outFile << "}\n";
    
    outFile.close();
}

int main(int argc, char** argv)
{
    OptConfig config;
    if (!config.parse(argc, argv)) {
        return 0;  // Help was printed
    }
    
    if (config.verbose) {
        config.print();
    }
    
    // Create output directory
    mkdir(config.outputDir.c_str(), 0755);
    
    // Initialize Geant4
    G4RunManager* runManager = new G4RunManager();
    
    // Set up detector with specified geometry
    DamsaDetectorConstruction* detector = new DamsaDetectorConstruction();
    detector->SetTargetLength(config.targetZ * cm);
    detector->SetTargetTransverse(config.targetXY * cm);
    detector->SetGapDistance(config.gap * cm);
    
    runManager->SetUserInitialization(detector);
    
    // Physics list
    G4VModularPhysicsList* physicsList = new QBBC();
    physicsList->SetVerboseLevel(config.verbose ? 1 : 0);
    runManager->SetUserInitialization(physicsList);
    
    // User actions
    runManager->SetUserInitialization(new DamsaActionInitialization());
    
    runManager->Initialize();
    
    // Reset flux collector
    DamsaFluxCollector::Instance()->Reset();
    DamsaAnalysis::Instance()->Reset();
    
    // Set config prefix for output filenames if in optimize mode
    DamsaAnalysis::Instance()->SetConfigPrefix(config.getConfigPrefix());
    
    // Run simulation
    G4double startTime = std::clock();
    runManager->BeamOn(config.nEvents);
    G4double endTime = std::clock();
    G4double runTime = (endTime - startTime) / CLOCKS_PER_SEC;
    
    // Get statistics from target exit (FluxCollector)
    G4int exitPhotons = DamsaFluxCollector::Instance()->GetPhotonCount();
    G4int exitNeutrons = DamsaFluxCollector::Instance()->GetNeutronCount();
    
    // Get statistics from calorimeter entrance (Analysis)
    G4int caloPhotons = DamsaAnalysis::Instance()->GetCaloEntrancePhotons();
    G4int caloNeutrons = DamsaAnalysis::Instance()->GetCaloEntranceNeutrons();
    
    if (config.verbose) {
        G4cout << "\n=== Run Complete ===" << G4endl;
        G4cout << "Exit photons:  " << exitPhotons << G4endl;
        G4cout << "Exit neutrons: " << exitNeutrons << G4endl;
        G4cout << "Calo photons:  " << caloPhotons << G4endl;
        G4cout << "Calo neutrons: " << caloNeutrons << G4endl;
        G4cout << "Run time:      " << runTime << " s" << G4endl;
    }
    
    // Write output files
    // Note: FluxData.h writes to "output/" directory by default
    // We need to use the configured output directory
    std::string filePrefix = config.getConfigPrefix();
    
    // Write photon flux CSV
    DamsaFluxCollector::Instance()->WritePhotonFluxCSV(filePrefix + "photon_flux_target_exit.csv");
    
    // Write background CSV
    DamsaFluxCollector::Instance()->WriteBackgroundCSV(filePrefix + "background_target_exit.csv");
    
    // Write alplib-compatible flux
    DamsaFluxCollector::Instance()->WriteAlplibFlux(filePrefix + "alplib_flux.csv", 
                                                     config.nEvents, 
                                                     62.5e-6);  // 62.5 uA beam
    
    // Write simulation metadata
    writeSimulationInfo(config, "output", exitPhotons, exitNeutrons, 
                        caloPhotons, caloNeutrons, runTime);
    
    // Cleanup
    delete runManager;
    
    if (config.verbose) {
        G4cout << "Output written to: " << config.outputDir << "/" << G4endl;
    }
    
    return 0;
}
