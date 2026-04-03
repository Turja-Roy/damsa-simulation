#include <iostream>

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "QBBC.hh"

#include "construction.h"
#include "action.h"
#include "analysis.h"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <vector>
#include <iomanip>
#include <ctime>

#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TLatex.h"

int main(int argc, char** argv)
{
    G4cout << "\n=============================================" << G4endl;
    G4cout << "=== DAMSA Gap Optimization Scan ===" << G4endl;
    G4cout << "=============================================" << G4endl;
    
    G4cout << "\nSelect scan type:" << G4endl;
    G4cout << "1. Gap distance only (baseline target)" << G4endl;
    G4cout << "2. Target length only (baseline gap)" << G4endl;
    G4cout << "3. Target transverse size only (baseline gap)" << G4endl;
    G4cout << "4. Full optimization (gap + target geometry)" << G4endl;
    G4cout << "Enter choice (1-4): ";
    
    int choice;
    if (argc > 1) {
        choice = std::atoi(argv[1]);
    } else {
        std::cin >> choice;
    }
    
    std::vector<G4double> gapDistances = {20, 30, 40, 47, 50, 60, 70, 80, 90, 100};
    std::vector<G4double> targetLengths = {5, 10, 15, 20};
    std::vector<G4double> targetTransverse = {5, 7.5, 10};
    
    G4double baseGap = 47.0;
    G4double baseLength = 10.0;
    G4double baseWidth = 5.0;
    
    const G4int eventsPerConfig = 1000;
    
    std::vector<G4double> gapScan, lengthScan, widthScan;
    
    switch(choice) {
        case 1:
            gapScan = gapDistances;
            lengthScan = {baseLength};
            widthScan = {baseWidth};
            break;
        case 2:
            gapScan = {baseGap};
            lengthScan = targetLengths;
            widthScan = {baseWidth};
            break;
        case 3:
            gapScan = {baseGap};
            lengthScan = {baseLength};
            widthScan = targetTransverse;
            break;
        case 4:
            gapScan = gapDistances;
            lengthScan = targetLengths;
            widthScan = targetTransverse;
            G4cout << "\n Full scan will run " 
                   << (gapScan.size() * lengthScan.size() * widthScan.size())
                   << " configurations!" << G4endl;
            G4cout << "Estimated time: ~" 
                   << (gapScan.size() * lengthScan.size() * widthScan.size() * eventsPerConfig * 0.1 / 3600.0) 
                   << " hours" << G4endl;
            G4cout << "Continue? (y/n): ";
            char confirm;
            std::cin >> confirm;
            if(confirm != 'y' && confirm != 'Y') {
                G4cout << "Scan cancelled." << G4endl;
                return 0;
            }
            break;
        default:
            G4cout << "Invalid choice!" << G4endl;
            return 1;
    }
    
    std::time_t now = std::time(nullptr);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    
    std::string filename;
    switch(choice) {
        case 1: filename = "gap_scan_"; break;
        case 2: filename = "target_length_scan_"; break;
        case 3: filename = "target_transverse_scan_"; break;
        case 4: filename = "full_optimization_"; break;
    }
    filename += timestamp;
    filename += ".txt";
    
    // Try to create output file - use current directory first
    std::ofstream outFile(filename.c_str());
    outFile << "# DAMSA Gap Optimization Scan Results\n";
    outFile << "# Scan type: " << choice << "\n";
    outFile << "# Events per configuration: " << eventsPerConfig << "\n#\n";
    outFile << std::setw(8) << "Config" << "  ";
    outFile << std::setw(8) << "Gap_cm" << "  ";
    outFile << std::setw(10) << "TgtLen_cm" << "  ";
    outFile << std::setw(10) << "TgtWid_cm" << "  ";
    outFile << std::setw(10) << "CsI_Z_cm" << "  ";
    outFile << std::setw(12) << "ExitPhotons" << "  ";
    outFile << std::setw(12) << "ExitNeutrons" << "  ";
    outFile << std::setw(12) << "DetPhotons" << "  ";
    outFile << std::setw(12) << "DetNeutrons" << "  ";
    outFile << std::setw(12) << "FwdPhotons" << "  ";
    outFile << std::setw(10) << "PhotonAcc%" << "  ";
    outFile << std::setw(8) << "S/B" << "\n";
    
    G4RunManager* runManager = new G4RunManager;
    
    G4VModularPhysicsList* physicsList = new QBBC;
    physicsList->SetVerboseLevel(0);
    runManager->SetUserInitialization(physicsList);
    
    runManager->SetUserInitialization(new DamsaActionInitialization());
    
    int configNum = 0;
    int totalConfigs = gapScan.size() * lengthScan.size() * widthScan.size();
    
    G4cout << "\nTotal configurations: " << totalConfigs << "\n" << G4endl;
    
    for (G4double gap : gapScan) {
        for (G4double length : lengthScan) {
            for (G4double width : widthScan) {
                
                configNum++;
                
                G4cout << "\n--- Configuration " << configNum << "/" << totalConfigs << " ---" << G4endl;
                G4cout << "Gap: " << gap << " cm, Target: " << width << "x" << width << "x" << length << " cm3" << G4endl;
                
                DamsaDetectorConstruction* detector = new DamsaDetectorConstruction();
                detector->SetGapDistance(gap * cm);
                detector->SetTargetLength(length * cm);
                detector->SetTargetTransverse(width * cm);
                
                runManager->SetUserInitialization(detector);
                
                if(configNum == 1) {
                    runManager->Initialize();
                } else {
                    runManager->ReinitializeGeometry(true);
                }
                
                DamsaAnalysis::Instance()->Reset();
                
                runManager->BeamOn(eventsPerConfig);
                
                G4int exitPhotons = DamsaAnalysis::Instance()->GetTargetExitPhotons();
                G4int exitNeutrons = DamsaAnalysis::Instance()->GetTargetExitNeutrons();
                G4int detPhotons = DamsaAnalysis::Instance()->GetCaloEntrancePhotons();
                G4int detNeutrons = DamsaAnalysis::Instance()->GetCaloEntranceNeutrons();
                G4int fwdPhotons = DamsaAnalysis::Instance()->GetForwardPhotons();
                
                G4double photonAcc = (exitPhotons > 0) ? 100.0 * detPhotons / exitPhotons : 0.0;
                G4double signalToBackground = (detNeutrons > 0) ? (G4double)detPhotons / detNeutrons : detPhotons;
                
                G4cout << "Results: " << detPhotons << " photons, " << detNeutrons 
                       << " neutrons, Efficiency: " << photonAcc << "%" << G4endl;
                
                G4double csiZ = detector->GetCaloEntranceZ() / cm;
                
                outFile << std::setw(8) << configNum << "  ";
                outFile << std::setw(8) << gap << "  ";
                outFile << std::setw(10) << length << "  ";
                outFile << std::setw(10) << width << "  ";
                outFile << std::setw(10) << std::fixed << std::setprecision(2) << csiZ << "  ";
                outFile << std::setw(12) << exitPhotons << "  ";
                outFile << std::setw(12) << exitNeutrons << "  ";
                outFile << std::setw(12) << detPhotons << "  ";
                outFile << std::setw(12) << detNeutrons << "  ";
                outFile << std::setw(12) << fwdPhotons << "  ";
                outFile << std::setw(10) << std::fixed << std::setprecision(2) << photonAcc << "  ";
                outFile << std::setw(8) << std::fixed << std::setprecision(2) << signalToBackground << "\n";
                outFile.flush();
            }
        }
    }
    
    outFile.close();
    
    G4cout << "\n=============================================" << G4endl;
    G4cout << "Scan complete! Results: " << filename << G4endl;
    G4cout << "=============================================" << G4endl;
    
    // Read back the results file and create ROOT plots
    G4cout << "\n=== Generating Analysis Plots ===" << G4endl;
    
    std::ifstream inFile(filename.c_str());
    std::vector<double> gaps, csiZ, exitPhotons, exitNeutrons, detPhotons, detNeutrons, fwdPhotons, efficiencies;
    
    std::string line;
    while (std::getline(inFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int config; double gap, csi; int exP, exN, detP, detN, fwdP; double eff, sb;
        if (iss >> config >> gap >> csi >> exP >> exN >> detP >> detN >> fwdP >> eff >> sb) {
            gaps.push_back(gap);
            csiZ.push_back(csi);
            exitPhotons.push_back(exP);
            exitNeutrons.push_back(exN);
            detPhotons.push_back(detP);
            detNeutrons.push_back(detN);
            fwdPhotons.push_back(fwdP);
            efficiencies.push_back(eff);
        }
    }
    inFile.close();
    
    if (gaps.empty()) {
        G4cout << "Error: No data found in results file!" << G4endl;
    } else {
        // Find optimal configuration
        int optimalIdx = 0;
        double maxEff = 0;
        for (size_t i = 0; i < efficiencies.size(); i++) {
            if (efficiencies[i] > maxEff) {
                maxEff = efficiencies[i];
                optimalIdx = i;
            }
        }
        
        G4cout << "Optimal configuration: Gap = " << gaps[optimalIdx] 
               << " cm, Efficiency = " << efficiencies[optimalIdx] << "%" << G4endl;
        
        // Create ROOT file and plots
        TFile* rootFile = TFile::Open("gap_scan_results.root", "RECREATE");
        if (rootFile && !rootFile->IsZombie()) {
            // Efficiency vs Gap
            TGraph* grEff = new TGraph(gaps.size(), gaps.data(), efficiencies.data());
            grEff->SetName("grEfficiency");
            grEff->SetTitle("Photon Detection Efficiency vs Gap Distance");
            grEff->GetXaxis()->SetTitle("Gap Distance (cm)");
            grEff->GetYaxis()->SetTitle("Efficiency (%)");
            grEff->SetMarkerStyle(20);
            grEff->SetMarkerColor(kBlue);
            grEff->SetLineColor(kBlue);
            grEff->Write();
            
            // Detector Photons vs Gap
            TGraph* grDetP = new TGraph(gaps.size(), gaps.data(), detPhotons.data());
            grDetP->SetName("grDetectorPhotons");
            grDetP->SetTitle("Photons Reaching Detector vs Gap Distance");
            grDetP->GetXaxis()->SetTitle("Gap Distance (cm)");
            grDetP->GetYaxis()->SetTitle("Number of Photons");
            grDetP->SetMarkerStyle(21);
            grDetP->SetMarkerColor(kGreen);
            grDetP->SetLineColor(kGreen);
            grDetP->Write();
            
            // Forward Photons vs Gap
            TGraph* grFwdP = new TGraph(gaps.size(), gaps.data(), fwdPhotons.data());
            grFwdP->SetName("grForwardPhotons");
            grFwdP->SetTitle("Forward Photons (0-20 deg) vs Gap Distance");
            grFwdP->GetXaxis()->SetTitle("Gap Distance (cm)");
            grFwdP->GetYaxis()->SetTitle("Number of Photons");
            grFwdP->SetMarkerStyle(22);
            grFwdP->SetMarkerColor(kMagenta);
            grFwdP->SetLineColor(kMagenta);
            grFwdP->Write();
            
            // Neutrons vs Gap
            TGraph* grDetN = new TGraph(gaps.size(), gaps.data(), detNeutrons.data());
            grDetN->SetName("grDetectorNeutrons");
            grDetN->SetTitle("Neutrons Reaching Detector vs Gap Distance");
            grDetN->GetXaxis()->SetTitle("Gap Distance (cm)");
            grDetN->GetYaxis()->SetTitle("Number of Neutrons");
            grDetN->SetMarkerStyle(23);
            grDetN->SetMarkerColor(kOrange);
            grDetN->SetLineColor(kOrange);
            grDetN->Write();
            
            rootFile->Close();
            delete rootFile;
            G4cout << "ROOT file saved: gap_scan_results.root" << G4endl;
            
            // Create summary canvas
            TCanvas* cSummary = new TCanvas("cSummary", "Gap Scan Summary", 1200, 800);
            cSummary->Divide(2, 2);
            
            cSummary->cd(1);
            grEff->Draw("APL");
            grEff->GetXaxis()->SetRangeUser(gaps[0] - 5, gaps[gaps.size()-1] + 5);
            grEff->GetYaxis()->SetRangeUser(0, maxEff * 1.2);
            // Mark optimal point
            TGraph* grOpt = new TGraph(1);
            grOpt->SetPoint(0, gaps[optimalIdx], efficiencies[optimalIdx]);
            grOpt->SetMarkerStyle(34);
            grOpt->SetMarkerSize(3);
            grOpt->SetMarkerColor(kRed);
            grOpt->Draw("P");
            
            cSummary->cd(2);
            grDetP->Draw("APL");
            grDetP->GetXaxis()->SetRangeUser(gaps[0] - 5, gaps[gaps.size()-1] + 5);
            
            cSummary->cd(3);
            grFwdP->Draw("APL");
            grFwdP->GetXaxis()->SetRangeUser(gaps[0] - 5, gaps[gaps.size()-1] + 5);
            
            cSummary->cd(4);
            grDetN->Draw("APL");
            grDetN->GetXaxis()->SetRangeUser(gaps[0] - 5, gaps[gaps.size()-1] + 5);
            
            cSummary->SaveAs("gap_scan_summary.png");
            G4cout << "Summary plot saved: gap_scan_summary.png" << G4endl;
            
            delete cSummary;
        }
    }
    
    G4cout << "\n=== Analysis Complete ===" << G4endl;
    
    delete runManager;
    return 0;
}
