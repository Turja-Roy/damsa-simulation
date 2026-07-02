#include "analysis.h"
#include "FluxData.h"
#include "pi0DecayData.h"
#include "G4ThreeVector.hh"
#include <algorithm>

TCanvas* CreateOverlayCanvasMap(std::map<G4String, TH1D*>& energyHists,
                               std::map<G4String, TH1D*>& angleHists,
                               const G4String& locationName);
void WriteEvolutionHistograms(TFile* rootFile, std::map<G4String, DamsaLocationData>& locations,
                              const std::string& configPrefix);

DamsaAnalysis* DamsaAnalysis::fInstance = nullptr;
G4Mutex DamsaAnalysis::fMutex;

DamsaFluxCollector* DamsaFluxCollector::fInstance = nullptr;
G4Mutex DamsaFluxCollector::fMutex;

DamsaPi0Collector* DamsaPi0Collector::fInstance = nullptr;
G4Mutex DamsaPi0Collector::fMutex;
G4double DamsaPi0Collector::fGeoCaloZ      = 0.0;
G4double DamsaPi0Collector::fGeoCaloHalfXY = 60.0;  // default: 12 cm / 2

DamsaAnalysis* DamsaAnalysis::Instance()
{
    if(!fInstance) {
        fInstance = new DamsaAnalysis();
    }
    return fInstance;
}

DamsaAnalysis::DamsaAnalysis()
{
    fLocations["TargetExit"] = DamsaLocationData();
    fLocations["MagnetEntrance"] = DamsaLocationData();
    fLocations["CaloEntrance"] = DamsaLocationData();
    fLocations["CaloExit"] = DamsaLocationData();
}

DamsaAnalysis::~DamsaAnalysis()
{}

G4bool DamsaAnalysis::WasTrackRecorded(G4int trackID, const G4String& location)
{
    auto it = fLocations.find(location);
    if (it != fLocations.end()) {
        return it->second.WasTrackRecorded(trackID);
    }
    return false;
}

void DamsaAnalysis::ResetEventTracking()
{
    G4AutoLock lock(&fMutex);
    for (auto& pair : fLocations) {
        pair.second.ResetEventTracking();
    }
}

void DamsaAnalysis::RecordParticle(const G4String& particleName, G4double energy,
                                   const G4String& location, G4double angle,
                                   G4int trackID, G4bool isPrimary)
{
    G4AutoLock lock(&fMutex);
    auto it = fLocations.find(location);
    if (it != fLocations.end()) {
        it->second.RecordParticle(particleName, energy, angle, trackID, isPrimary);
    }
}

void DamsaAnalysis::PrintSummary()
{
    std::vector<G4String> order = {"TargetExit", "MagnetEntrance", "CaloEntrance", "CaloExit"};
    for (const G4String& locName : order) {
        fLocations[locName].PrintSummary(locName);
    }
}

void DamsaAnalysis::SaveToFile(const G4String& filename)
{
    mkdir("output", 0755);
    
    std::string fullPath = "output/" + filename;
    std::ofstream outFile(fullPath);
    
    if(!outFile.is_open()) {
        G4cout << "ERROR: Could not open file " << fullPath << " for writing!" << G4endl;
        return;
    }
    
    outFile << "=============================================" << std::endl;
    outFile << "=== DAMSA Background Analysis Results ===" << std::endl;
    outFile << "=============================================" << std::endl << std::endl;
    
    std::vector<G4String> order = {"TargetExit", "MagnetEntrance", "CaloEntrance", "CaloExit"};
    for (const G4String& locName : order) {
        fLocations[locName].WriteToFile(outFile, locName);
    }
    
    outFile << "=============================================" << std::endl;
    outFile.close();
    
    G4cout << "Analysis results saved to: " << fullPath << G4endl;
}

void DamsaAnalysis::WriteROOTHistograms(const G4String& filename)
{
    G4cout << "\n=== Generating Publication-Quality Plots ===" << G4endl;
    
    CreatePlotDirectories();
    SetPublicationStyle();
    
    // Get config prefix for filenames (empty if not in optimize mode)
    std::string prefix = fConfigPrefix;
    
    std::map<G4String, std::vector<double>> particleEnergies;
    std::map<G4String, std::vector<double>> particleAngles;
    
    for (auto& locPair : fLocations) {
        const G4String& locationName = locPair.first;
        const DamsaLocationData& locData = locPair.second;
        
        for (const auto& partPair : locData.GetParticleDataMap().GetMap()) {
            const G4String& particleName = partPair.first;
            const ParticleData& pdata = partPair.second;
            
            for (size_t i = 0; i < pdata.energies.size(); i++) {
                particleEnergies[particleName].push_back(pdata.energies[i]/GeV);
                particleAngles[particleName].push_back(pdata.angles[i]);
            }
        }
    }
    
    std::string rootPath = "plots/" + prefix + "damsa_analysis.root";
    TFile* rootFile = new TFile(rootPath.c_str(), "RECREATE");
    
    if(!rootFile || rootFile->IsZombie()) {
        G4cout << "ERROR: Could not create ROOT file " << rootPath << G4endl;
        return;
    }
    
    G4cout << "Creating histograms in organized ROOT file..." << G4endl;
    
    for (auto& locPair : fLocations) {
        const G4String& locationName = locPair.first;
        DamsaLocationData& locData = locPair.second;
        
        rootFile->mkdir(locationName.c_str());
        rootFile->mkdir((locationName + "/Energy").c_str());
        rootFile->mkdir((locationName + "/Angular").c_str());
        rootFile->mkdir((locationName + "/Correlation").c_str());
        rootFile->mkdir((locationName + "/Summary").c_str());
        
        rootFile->cd((locationName + "/Energy").c_str());
        
        std::map<G4String, TH1D*> energyHists;
        std::map<G4String, TH1D*> angleHists;
        std::map<G4String, TH2D*> corrHists;
        
        Color_t particleColors[] = {kRed, kBlue, kGreen+2, kMagenta, kCyan, kOrange, kPink, kTeal};
        
        G4int colorIdx = 0;
        for (const auto& partPair : locData.GetParticleDataMap().GetMap()) {
            const G4String& particleName = partPair.first;
            const ParticleData& pdata = partPair.second;
            
            if (pdata.count == 0) continue;
            
            std::vector<double> energies;
            std::vector<double> angles;
            for (size_t i = 0; i < pdata.energies.size(); i++) {
                energies.push_back(pdata.energies[i]/MeV);
                angles.push_back(pdata.angles[i]);
            }
            
            Color_t color = particleColors[colorIdx % 8];
            colorIdx++;
            
            TH1D* hEnergy = CreateEnergyHist(energies, 
                ("h" + particleName + "Energy").c_str(),
                (particleName + " Energy at " + locationName).c_str(),
                color, -1.0);
            hEnergy->Write();
            energyHists[particleName] = hEnergy;
            
            TH1D* hEnergyNorm = CreateNormalizedCopy(hEnergy, "_norm");
            hEnergyNorm->Write();
            
            TH1D* hAngle = CreateAngleHist(angles,
                ("h" + particleName + "Angle").c_str(),
                (particleName + " Angle at " + locationName).c_str(),
                color);
            hAngle->Write();
            angleHists[particleName] = hAngle;
            
            TH1D* hAngleNorm = CreateNormalizedCopy(hAngle, "_norm");
            hAngleNorm->Write();
            
            rootFile->cd((locationName + "/Correlation").c_str());
            TH2D* hCorr = CreateEnergyAngleHist(energies, angles,
                ("h2" + particleName + "EvsA").c_str(),
                (particleName + " Energy vs Angle at " + locationName).c_str());
            hCorr->Write();
            corrHists[particleName] = hCorr;
        }
        
        rootFile->cd((locationName + "/Summary").c_str());
        
        G4int nParticles = locData.GetParticleDataMap().GetMap().size();
        if (nParticles > 0) {
            TH1D* hCounts = new TH1D("hParticleCounts", "Particle Counts", nParticles, 0, nParticles);
            G4int bin = 1;
            G4double totalEnergy = 0;
            G4double primaryEnergy = 0;
            for (const auto& partPair : locData.GetParticleDataMap().GetMap()) {
                hCounts->SetBinContent(bin, partPair.second.count);
                hCounts->GetXaxis()->SetBinLabel(bin, GetParticleName(partPair.first).c_str());
                totalEnergy += locData.GetTotalEnergy();
                primaryEnergy += locData.GetPrimaryEnergy();
                bin++;
            }
            hCounts->Write();
            
            TH1D* hEnergyBudget = CreateEnergyBudgetHist(
                locData.GetPrimaryEnergy()/GeV,
                (locData.GetTotalEnergy() - locData.GetPrimaryEnergy())/GeV,
                "hEnergyBudget", "Energy Budget");
            hEnergyBudget->Write();
            
            TH1D* hMeanEnergy = new TH1D("hMeanEnergy", "Mean Energy per Particle Type", nParticles, 0, nParticles);
            TH1D* hRMSEnergy = new TH1D("hRMSEnergy", "Energy RMS per Particle Type", nParticles, 0, nParticles);
            bin = 1;
            for (const auto& partPair : locData.GetParticleDataMap().GetMap()) {
                const ParticleData& pdata = partPair.second;
                double sumE = 0, sumE2 = 0;
                for (double e : pdata.energies) {
                    sumE += e / GeV;
                    sumE2 += (e / GeV) * (e / GeV);
                }
                double meanE = (pdata.count > 0) ? sumE / pdata.count : 0;
                double rmsE = (pdata.count > 1) ? sqrt(sumE2 / pdata.count - meanE * meanE) : 0;
                hMeanEnergy->SetBinContent(bin, meanE);
                hRMSEnergy->SetBinContent(bin, rmsE);
                hMeanEnergy->GetXaxis()->SetBinLabel(bin, GetParticleName(partPair.first).c_str());
                hRMSEnergy->GetXaxis()->SetBinLabel(bin, GetParticleName(partPair.first).c_str());
                bin++;
            }
            hMeanEnergy->Write();
            hRMSEnergy->Write();
        }
        
        TCanvas* cOverlay = CreateOverlayCanvasMap(energyHists, angleHists, locationName);
        if (cOverlay) {
            SaveCanvasWithPrefix(cOverlay, "plots/png/" + locationName, "overlay", prefix);
            delete cOverlay;
        }
        
        TCanvas* c1 = new TCanvas("c1", "", 800, 600);
        for (auto& eh : energyHists) {
            eh.second->Draw("HIST");
            SaveCanvasWithPrefix(c1, "plots/png/" + locationName, "energy_" + eh.first, prefix);
        }
        
        c1->SetLogy(0);
        for (auto& ah : angleHists) {
            ah.second->Draw("HIST");
            SaveCanvasWithPrefix(c1, "plots/png/" + locationName, "angle_" + ah.first, prefix);
        }
        
        c1->SetRightMargin(0.15);
        for (auto& ch : corrHists) {
            ch.second->Draw("COLZ");
            SaveCanvasWithPrefix(c1, "plots/png/" + locationName, "corr_" + ch.first, prefix);
        }
        delete c1;
    }
    
    WriteEvolutionHistograms(rootFile, fLocations, prefix);
    
    rootFile->Close();
    delete rootFile;
    
    G4cout << "\n=== Plot Generation Complete ===" << G4endl;
    G4cout << "ROOT file: plots/damsa_analysis.root" << G4endl;
    G4cout << "PNG plots: plots/png/ (subdirectories by location)" << G4endl;
    G4cout << "========================================\n" << G4endl;
}

void DamsaAnalysis::Reset()
{
    for (auto& pair : fLocations) {
        pair.second.Reset();
    }
}

G4int DamsaAnalysis::GetTargetExitPhotons() const
{
    return fLocations.at("TargetExit").GetParticleCount("gamma");
}

G4int DamsaAnalysis::GetTargetExitNeutrons() const
{
    return fLocations.at("TargetExit").GetParticleCount("neutron");
}

G4int DamsaAnalysis::GetCaloEntrancePhotons() const
{
    return fLocations.at("CaloEntrance").GetParticleCount("gamma");
}

G4int DamsaAnalysis::GetCaloEntranceNeutrons() const
{
    return fLocations.at("CaloEntrance").GetParticleCount("neutron");
}

G4int DamsaAnalysis::GetForwardPhotons() const
{
    const auto& locData = fLocations.at("CaloEntrance");
    const auto* pdata = locData.GetParticleDataMap().GetParticlePtr("gamma");
    if (!pdata) return 0;
    
    G4int count = 0;
    for (G4double angle : pdata->angles) {
        if (angle <= 20.0 * deg) count++;  // 0-20 degrees
    }
    return count;
}

G4double DamsaAnalysis::GetEfficiency() const
{
    G4int exitPhotons = GetTargetExitPhotons();
    G4int detPhotons = GetCaloEntrancePhotons();
    if (exitPhotons > 0) {
        return 100.0 * detPhotons / exitPhotons;
    }
    return 0.0;
}

TCanvas* CreateOverlayCanvasMap(std::map<G4String, TH1D*>& energyHists,
                               std::map<G4String, TH1D*>& angleHists,
                               const G4String& locationName)
{
    if (energyHists.empty()) return nullptr;
    
    TCanvas* cOverlay = new TCanvas("cOverlay", "", 1600, 600);
    cOverlay->Divide(2, 1);
    
    cOverlay->cd(1);
    
    double maxEnergyCount = 0;
    for (auto& eh : energyHists) {
        if (eh.second->GetMaximum() > maxEnergyCount) {
            maxEnergyCount = eh.second->GetMaximum();
        }
    }
    
    gPad->SetLogy();
    bool firstEnergy = true;
    TLegend* legE = new TLegend(0.65, 0.70, 0.94, 0.91);
    legE->SetBorderSize(0);
    legE->SetFillStyle(0);

    for (auto& eh : energyHists) {
        eh.second->SetMaximum(maxEnergyCount * 5);
        eh.second->SetMinimum(0.5);
        if (firstEnergy) {
            eh.second->SetTitle((locationName + " - Energy").c_str());
            eh.second->Draw("HIST");
            firstEnergy = false;
        } else {
            eh.second->Draw("HIST SAME");
        }
        legE->AddEntry(eh.second, GetParticleName(eh.first).c_str(), "l");
    }
    legE->Draw();
    
    cOverlay->cd(2);
    gPad->SetLogy();

    double maxAngleCount = 0;
    for (auto& ah : angleHists) {
        if (ah.second->GetMaximum() > maxAngleCount) {
            maxAngleCount = ah.second->GetMaximum();
        }
    }

    bool firstAngle = true;
    TLegend* legA = new TLegend(0.65, 0.70, 0.94, 0.91);
    legA->SetBorderSize(0);
    legA->SetFillStyle(0);

    for (auto& ah : angleHists) {
        ah.second->SetMaximum(maxAngleCount * 5);
        ah.second->SetMinimum(0.5);
        if (firstAngle) {
            ah.second->SetTitle((locationName + " - Angular").c_str());
            ah.second->Draw("HIST");
            firstAngle = false;
        } else {
            ah.second->Draw("HIST SAME");
        }
        legA->AddEntry(ah.second, GetParticleName(ah.first).c_str(), "l");
    }
    legA->Draw();
    
    return cOverlay;
}

void WriteEvolutionHistograms(TFile* rootFile, std::map<G4String, DamsaLocationData>& locations,
                              const std::string& prefix)
{
    rootFile->mkdir("Evolution");
    rootFile->mkdir("Statistics");
    rootFile->mkdir("Comparison");
    
    rootFile->cd("Evolution");
    
    std::vector<G4String> orderedLocations;
    orderedLocations.push_back("TargetExit");
    orderedLocations.push_back("MagnetEntrance");
    orderedLocations.push_back("CaloEntrance");
    orderedLocations.push_back("CaloExit");
    
    std::map<G4String, Color_t> locationColors;
    locationColors["TargetExit"] = kRed;
    locationColors["MagnetEntrance"] = kBlue;
    locationColors["CaloEntrance"] = kGreen+2;
    locationColors["CaloExit"] = kMagenta;
    
    G4int nLocs = orderedLocations.size();
    
    std::set<G4String> allParticles;
    for (const auto& locPair : locations) {
        for (const auto& partPair : locPair.second.GetParticleDataMap().GetMap()) {
            allParticles.insert(partPair.first);
        }
    }
    
    std::map<G4String, std::vector<G4int>> particleCounts;
    std::map<G4String, std::vector<double>> particleMeanEnergies;
    
    for (const G4String& particleName : allParticles) {
        particleCounts[particleName].resize(nLocs, 0);
        particleMeanEnergies[particleName].resize(nLocs, 0.0);
        
        for (G4int i = 0; i < nLocs; i++) {
            const DamsaLocationData& locData = locations[orderedLocations[i]];
            const ParticleData* pdata = locData.GetParticleDataMap().GetParticlePtr(particleName);
            
            if (pdata) {
                particleCounts[particleName][i] = pdata->count;
                
                if (!pdata->energies.empty()) {
                    double sumE = 0;
                    for (double e : pdata->energies) {
                        sumE += e / GeV;
                    }
                    particleMeanEnergies[particleName][i] = sumE / pdata->energies.size();
                }
            }
        }
    }
    
    Color_t particleColors[] = {kRed, kBlue, kGreen+2, kMagenta, kCyan, kOrange};
    G4int colorIdx = 0;
    
    for (const auto& pc : particleCounts) {
        const G4String& particleName = pc.first;
        const std::vector<G4int>& counts = pc.second;
        
        TH1D* hEvolution = new TH1D(("h" + particleName + "Evolution").c_str(),
            ("Count Evolution - " + particleName).c_str(),
            nLocs, 0, nLocs);
        
        for (G4int i = 0; i < nLocs; i++) {
            hEvolution->SetBinContent(i + 1, counts[i]);
            hEvolution->GetXaxis()->SetBinLabel(i + 1, orderedLocations[i].c_str());
        }
        
        hEvolution->SetLineColor(particleColors[colorIdx % 6]);
        hEvolution->SetLineWidth(2);
        hEvolution->SetStats(0);
        hEvolution->Write();
        
        colorIdx++;
    }
    
    rootFile->cd("Statistics");
    
    double xPos[4] = {1, 2, 3, 4};
    double xErr[4] = {0, 0, 0, 0};
    
    colorIdx = 0;
    for (const auto& pme : particleMeanEnergies) {
        const G4String& particleName = pme.first;
        const std::vector<double>& means = pme.second;
        
        TGraphErrors* gr = new TGraphErrors(nLocs, xPos, means.data(), xErr, xErr);
        gr->SetName(("gr" + particleName + "MeanEnergy").c_str());
        gr->SetTitle(("Mean Energy Evolution - " + particleName).c_str());
        gr->SetMarkerStyle(20 + colorIdx);
        gr->SetMarkerColor(particleColors[colorIdx % 6]);
        gr->SetLineColor(particleColors[colorIdx % 6]);
        gr->SetLineWidth(2);
        gr->Write();
        
        colorIdx++;
    }
    
    rootFile->cd("Comparison");
    
    const std::vector<G4String> trackParticles = {"gamma", "proton", "neutron", "e-"};
    Color_t partColors[] = {kRed, kBlue, kGreen+2, kMagenta, kCyan, kOrange, kPink, kTeal};
    
    // Max energy (MeV) across all tracked particle types at a given location.
    // Used for overlays where all species share one x-axis.
    auto GetMaxEnergyMeV = [&](G4int locIdx) -> double {
        const DamsaLocationData& locData = locations[orderedLocations[locIdx]];
        double maxE = 0;
        for (const G4String& ptype : trackParticles) {
            const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
            if (pd) {
                for (double e : pd->energies) {
                    double eMeV = e / MeV;
                    if (eMeV > maxE) maxE = eMeV;
                }
            }
        }
        return SnapEnergyMax(maxE * 1.2);
    };

    // Per-particle energy max — for pads that show only one species.
    auto GetParticleMaxEnergyMeV = [&](G4int locIdx, const G4String& ptype) -> double {
        const DamsaLocationData& locData = locations[orderedLocations[locIdx]];
        const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
        if (!pd || pd->energies.empty()) return 10.0;
        double maxE = *std::max_element(pd->energies.begin(), pd->energies.end()) / MeV;
        return SnapEnergyMax(maxE * 1.2);
    };
    
    TCanvas* cCountEnergy = new TCanvas("cCountEnergy", "Count vs Energy by Location", 1600, 1200);
    cCountEnergy->Divide(2, 2);
    
    for (G4int l = 0; l < 4; l++) {
        cCountEnergy->cd(l + 1);
        
        const DamsaLocationData& locData = locations[orderedLocations[l]];
        double maxE = GetMaxEnergyMeV(l);
        
        bool first = true;
        G4int colorIdx = 0;
        double maxCount = 0;
        std::vector<TH1D*> hists;
        
        for (const G4String& ptype : trackParticles) {
            const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
            if (!pd || pd->count == 0) continue;
            
            std::vector<double> energies;
            for (double e : pd->energies) {
                energies.push_back(e / MeV);
            }
            
            TH1D* h = new TH1D(Form("hCE_%s_%s", ptype.c_str(), orderedLocations[l].c_str()),
                orderedLocations[l].c_str(),
                50, 0, maxE);
            for (double e : energies) {
                h->Fill(e);
            }
            h->SetLineColor(partColors[colorIdx]);
            h->SetLineWidth(2);
            h->SetStats(0);
            h->GetXaxis()->SetTitle("Energy [MeV]");
            h->GetYaxis()->SetTitle("Counts");
            
            if (h->GetMaximum() > maxCount) maxCount = h->GetMaximum();
            hists.push_back(h);
            colorIdx++;
        }
        
        for (size_t i = 0; i < hists.size(); i++) {
            hists[i]->SetMaximum(maxCount * 1.2);
            if (i == 0) {
                hists[i]->Draw("HIST");
            } else {
                hists[i]->Draw("HIST SAME");
            }
        }
        
        if (!hists.empty()) {
            TLatex* lat = new TLatex();
            lat->SetNDC();
            lat->SetTextSize(0.025);
            G4double yPos = 0.88;
            G4int legIdx = 0;
            for (const G4String& ptype : trackParticles) {
                const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
                if (pd && pd->count > 0) {
                    lat->SetTextColor(partColors[legIdx % 8]);
                    lat->DrawLatex(0.60, yPos, GetParticleName(ptype).c_str());
                    yPos -= 0.04;
                    legIdx++;
                }
            }
        }
    }
    cCountEnergy->Write();
    SaveCanvasWithPrefix(cCountEnergy, "plots/png/summary", "count_energy_grid", prefix);
    delete cCountEnergy;
    
    TCanvas* cCountAngle = new TCanvas("cCountAngle", "Count vs Angle by Location", 1600, 1200);
    cCountAngle->Divide(2, 2);
    
    for (G4int l = 0; l < 4; l++) {
        cCountAngle->cd(l + 1);
        
        const DamsaLocationData& locData = locations[orderedLocations[l]];
        
        G4int colorIdx = 0;
        double maxCount = 0;
        std::vector<TH1D*> hists;
        
        for (const G4String& ptype : trackParticles) {
            const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
            if (!pd || pd->count == 0) continue;
            
            std::vector<double> angles;
            for (double a : pd->angles) {
                angles.push_back(a * 180.0 / 3.14159265);
            }
            
            TH1D* h = new TH1D(Form("hCA_%s_%s", ptype.c_str(), orderedLocations[l].c_str()),
                orderedLocations[l].c_str(),
                90, 0, 90);
            for (double a : angles) {
                h->Fill(a);
            }
            h->SetLineColor(partColors[colorIdx]);
            h->SetLineWidth(2);
            h->SetStats(0);
            h->GetXaxis()->SetTitle("Angle [degrees]");
            h->GetYaxis()->SetTitle("Counts");
            
            if (h->GetMaximum() > maxCount) maxCount = h->GetMaximum();
            hists.push_back(h);
            colorIdx++;
        }
        
        for (size_t i = 0; i < hists.size(); i++) {
            hists[i]->SetMaximum(maxCount * 1.2);
            if (i == 0) {
                hists[i]->Draw("HIST");
            } else {
                hists[i]->Draw("HIST SAME");
            }
        }
        
        if (!hists.empty()) {
            TLatex* lat = new TLatex();
            lat->SetNDC();
            lat->SetTextSize(0.025);
            G4double yPos = 0.88;
            G4int legIdx = 0;
            for (const G4String& ptype : trackParticles) {
                const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
                if (pd && pd->count > 0) {
                    lat->SetTextColor(partColors[legIdx % 8]);
                    lat->DrawLatex(0.60, yPos, GetParticleName(ptype).c_str());
                    yPos -= 0.04;
                    legIdx++;
                }
            }
        }
    }
    cCountAngle->Write();
    SaveCanvasWithPrefix(cCountAngle, "plots/png/summary", "count_angle_grid", prefix);
    delete cCountAngle;
    
    TCanvas* cEnergyAngleCount = new TCanvas("cEnergyAngleCount", "Energy-Count vs Angle by Location and Particle", 1600, 1200);
    cEnergyAngleCount->Divide(4, 4);
    
    for (G4int l = 0; l < 4; l++) {
        const DamsaLocationData& locData = locations[orderedLocations[l]];

        for (G4int p = 0; p < 4; p++) {
            const G4String& ptype = trackParticles[p];
            cEnergyAngleCount->cd(l * 4 + p + 1);

            const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
            // Per-particle range so a 8-GeV photon doesn't squash the neutron pad
            double maxE = GetParticleMaxEnergyMeV(l, ptype);

            TH2D* h2d = new TH2D(Form("h2d_%s_%s", ptype.c_str(), orderedLocations[l].c_str()),
                (GetParticleName(ptype) + " @ " + orderedLocations[l]).c_str(),
                45, 0, 90, 50, 0, maxE);
            
            if (pd && pd->count > 0) {
                for (size_t i = 0; i < pd->energies.size() && i < pd->angles.size(); i++) {
                    double eMeV = pd->energies[i] / MeV;
                    double angleDeg = pd->angles[i] * 180.0 / 3.14159265;
                    h2d->Fill(angleDeg, eMeV);
                }
            }
            
            h2d->SetStats(0);
            h2d->GetXaxis()->SetTitle("Angle [degrees]");
            h2d->GetYaxis()->SetTitle("Energy [MeV]");
            h2d->GetZaxis()->SetTitle("Counts");
            h2d->GetZaxis()->SetTitleOffset(0.8);
            h2d->Draw("COLZ");
        }
    }
    cEnergyAngleCount->Write();
    SaveCanvasWithPrefix(cEnergyAngleCount, "plots/png/summary", "energy_angle_count_grid", prefix);
    delete cEnergyAngleCount;
}

// ─── pi0 -> gamma+gamma study ────────────────────────────────────────────────

void WritePi0ROOTHistograms(const G4String& filename, const std::string& prefix)
{
    const auto& decays = DamsaPi0Collector::Instance()->GetDecays();
    if (decays.empty()) {
        G4cout << "No pi0->gg decays recorded; skipping pi0 histogram output." << G4endl;
        return;
    }

    SetPublicationStyle();
    mkdir("plots/png/pi0", 0755);

    std::string rootPath = "plots/" + prefix + filename;
    TFile* f = new TFile(rootPath.c_str(), "RECREATE");
    if (!f || f->IsZombie()) {
        G4cout << "ERROR: Cannot create ROOT file " << rootPath << G4endl;
        return;
    }

    // ── collect data ────────────────────────────────────────────────────────
    std::vector<double> vz, vx, vy, vr;
    std::vector<double> eAll, e1vec, e2vec, pi0Evec;
    std::vector<double> openAngle, gammaAngle;
    std::vector<double> eDouble, eSingle;          // scoring-plane (physical)
    std::vector<double> eGeomDouble, eGeomSingle;  // geometric acceptance (no material)
    std::vector<double> caloEnergyVec;             // total calo energy per pi0 decay
    std::vector<int>    caloHits;
    std::vector<int>    geomHits;

    for (const auto& d : decays) {
        double xMm = d.vx / mm;
        double yMm = d.vy / mm;
        double zMm = d.vz / mm;
        double e1  = d.e1 / MeV;
        double e2  = d.e2 / MeV;

        G4ThreeVector m1(d.px1, d.py1, d.pz1);
        G4ThreeVector m2(d.px2, d.py2, d.pz2);
        G4ThreeVector beam(0, 0, 1);
        double a1 = m1.angle(beam) * 180.0 / M_PI;
        double a2 = m2.angle(beam) * 180.0 / M_PI;

        vx.push_back(xMm);
        vy.push_back(yMm);
        vz.push_back(zMm);
        vr.push_back(std::sqrt(xMm*xMm + yMm*yMm));
        eAll.push_back(e1); eAll.push_back(e2);
        e1vec.push_back(e1); e2vec.push_back(e2);
        pi0Evec.push_back(e1 + e2);
        openAngle.push_back(d.openingAngle * 180.0 / M_PI);
        gammaAngle.push_back(a1); gammaAngle.push_back(a2);
        caloEnergyVec.push_back(d.caloEnergyMeV);

        G4int hits  = (d.gamma1AtCalo     ? 1 : 0) + (d.gamma2AtCalo     ? 1 : 0);
        G4int ghits = (d.gamma1GeomAccept ? 1 : 0) + (d.gamma2GeomAccept ? 1 : 0);
        caloHits.push_back(hits);
        geomHits.push_back(ghits);

        // Scoring-plane categories
        if (hits == 2) {
            eDouble.push_back(e1);
            eDouble.push_back(e2);
        } else if (hits == 1) {
            if (d.gamma1AtCalo) eSingle.push_back(e1);
            if (d.gamma2AtCalo) eSingle.push_back(e2);
        }
        // Geometric-acceptance categories
        if (ghits == 2) {
            eGeomDouble.push_back(e1);
            eGeomDouble.push_back(e2);
        } else if (ghits == 1) {
            if (d.gamma1GeomAccept) eGeomSingle.push_back(e1);
            if (d.gamma2GeomAccept) eGeomSingle.push_back(e2);
        }
    }

    // ── energy axis ranges (auto-ranged, no hardcoded 8000 MeV) ─────────────
    double eMax     = eAll.empty()     ? 10.0 : SnapEnergyMax(*std::max_element(eAll.begin(),     eAll.end())     * 1.2);
    double pi0EMax  = pi0Evec.empty()  ? 10.0 : SnapEnergyMax(*std::max_element(pi0Evec.begin(),  pi0Evec.end())  * 1.2);
    double eDoubleMax = eDouble.empty() ? eMax : SnapEnergyMax(*std::max_element(eDouble.begin(), eDouble.end())  * 1.2);
    double eSingleMax = eSingle.empty() ? eMax : SnapEnergyMax(*std::max_element(eSingle.begin(), eSingle.end())  * 1.2);
    double eOverlayMax = SnapEnergyMax(std::max(eDoubleMax, eSingleMax));

    // ── Pi0Decays directory ──────────────────────────────────────────────────
    f->mkdir("Pi0Decays");
    f->cd("Pi0Decays");

    // vertex z — auto-range with padding
    double zMin = *std::min_element(vz.begin(), vz.end());
    double zMax = *std::max_element(vz.begin(), vz.end());
    double zPad = std::max((zMax - zMin) * 0.1, 1.0);
    TH1D* hVz = new TH1D("hDecayVertexZ",
        "#pi^{0} Decay Vertex Z (lab frame);z [mm];Counts",
        200, zMin - zPad, zMax + zPad);
    for (double z : vz) hVz->Fill(z);
    StyleHist1D(hVz, kBlue, "z [mm]", "Counts");
    hVz->Write();

    double rMax = vr.empty() ? 50.0 : SnapEnergyMax(*std::max_element(vr.begin(), vr.end()) * 1.2);
    TH1D* hVr = new TH1D("hDecayVertexR",
        "#pi^{0} Decay Vertex Radial (lab frame);r = #sqrt{x^{2}+y^{2}} [mm];Counts",
        100, 0, rMax);
    for (double r : vr) hVr->Fill(r);
    StyleHist1D(hVr, kBlue, "r [mm]", "Counts");
    hVr->Write();

    double xyAbsMax = 0;
    for (size_t i = 0; i < vx.size(); i++)
        xyAbsMax = std::max(xyAbsMax, std::max(std::abs(vx[i]), std::abs(vy[i])));
    xyAbsMax = std::max(xyAbsMax * 1.2, 5.0);
    TH2D* hVxy = new TH2D("hDecayVertexXY",
        "#pi^{0} Decay Vertex XY (lab frame);x [mm];y [mm]",
        100, -xyAbsMax, xyAbsMax, 100, -xyAbsMax, xyAbsMax);
    for (size_t i = 0; i < vx.size(); i++) hVxy->Fill(vx[i], vy[i]);
    StyleHist2D(hVxy, "x [mm]", "y [mm]", "Counts");
    hVxy->Write();

    TH1D* hOpenAngle = new TH1D("hOpeningAngle",
        "#pi^{0}#rightarrow#gamma#gamma Opening Angle (lab frame);#theta_{#gamma#gamma} [deg];Counts",
        90, 0, 180);
    for (double a : openAngle) hOpenAngle->Fill(a);
    StyleHist1D(hOpenAngle, kRed, "#theta_{#gamma#gamma} [deg]", "Counts");
    hOpenAngle->Write();

    TH1D* hGammaAngle = new TH1D("hGammaAngle",
        "Daughter #gamma Angle w.r.t. Beam (lab frame);#theta [deg];Counts",
        90, 0, 90);
    for (double a : gammaAngle) hGammaAngle->Fill(a);
    StyleHist1D(hGammaAngle, kRed, "#theta [deg]", "Counts");
    hGammaAngle->Write();

    TH1D* hGammaE = new TH1D("hGammaEnergy",
        "Daughter #gamma Energy (both, lab frame);E [MeV];Counts",
        100, 0, eMax);
    for (double e : eAll) hGammaE->Fill(e);
    StyleHist1D(hGammaE, kRed, "E [MeV]", "Counts");
    hGammaE->Write();

    TH1D* hPi0E = new TH1D("hPi0Energy",
        "#pi^{0} Total Energy (lab frame);E [MeV];Counts",
        100, 0, pi0EMax);
    for (double e : pi0Evec) hPi0E->Fill(e);
    StyleHist1D(hPi0E, kBlue, "E [MeV]", "Counts");
    hPi0E->Write();

    TH2D* hE1E2 = new TH2D("hGammaE1vsE2",
        "Daughter #gamma Energies (lab frame);E_{1} [MeV];E_{2} [MeV]",
        50, 0, eMax, 50, 0, eMax);
    for (size_t i = 0; i < e1vec.size(); i++) hE1E2->Fill(e1vec[i], e2vec[i]);
    StyleHist2D(hE1E2, "E_{1} [MeV]", "E_{2} [MeV]", "Counts");
    hE1E2->Write();

    TH2D* hEvsA = new TH2D("hGammaEnergyVsAngle",
        "Daughter #gamma E vs Angle (lab frame);#theta [deg];E [MeV]",
        45, 0, 90, 50, 0, eMax);
    for (size_t i = 0; i < gammaAngle.size(); i++)
        hEvsA->Fill(gammaAngle[i], eAll[i]);
    StyleHist2D(hEvsA, "#theta [deg]", "E [MeV]", "Counts");
    hEvsA->Write();

    TH1D* hCaloN = new TH1D("hCaloPhotonsPerDecay",
        "#gamma Reaching Calo per #pi^{0} Decay (physical, with material);N_{#gamma} at calo scoring plane;Counts",
        3, -0.5, 2.5);
    for (int h : caloHits) hCaloN->Fill(h);
    StyleHist1D(hCaloN, kGreen+2, "N_{#gamma}", "Counts");
    hCaloN->Write();

    TH1D* hGeomN = new TH1D("hGeomPhotonsPerDecay",
        "#gamma Geometrically Accepted per #pi^{0} Decay (no material effects);N_{#gamma} projecting to calo aperture;Counts",
        3, -0.5, 2.5);
    for (int h : geomHits) hGeomN->Fill(h);
    StyleHist1D(hGeomN, kCyan+1, "N_{#gamma}", "Counts");
    hGeomN->Write();

    double caloEMax = caloEnergyVec.empty() ? 10.0
        : SnapEnergyMax(*std::max_element(caloEnergyVec.begin(), caloEnergyVec.end()) * 1.2);
    if (caloEMax < 1.0) caloEMax = 10.0;
    TH1D* hCaloE = new TH1D("hPi0CaloEnergy",
        "#pi^{0} Total Calo Energy Deposit (all descendants);E_{calo} [MeV];Counts",
        100, 0, caloEMax);
    for (double e : caloEnergyVec) if (e > 0) hCaloE->Fill(e);
    StyleHist1D(hCaloE, kMagenta, "E_{calo} [MeV]", "Counts");
    hCaloE->Write();

    // ── 3D vertex + track plot ───────────────────────────────────────────────
    // Vertices as TPolyMarker3D; photon directions as TPolyLine3D segments
    // (capped at 2000 decays to keep file manageable).
    // Track length = 200 mm (shows direction; not actual photon endpoint).
    // Color: red = gamma reached calo, kGray+1 = did not reach calo.
    {
        const int maxPlot = std::min((int)decays.size(), 2000);
        const double trackLen = 200.0; // mm

        TCanvas* c3d = new TCanvas("cPi0Tracks3D",
            "#pi^{0} Decay Vertices and Photon Directions (lab frame, first 2000 decays)",
            1000, 900);
        c3d->cd();

        // Draw all decay vertices as a 3D scatter to establish the TView.
        TPolyMarker3D* pmVtx = new TPolyMarker3D(vx.size());
        for (size_t i = 0; i < vx.size(); i++)
            pmVtx->SetPoint(i, vx[i], vy[i], vz[i]);
        pmVtx->SetMarkerStyle(6);   // tiny dot
        pmVtx->SetMarkerColor(kBlue);
        pmVtx->SetMarkerSize(1);
        pmVtx->Draw();              // establishes TView / coordinate frame

        // Per-decay track line segments (keep pointers alive until canvas saved)
        std::vector<TPolyLine3D*> trackLines;
        trackLines.reserve(maxPlot * 2);

        for (int i = 0; i < maxPlot; i++) {
            const Pi0Decay& d = decays[i];
            double ox = d.vx / mm, oy = d.vy / mm, oz = d.vz / mm;

            // gamma1: red = geom-accepted, dark-red = geom+physical, gray = missed
            TPolyLine3D* tl1 = new TPolyLine3D(2);
            tl1->SetPoint(0, ox, oy, oz);
            tl1->SetPoint(1, ox + trackLen*d.px1, oy + trackLen*d.py1, oz + trackLen*d.pz1);
            tl1->SetLineColor(d.gamma1GeomAccept ? (d.gamma1AtCalo ? kRed+2 : kRed) : kGray+1);
            tl1->SetLineWidth(1);
            tl1->Draw();
            trackLines.push_back(tl1);

            // gamma2: orange = geom-accepted, dark-orange = geom+physical, gray = missed
            TPolyLine3D* tl2 = new TPolyLine3D(2);
            tl2->SetPoint(0, ox, oy, oz);
            tl2->SetPoint(1, ox + trackLen*d.px2, oy + trackLen*d.py2, oz + trackLen*d.pz2);
            tl2->SetLineColor(d.gamma2GeomAccept ? (d.gamma2AtCalo ? kOrange+2 : kOrange+7) : kGray+1);
            tl2->SetLineWidth(1);
            tl2->Draw();
            trackLines.push_back(tl2);
        }

        // Axis labels via TLatex (3D axes are labeled by the TView automatically)
        TLatex lat3d;
        lat3d.SetNDC();
        lat3d.SetTextSize(0.025);
        lat3d.DrawLatex(0.02, 0.95, "#pi^{0}#rightarrow#gamma#gamma: blue=vertex, bright=geom-accepted, dark=geom+physical, gray=missed calo aperture");
        lat3d.DrawLatex(0.02, 0.02, "Axes: x, y, z [mm] (lab frame). Track segments = 200 mm extensions in momentum direction.");

        c3d->Write();
        SaveCanvasWithPrefix(c3d, "plots/png/pi0", "decay_vertices_3d_tracks", prefix);

        for (auto* tl : trackLines) delete tl;
        delete pmVtx;
        delete c3d;
    }

    // ── GeomCategories directory (geometric acceptance, no material effects) ────
    {
        double eGDMax = eGeomDouble.empty() ? eMax : SnapEnergyMax(*std::max_element(eGeomDouble.begin(), eGeomDouble.end()) * 1.2);
        double eGSMax = eGeomSingle.empty() ? eMax : SnapEnergyMax(*std::max_element(eGeomSingle.begin(), eGeomSingle.end()) * 1.2);
        double eGOvlMax = SnapEnergyMax(std::max(eGDMax, eGSMax));

        f->mkdir("GeomCategories");
        f->cd("GeomCategories");

        TH1D* hGD = new TH1D("hGeomDoubleGammaEnergy",
            "#pi^{0} Geom-Accepted Double-#gamma: Daughter Energy;E [MeV];Counts",
            100, 0, eGDMax);
        for (double e : eGeomDouble) hGD->Fill(e);
        StyleHist1D(hGD, kRed, "E [MeV]", "Counts");
        hGD->Write();

        TH1D* hGS = new TH1D("hGeomSingleGammaEnergy",
            "#pi^{0} Geom-Accepted Single-#gamma: Daughter Energy;E [MeV];Counts",
            100, 0, eGSMax);
        for (double e : eGeomSingle) hGS->Fill(e);
        StyleHist1D(hGS, kBlue, "E [MeV]", "Counts");
        hGS->Write();

        TH1D* hGDO = new TH1D("hGeomDoubleOvl", "#pi^{0} Geom Double-#gamma;E [MeV];Counts", 100, 0, eGOvlMax);
        TH1D* hGSO = new TH1D("hGeomSingleOvl", "#pi^{0} Geom Single-#gamma (excl.);E [MeV];Counts", 100, 0, eGOvlMax);
        for (double e : eGeomDouble) hGDO->Fill(e);
        for (double e : eGeomSingle) hGSO->Fill(e);
        StyleHist1D(hGDO, kRed,  "E [MeV]", "Counts");
        StyleHist1D(hGSO, kBlue, "E [MeV]", "Counts");

        TCanvas* cGeomOvl = new TCanvas("cGeomOverlay",
            "#pi^{0} Geom-Accepted #gamma at Calorimeter (no material effects)", 900, 600);
        double gmaxC = std::max(hGDO->GetMaximum(), hGSO->GetMaximum());
        if (gmaxC <= 0) gmaxC = 1.0;
        hGDO->SetMaximum(gmaxC * 1.35); hGDO->SetMinimum(0);
        hGDO->SetTitle("#pi^{0}#rightarrow#gamma#gamma Geometric Acceptance (lab frame)");
        hGDO->Draw("HIST");
        hGSO->Draw("HIST SAME");
        TLegend* legG = new TLegend(0.50, 0.72, 0.84, 0.88);
        legG->SetBorderSize(0); legG->SetFillStyle(0);
        legG->AddEntry(hGDO, "Both #gamma geom-accepted (double)", "l");
        legG->AddEntry(hGSO, "One #gamma geom-accepted (single, excl.)", "l");
        legG->Draw();
        cGeomOvl->Write();
        SaveCanvasWithPrefix(cGeomOvl, "plots/png/pi0", "overlay_geom_photons", prefix);
        delete cGeomOvl;
    }

    // ── CaloCategories directory ─────────────────────────────────────────────
    f->mkdir("CaloCategories");
    f->cd("CaloCategories");

    TH1D* hDoubleE = new TH1D("hDoubleGammaEnergy",
        "#pi^{0} Double-#gamma at Calo: Daughter Energy (lab frame);E [MeV];Counts",
        100, 0, eDoubleMax);
    for (double e : eDouble) hDoubleE->Fill(e);
    StyleHist1D(hDoubleE, kRed, "E [MeV]", "Counts");
    hDoubleE->Write();

    TH1D* hSingleE = new TH1D("hSingleGammaEnergy",
        "#pi^{0} Single-#gamma at Calo (excl.): Daughter Energy (lab frame);E [MeV];Counts",
        100, 0, eSingleMax);
    for (double e : eSingle) hSingleE->Fill(e);
    StyleHist1D(hSingleE, kBlue, "E [MeV]", "Counts");
    hSingleE->Write();

    // overlay: double-gamma vs single-gamma (exclusive) — common x-axis
    // Recreate on common range for a fair visual comparison
    TH1D* hDoubleEOvl = new TH1D("hDoubleGammaEnergyOvl",
        "#pi^{0} Double-#gamma;E [MeV];Counts", 100, 0, eOverlayMax);
    TH1D* hSingleEOvl = new TH1D("hSingleGammaEnergyOvl",
        "#pi^{0} Single-#gamma (excl.);E [MeV];Counts", 100, 0, eOverlayMax);
    for (double e : eDouble) hDoubleEOvl->Fill(e);
    for (double e : eSingle) hSingleEOvl->Fill(e);
    StyleHist1D(hDoubleEOvl, kRed,  "E [MeV]", "Counts");
    StyleHist1D(hSingleEOvl, kBlue, "E [MeV]", "Counts");

    TCanvas* cOverlay = new TCanvas("cOverlay",
        "#pi^{0} Decay #gamma at Calorimeter Entrance", 900, 600);
    double maxC = std::max(hDoubleEOvl->GetMaximum(), hSingleEOvl->GetMaximum());
    if (maxC <= 0) maxC = 1.0;
    hDoubleEOvl->SetMaximum(maxC * 1.35);
    hDoubleEOvl->SetMinimum(0);
    hDoubleEOvl->SetTitle("#pi^{0}#rightarrow#gamma#gamma at Calorimeter (lab frame)");
    hDoubleEOvl->Draw("HIST");
    hSingleEOvl->Draw("HIST SAME");
    TLegend* leg = new TLegend(0.50, 0.72, 0.84, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->AddEntry(hDoubleEOvl, "Both #gamma reach calo (double-hit)", "l");
    leg->AddEntry(hSingleEOvl, "One #gamma reaches calo (single, excl.)", "l");
    leg->Draw();
    cOverlay->Write();
    SaveCanvasWithPrefix(cOverlay, "plots/png/pi0", "overlay_calo_photons", prefix);
    delete cOverlay;

    // ── individual PNG saves ─────────────────────────────────────────────────
    TCanvas* c = new TCanvas("cPi0Tmp", "", 900, 600);

    c->SetRightMargin(0.20);
    hVz->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "decay_vertex_z", prefix);

    hVr->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "decay_vertex_r", prefix);

    c->SetRightMargin(0.15);
    hVxy->Draw("COLZ");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "decay_vertex_xy", prefix);

    c->SetRightMargin(0.20);
    hOpenAngle->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "opening_angle", prefix);

    hGammaAngle->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "gamma_beam_angle", prefix);

    hGammaE->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "gamma_energy", prefix);

    hPi0E->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "pi0_energy", prefix);

    hCaloN->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "calo_hits_per_decay", prefix);

    hGeomN->Draw("HIST");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "geom_accept_per_decay", prefix);

    if (hCaloE->GetEntries() > 0) {
        hCaloE->Draw("HIST");
        SaveCanvasWithPrefix(c, "plots/png/pi0", "pi0_calo_energy", prefix);
    }

    c->SetRightMargin(0.15);
    hE1E2->Draw("COLZ");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "gamma_e1_vs_e2", prefix);

    hEvsA->Draw("COLZ");
    SaveCanvasWithPrefix(c, "plots/png/pi0", "gamma_energy_vs_angle", prefix);

    delete c;

    f->Close();
    delete f;

    G4cout << "\n=== pi0 Analysis Output ===" << G4endl;
    G4cout << "ROOT file:  " << rootPath << G4endl;
    G4cout << "PNG plots:  plots/png/pi0/" << G4endl;
    G4cout << "Total pi0 produced:           " << DamsaPi0Collector::Instance()->GetTotalPi0Produced() << G4endl;
    G4cout << "Total pi0->gg decays tracked: " << decays.size() << G4endl;
}
