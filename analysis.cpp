#include "analysis.h"

TCanvas* CreateOverlayCanvasMap(std::map<G4String, TH1D*>& energyHists,
                               std::map<G4String, TH1D*>& angleHists,
                               const G4String& locationName);
void WriteEvolutionHistograms(TFile* rootFile, std::map<G4String, DamsaLocationData>& locations);

DamsaAnalysis* DamsaAnalysis::fInstance = nullptr;

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
    for (auto& pair : fLocations) {
        pair.second.ResetEventTracking();
    }
}

void DamsaAnalysis::RecordParticle(const G4String& particleName, G4double energy,
                                   const G4String& location, G4double angle, 
                                   G4int trackID, G4bool isPrimary)
{
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
    
    std::string rootPath = "plots/damsa_analysis.root";
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
            SaveCanvas(cOverlay, ("plots/png/" + locationName + "/overlay").c_str());
            delete cOverlay;
        }
        
        TCanvas* c1 = new TCanvas("c1", "", 800, 600);
        for (auto& eh : energyHists) {
            eh.second->Draw("HIST");
            SaveCanvas(c1, ("plots/png/" + locationName + "/energy_" + eh.first).c_str());
        }
        
        c1->SetLogy(0);
        for (auto& ah : angleHists) {
            ah.second->Draw("HIST");
            SaveCanvas(c1, ("plots/png/" + locationName + "/angle_" + ah.first).c_str());
        }
        
        c1->SetRightMargin(0.15);
        for (auto& ch : corrHists) {
            ch.second->Draw("COLZ");
            SaveCanvas(c1, ("plots/png/" + locationName + "/corr_" + ch.first).c_str());
        }
        delete c1;
    }
    
    WriteEvolutionHistograms(rootFile, fLocations);
    
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
    
    bool firstEnergy = true;
    TLegend* legE = new TLegend(0.65, 0.70, 0.94, 0.91);
    legE->SetBorderSize(0);
    legE->SetFillStyle(0);
    
    for (auto& eh : energyHists) {
        eh.second->SetMaximum(maxEnergyCount * 1.2);
        eh.second->SetMinimum(0);
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
        ah.second->SetMaximum(maxAngleCount * 1.2);
        ah.second->SetMinimum(0);
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

void WriteEvolutionHistograms(TFile* rootFile, std::map<G4String, DamsaLocationData>& locations)
{
    rootFile->mkdir("Evolution");
    rootFile->mkdir("Statistics");
    
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
    
    auto GetMaxEnergyMeV = [&](G4int locIdx) -> double {
        const DamsaLocationData& locData = locations[orderedLocations[locIdx]];
        double maxE = 1.0;
        for (const G4String& ptype : trackParticles) {
            const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
            if (pd) {
                for (double e : pd->energies) {
                    double eMeV = e / MeV;
                    if (eMeV > maxE) maxE = eMeV;
                }
            }
        }
        double range = maxE;
        maxE = maxE + range * 0.2;
        if (maxE < 10.0) maxE = 10.0;
        else if (maxE < 50.0) maxE = 50.0;
        else if (maxE < 100.0) maxE = 100.0;
        else if (maxE < 500.0) maxE = 500.0;
        else if (maxE < 1000.0) maxE = 1000.0;
        else maxE = 8000.0;
        return maxE;
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
    SaveCanvas(cCountEnergy, "plots/png/summary/count_energy_grid");
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
    SaveCanvas(cCountAngle, "plots/png/summary/count_angle_grid");
    delete cCountAngle;
    
    TCanvas* cEnergyAngleCount = new TCanvas("cEnergyAngleCount", "Energy-Count vs Angle by Location and Particle", 1600, 1200);
    cEnergyAngleCount->Divide(4, 4);
    
    for (G4int l = 0; l < 4; l++) {
        const DamsaLocationData& locData = locations[orderedLocations[l]];
        double maxE = GetMaxEnergyMeV(l);
        
        for (G4int p = 0; p < 4; p++) {
            const G4String& ptype = trackParticles[p];
            cEnergyAngleCount->cd(l * 4 + p + 1);
            
            const ParticleData* pd = locData.GetParticleDataMap().GetParticlePtr(ptype);
            
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
    SaveCanvas(cEnergyAngleCount, "plots/png/summary/energy_angle_count_grid");
    delete cEnergyAngleCount;
}
