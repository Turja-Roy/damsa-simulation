#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4AutoLock.hh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <vector>
#include <map>
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TPad.h"
#include "plotting.h"
#include "locationData.h"

// Free function: pi0->gg decay histograms + plots (implemented in analysis.cpp)
void WritePi0ROOTHistograms(const G4String& filename, const std::string& prefix = "");

class DamsaAnalysis
{
public:
    static DamsaAnalysis* Instance();
    
    void RecordParticle(const G4String& particleName, G4double energy, 
                       const G4String& location, G4double angle, G4int trackID, G4bool isPrimary);
    void PrintSummary();
    void SaveToFile(const G4String& filename);
    void WriteROOTHistograms(const G4String& filename);
    void Reset();
    void ResetEventTracking();
    
    G4int GetTargetExitPhotons() const;
    G4int GetTargetExitNeutrons() const;
    G4int GetCaloEntrancePhotons() const;
    G4int GetCaloEntranceNeutrons() const;
    G4int GetForwardPhotons() const;  // Photons within 0-20 degrees
    G4double GetEfficiency() const;
    
    G4bool WasTrackRecorded(G4int trackID, const G4String& location);
    
    // Config prefix for output filenames (e.g., "Tz10_xy5_G50_")
    void SetConfigPrefix(const std::string& prefix) { fConfigPrefix = prefix; }
    std::string GetConfigPrefix() const { return fConfigPrefix; }

private:
    DamsaAnalysis();
    ~DamsaAnalysis();
    static DamsaAnalysis* fInstance;
    static G4Mutex fMutex;

    std::map<G4String, DamsaLocationData> fLocations;
    std::string fConfigPrefix;  // Prefix for config-specific filenames
};

#endif
