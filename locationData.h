#ifndef LOCATIONDATA_H
#define LOCATIONDATA_H

#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "particleData.h"
#include <set>
#include <map>
#include <fstream>
#include <iomanip>

G4String GetParticleName(const G4String& particleName);
void PrintEnergySpectrum(const ParticleData& pdata, const G4String& particleName, 
                         const G4String& locationName);
void PrintAngularDistribution(const ParticleData& pdata, const G4String& particleName,
                               const G4String& locationName);

class DamsaLocationData {
public:
    DamsaLocationData();
    
    void RecordParticle(const G4String& particleName, G4double energy, 
                        G4double angle, G4int trackID, G4bool isPrimary);
    
    G4bool WasTrackRecorded(G4int trackID) const;
    void Reset();
    void ResetEventTracking();
    
    G4int GetParticleCount(const G4String& particleName) const;
    G4double GetTotalEnergy() const { return fTotalEnergy; }
    G4double GetPrimaryEnergy() const { return fPrimaryEnergy; }
    
    const ParticleDataMap& GetParticleDataMap() const { return fParticles; }
    ParticleDataMap& GetParticleDataMap() { return fParticles; }
    
    void PrintSummary(const G4String& locationName) const;
    void WriteToFile(std::ofstream& outFile, const G4String& locationName) const;
    
    // MT support: merge data from another location
    void MergeFrom(const DamsaLocationData& other);
    
private:
    ParticleDataMap fParticles;
    std::set<G4int> fTrackIDs;
    G4double fTotalEnergy;
    G4double fPrimaryEnergy;
};

inline DamsaLocationData::DamsaLocationData()
: fTotalEnergy(0), fPrimaryEnergy(0)
{}

inline void DamsaLocationData::RecordParticle(const G4String& particleName, G4double energy,
                                        G4double angle, G4int trackID, G4bool isPrimary)
{
    fTrackIDs.insert(trackID);
    fParticles.GetParticle(particleName).Record(energy, angle);
    
    if (isPrimary) {
        fPrimaryEnergy += energy;
    }
    fTotalEnergy += energy;
}

inline G4bool DamsaLocationData::WasTrackRecorded(G4int trackID) const
{
    return fTrackIDs.find(trackID) != fTrackIDs.end();
}

inline void DamsaLocationData::Reset()
{
    fParticles.Reset();
    fTrackIDs.clear();
    fTotalEnergy = 0;
    fPrimaryEnergy = 0;
}

inline void DamsaLocationData::ResetEventTracking()
{
    fTrackIDs.clear();
}

inline G4int DamsaLocationData::GetParticleCount(const G4String& particleName) const
{
    return fParticles.GetCount(particleName);
}

inline void DamsaLocationData::PrintSummary(const G4String& locationName) const
{
    G4cout << "\n=============================================" << G4endl;
    G4cout << "=== " << locationName << " ===" << G4endl;
    G4cout << "=============================================" << G4endl;
    
    for (const auto& pair : fParticles.GetMap()) {
        const G4String& particleName = pair.first;
        const ParticleData& pdata = pair.second;
        G4cout << GetParticleName(particleName) << ": " << pdata.count << G4endl;
    }
    
    G4cout << "Total Energy: " << fTotalEnergy/GeV << " GeV" << G4endl;
    G4cout << "  (Primary beam: " << fPrimaryEnergy/GeV << " GeV, Secondaries: " 
           << (fTotalEnergy - fPrimaryEnergy)/GeV << " GeV)" << G4endl;
    
    for (const auto& pair : fParticles.GetMap()) {
        const G4String& particleName = pair.first;
        const ParticleData& pdata = pair.second;
        
        if (pdata.energies.size() > 0) {
            PrintEnergySpectrum(pdata, particleName, locationName);
        }
        
        if (pdata.angles.size() > 0) {
            PrintAngularDistribution(pdata, particleName, locationName);
        }
    }
    
    G4cout << "=============================================\n" << G4endl;
}

inline G4String GetParticleName(const G4String& particleName)
{
    if (particleName == "neutron") return "Neutrons";
    if (particleName == "gamma") return "Photons";
    if (particleName == "e-") return "Electrons";
    if (particleName == "e+") return "Positrons";
    if (particleName == "proton") return "Protons";
    if (particleName == "pi+") return "Pions+";
    if (particleName == "pi-") return "Pions-";
    if (particleName == "kaon+") return "Kaons+";
    if (particleName == "kaon-") return "Kaons-";
    return particleName;
}

inline void PrintEnergySpectrum(const ParticleData& pdata, const G4String& particleName, 
                         const G4String& locationName)
{
    G4double tempBins[10];
    G4int nBins = 0;
    
    if (particleName == "gamma" || particleName == "e-" || particleName == "e+") {
        G4double bins[] = {0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0};
        nBins = 7;
        for (int i = 0; i <= nBins; i++) tempBins[i] = bins[i];
    } else if (particleName == "neutron") {
        G4double bins[] = {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0};
        nBins = 8;
        for (int i = 0; i <= nBins; i++) tempBins[i] = bins[i];
    } else {
        return;
    }
    
    G4int* counts = new G4int[nBins];
    for (G4int i = 0; i < nBins; i++) counts[i] = 0;
    
    for (size_t i = 0; i < pdata.energies.size(); i++) {
        G4double E = pdata.energies[i] / GeV;
        for (G4int j = 0; j < nBins; j++) {
            if (E >= tempBins[j] && E < tempBins[j+1]) {
                counts[j]++;
                break;
            }
        }
    }
    
    G4cout << "\n=== " << GetParticleName(particleName) << " Energy Spectrum at " 
           << locationName << " ===" << G4endl;
    for (G4int i = 0; i < nBins; i++) {
        G4cout << "  [" << tempBins[i] << " - " << tempBins[i+1] << " GeV]: "
               << counts[i] << " " << GetParticleName(particleName) << G4endl;
    }
    
    delete[] counts;
}

inline void PrintAngularDistribution(const ParticleData& pdata, const G4String& particleName,
                               const G4String& locationName)
{
    G4double angleBins[] = {0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0};
    G4int nAngleBins = 7;
    G4int angleCounts[7] = {0};
    
    for (size_t i = 0; i < pdata.angles.size(); i++) {
        G4double angleDeg = pdata.angles[i] * 180.0 / 3.14159265;
        for (G4int j = 0; j < nAngleBins; j++) {
            if (angleDeg >= angleBins[j] && angleDeg < angleBins[j+1]) {
                angleCounts[j]++;
                break;
            }
        }
    }
    
    G4cout << "\n=== " << GetParticleName(particleName) << " Angular Distribution at "
           << locationName << " ===" << G4endl;
    G4cout << "  (Angle relative to beam axis, 0 deg = straight forward)" << G4endl;
    for (G4int i = 0; i < nAngleBins; i++) {
        G4cout << "  [" << angleBins[i] << " - " << angleBins[i+1] << " degrees]: " 
               << angleCounts[i] << " " << GetParticleName(particleName) << G4endl;
    }
}

inline void DamsaLocationData::WriteToFile(std::ofstream& outFile, const G4String& locationName) const
{
    outFile << "=== " << locationName << " ===" << std::endl;
    
    for (const auto& pair : fParticles.GetMap()) {
        const G4String& particleName = pair.first;
        const ParticleData& pdata = pair.second;
        outFile << GetParticleName(particleName) << ":     " << pdata.count << std::endl;
    }
    
    outFile << "Total Energy: " << std::fixed << std::setprecision(6) 
            << fTotalEnergy/GeV << " GeV" << std::endl << std::endl;
}

inline void DamsaLocationData::MergeFrom(const DamsaLocationData& other)
{
    fTotalEnergy += other.fTotalEnergy;
    fPrimaryEnergy += other.fPrimaryEnergy;
    
    for (const auto& pair : other.fParticles.GetMap()) {
        fParticles.GetParticle(pair.first).MergeFrom(pair.second);
    }
    
    for (G4int tid : other.fTrackIDs) {
        fTrackIDs.insert(tid);
    }
}

#endif
