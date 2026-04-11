#ifndef PARTICLEDATA_H
#define PARTICLEDATA_H

#include "globals.hh"
#include <vector>
#include <map>

struct ParticleData {
    G4int count;
    std::vector<G4double> energies;
    std::vector<G4double> angles;
    
    ParticleData() : count(0) {}
    
    void Record(G4double energy, G4double angle) {
        count++;
        energies.push_back(energy);
        angles.push_back(angle);
    }
    
    void Reset() {
        count = 0;
        energies.clear();
        angles.clear();
    }
    
    void MergeFrom(const ParticleData& other) {
        count += other.count;
        energies.insert(energies.end(), other.energies.begin(), other.energies.end());
        angles.insert(angles.end(), other.angles.begin(), other.angles.end());
    }
};

class ParticleDataMap {
public:
    ParticleData& GetParticle(const G4String& particleName) {
        return fData[particleName];
    }
    
    const ParticleData* GetParticlePtr(const G4String& particleName) const {
        auto it = fData.find(particleName);
        if (it != fData.end()) {
            return &(it->second);
        }
        return nullptr;
    }
    
    G4int GetCount(const G4String& particleName) const {
        auto it = fData.find(particleName);
        return (it != fData.end()) ? it->second.count : 0;
    }
    
    G4double GetTotalEnergy(const G4String& particleName) const {
        auto it = fData.find(particleName);
        if (it == fData.end()) return 0.0;
        
        G4double total = 0.0;
        for (G4double e : it->second.energies) {
            total += e;
        }
        return total;
    }
    
    void Reset() {
        for (auto& pair : fData) {
            pair.second.Reset();
        }
    }
    
    std::map<G4String, ParticleData>& GetMap() { return fData; }
    const std::map<G4String, ParticleData>& GetMap() const { return fData; }
    
    void MergeFrom(const ParticleDataMap& other) {
        for (const auto& pair : other.fData) {
            fData[pair.first].MergeFrom(pair.second);
        }
    }
    
private:
    std::map<G4String, ParticleData> fData;
};

#endif
