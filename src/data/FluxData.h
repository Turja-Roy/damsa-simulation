#ifndef FLUXDATA_H
#define FLUXDATA_H

#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4AutoLock.hh"
#include <vector>
#include <fstream>
#include <iomanip>
#include <map>
#include <sys/stat.h>
#include "damsa_config.h"

// Structure to hold complete particle information for flux extraction
// Used primarily for photon flux output to alplib
struct FluxParticle {
    G4double energy;      // Kinetic energy (MeV)
    G4double time;        // Global time (ns)
    G4double x, y, z;     // Position (mm)
    G4double px, py, pz;  // Momentum direction (unit vector)
    G4double weight;      // Statistical weight (for biasing, default 1.0)
    G4int pdgCode;        // PDG particle code
    G4int trackID;        // Track ID for deduplication
    G4int eventID;        // Event ID for correlation
    
    FluxParticle() : energy(0), time(0), x(0), y(0), z(0),
                     px(0), py(0), pz(1), weight(1.0),
                     pdgCode(0), trackID(0), eventID(0) {}
};

// Class to collect and export photon flux data for alplib integration
class DamsaFluxCollector {
public:
    static DamsaFluxCollector* Instance();
    
    // Record a photon crossing the target exit plane
    void RecordPhoton(G4double energy, G4double time,
                      G4double x, G4double y, G4double z,
                      G4double px, G4double py, G4double pz,
                      G4int trackID, G4int eventID, G4double weight = 1.0);
    
    // Record a bremsstrahlung photon born INSIDE the target
    // These hard photons are the correct input for alplib Primakoff production
    void RecordBremsPhoton(G4double energy, G4double time,
                           G4double x, G4double y, G4double z,
                           G4double px, G4double py, G4double pz,
                           G4int trackID, G4int eventID, G4double weight = 1.0);
    
    // Record any particle (for background studies)
    void RecordParticle(G4int pdgCode, G4double energy, G4double time,
                        G4double x, G4double y, G4double z,
                        G4double px, G4double py, G4double pz,
                        G4int trackID, G4int eventID, G4double weight = 1.0);
    
    // Record any particle at the calorimeter entrance face
    void RecordCaloFaceParticle(G4int pdgCode, G4double energy, G4double time,
                                G4double x, G4double y, G4double z,
                                G4double px, G4double py, G4double pz,
                                G4int trackID, G4int eventID, G4double weight = 1.0);
    
    // Export functions
    void WriteCSV(const G4String& filename) const;
    void WritePhotonFluxCSV(const G4String& filename) const;
    void WriteBackgroundCSV(const G4String& filename) const;
    void WriteCaloFaceCSV(const G4String& filename) const;
    
    // Get binned photon spectrum for quick alplib input
    // Returns map of energy bin center (MeV) -> count
    std::map<G4double, G4int> GetBinnedPhotonSpectrum(G4double binWidth = 1.0*MeV) const;
    std::map<G4double, G4int> GetBinnedBremsSpectrum(G4double binWidth = 1.0*MeV) const;
    
    // Write alplib-compatible flux file (energy, rate format)
    // Exit photon version (for background reference)
    void WriteAlplibFlux(const G4String& filename, G4double nPrimaries,
                         G4double beamCurrent) const;
    // Bremsstrahlung version (correct input for Primakoff ALP production)
    void WriteAlplibBremsFlux(const G4String& filename, G4double nPrimaries,
                              G4double beamCurrent) const;
    
    // Statistics
    G4int GetPhotonCount() const { return fPhotons.size(); }
    G4long GetBremsPhotonCount() const { return fBremsCount; }
    G4int GetNeutronCount() const;
    G4int GetTotalParticleCount() const { return fAllParticles.size(); }
    G4int GetCaloFaceCount() const { return fCaloFaceParticles.size(); }
    
    // Clear data between runs
    void Reset();
    
    // Access raw data (for ROOT ntuple filling)
    const std::vector<FluxParticle>& GetPhotons() const { return fPhotons; }
    const std::vector<FluxParticle>& GetAllParticles() const { return fAllParticles; }
    const std::vector<FluxParticle>& GetCaloFaceParticles() const { return fCaloFaceParticles; }
    
private:
    DamsaFluxCollector();
    ~DamsaFluxCollector();
    static DamsaFluxCollector* fInstance;
    static G4Mutex fMutex;

    std::vector<FluxParticle> fPhotons;        // Photons at target exit
    // Bremsstrahlung photons inside the target are histogrammed at record
    // time rather than stored per photon: at 1e6 primary electrons the raw
    // vector required ~70 GB of memory (raw CSV ~145 GB), while all later
    // analysis stages read only the 1-MeV-binned spectrum written to
    // alplib_brems_flux.csv. Key = 1-MeV bin index, value = weighted count.
    std::map<G4int, G4long> fBremsSpectrum;
    G4long fBremsCount = 0;                    // total brems photons recorded
    std::vector<FluxParticle> fAllParticles;   // All particles (for background)
    std::vector<FluxParticle> fCaloFaceParticles;  // All particles at CaloEntrance
};

// Implementation

inline DamsaFluxCollector* DamsaFluxCollector::Instance()
{
    if (!fInstance) {
        fInstance = new DamsaFluxCollector();
    }
    return fInstance;
}

inline DamsaFluxCollector::DamsaFluxCollector() {}
inline DamsaFluxCollector::~DamsaFluxCollector() {}

inline void DamsaFluxCollector::RecordPhoton(G4double energy, G4double time,
                                              G4double x, G4double y, G4double z,
                                              G4double px, G4double py, G4double pz,
                                              G4int trackID, G4int eventID, G4double weight)
{
    FluxParticle p;
    p.energy = energy;
    p.time = time;
    p.x = x;
    p.y = y;
    p.z = z;
    p.px = px;
    p.py = py;
    p.pz = pz;
    p.weight = weight;
    p.pdgCode = 22;  // Photon PDG code
    p.trackID = trackID;
    p.eventID = eventID;
    G4AutoLock lock(&fMutex);
    fPhotons.push_back(p);
}

inline void DamsaFluxCollector::RecordParticle(G4int pdgCode, G4double energy, G4double time,
                                                G4double x, G4double y, G4double z,
                                                G4double px, G4double py, G4double pz,
                                                G4int trackID, G4int eventID, G4double weight)
{
    FluxParticle p;
    p.pdgCode = pdgCode;
    p.energy = energy;
    p.time = time;
    p.x = x;
    p.y = y;
    p.z = z;
    p.px = px;
    p.py = py;
    p.pz = pz;
    p.weight = weight;
    p.trackID = trackID;
    p.eventID = eventID;
    G4AutoLock lock(&fMutex);
    fAllParticles.push_back(p);
}

inline void DamsaFluxCollector::RecordCaloFaceParticle(G4int pdgCode, G4double energy, G4double time,
                                                        G4double x, G4double y, G4double z,
                                                        G4double px, G4double py, G4double pz,
                                                        G4int trackID, G4int eventID, G4double weight)
{
    FluxParticle p;
    p.pdgCode = pdgCode;
    p.energy = energy;
    p.time = time;
    p.x = x;
    p.y = y;
    p.z = z;
    p.px = px;
    p.py = py;
    p.pz = pz;
    p.weight = weight;
    p.trackID = trackID;
    p.eventID = eventID;
    G4AutoLock lock(&fMutex);
    fCaloFaceParticles.push_back(p);
}

inline void DamsaFluxCollector::Reset()
{
    fPhotons.clear();
    fBremsSpectrum.clear();
    fBremsCount = 0;
    fAllParticles.clear();
    fCaloFaceParticles.clear();
}


inline G4int DamsaFluxCollector::GetNeutronCount() const
{
    G4int count = 0;
    for (const auto& p : fAllParticles) {
        if (p.pdgCode == 2112) count++;  // Neutron PDG code
    }
    return count;
}

inline void DamsaFluxCollector::WriteCSV(const G4String& filename) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream outFile(fullPath);
    
    if (!outFile.is_open()) {
        G4cout << "ERROR: Could not open file " << fullPath << " for writing!" << G4endl;
        return;
    }
    
    outFile << "pdg,energy_MeV,time_ns,x_mm,y_mm,z_mm,px,py,pz,weight,trackID,eventID" << std::endl;

    for (const auto& p : fAllParticles) {
        outFile << p.pdgCode << ","
                << std::scientific << std::setprecision(6)
                << p.energy/MeV << ","
                << p.time/ns << ","
                << p.x/mm << ","
                << p.y/mm << ","
                << p.z/mm << ","
                << p.px << ","
                << p.py << ","
                << p.pz << ","
                << p.weight << ","
                << p.trackID << ","
                << p.eventID << std::endl;
    }

    outFile.close();
    G4cout << "Flux data written to: " << fullPath << " (" << fAllParticles.size() << " particles)" << G4endl;
}

inline void DamsaFluxCollector::WritePhotonFluxCSV(const G4String& filename) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream outFile(fullPath);
    
    if (!outFile.is_open()) {
        G4cout << "ERROR: Could not open file " << fullPath << " for writing!" << G4endl;
        return;
    }
    
    outFile << "energy_MeV,time_ns,x_mm,y_mm,z_mm,px,py,pz,weight,trackID,eventID" << std::endl;
    
    for (const auto& p : fPhotons) {
        outFile << std::scientific << std::setprecision(6)
                << p.energy/MeV << ","
                << p.time/ns << ","
                << p.x/mm << ","
                << p.y/mm << ","
                << p.z/mm << ","
                << p.px << ","
                << p.py << ","
                << p.pz << ","
                << p.weight << ","
                << p.trackID << ","
                << p.eventID << std::endl;
    }
    
    outFile.close();
    G4cout << "Photon flux written to: " << fullPath << " (" << fPhotons.size() << " photons)" << G4endl;
}

inline void DamsaFluxCollector::WriteBackgroundCSV(const G4String& filename) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream outFile(fullPath);
    
    if (!outFile.is_open()) {
        G4cout << "ERROR: Could not open file " << fullPath << " for writing!" << G4endl;
        return;
    }
    
    outFile << "pdg,particle_name,energy_MeV,time_ns,x_mm,y_mm,z_mm,px,py,pz,weight,trackID,eventID" << std::endl;
    
    for (const auto& p : fAllParticles) {
        // Skip photons (they're signal, not background)
        if (p.pdgCode == 22) continue;
        
        // Map PDG to name for readability
        std::string name;
        switch(p.pdgCode) {
            case 2112: name = "neutron"; break;
            case 2212: name = "proton"; break;
            case 11: name = "e-"; break;
            case -11: name = "e+"; break;
            case 211: name = "pi+"; break;
            case -211: name = "pi-"; break;
            case 111: name = "pi0"; break;
            default: name = "other"; break;
        }
        
        outFile << p.pdgCode << ","
                << name << ","
                << std::scientific << std::setprecision(6)
                << p.energy/MeV << ","
                << p.time/ns << ","
                << p.x/mm << ","
                << p.y/mm << ","
                << p.z/mm << ","
                << p.px << ","
                << p.py << ","
                << p.pz << ","
                << p.weight << ","
                << p.trackID << ","
                << p.eventID << std::endl;
    }
    
    outFile.close();
    G4cout << "Background data written to: " << fullPath << G4endl;
}

inline void DamsaFluxCollector::WriteCaloFaceCSV(const G4String& filename) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream outFile(fullPath);
    
    if (!outFile.is_open()) {
        G4cout << "ERROR: Could not open file " << fullPath << " for writing!" << G4endl;
        return;
    }
    
    outFile << "pdg,energy_MeV,time_ns,x_mm,y_mm,z_mm,px,py,pz,weight,trackID,eventID" << std::endl;
    
    // Write all particles at calorimeter face
    for (const auto& p : fCaloFaceParticles) {
        outFile << p.pdgCode << ","
                << std::scientific << std::setprecision(6)
                << p.energy/MeV << ","
                << p.time/ns << ","
                << p.x/mm << ","
                << p.y/mm << ","
                << p.z/mm << ","
                << p.px << ","
                << p.py << ","
                << p.pz << ","
                << p.weight << ","
                << p.trackID << ","
                << p.eventID << std::endl;
    }
    
    outFile.close();
    G4cout << "Calo face data written to: " << fullPath << " (" << fCaloFaceParticles.size() << " particles)" << G4endl;
}

inline std::map<G4double, G4int> DamsaFluxCollector::GetBinnedPhotonSpectrum(G4double binWidth) const
{
    std::map<G4double, G4int> spectrum;
    
    for (const auto& p : fPhotons) {
        // Bin center
        G4int binIndex = static_cast<G4int>(p.energy / binWidth);
        G4double binCenter = (binIndex + 0.5) * binWidth;
        spectrum[binCenter] += static_cast<G4int>(p.weight);
    }
    
    return spectrum;
}

inline void DamsaFluxCollector::WriteAlplibFlux(const G4String& filename, G4double nPrimaries,
                                                 G4double beamCurrent) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream outFile(fullPath);
    
    if (!outFile.is_open()) {
        G4cout << "ERROR: Could not open file " << fullPath << " for writing!" << G4endl;
        return;
    }
    
    G4double electronsPerSecond = beamCurrent / (1.602176634e-19);  // I = Q/t, e- /s = I/e
    G4double scaleFactor = electronsPerSecond / nPrimaries;  // photons/electron * electrons/s

    outFile << "# Photon flux at target exit for alplib input" << std::endl;
    outFile << "# Generated by DAMSA Geant4 simulation" << std::endl;
    outFile << "# Beam mode: " << DamsaConfig::BeamModeName(DamsaConfig::gBeamMode) << std::endl;
    outFile << "# Beam current [A]: " << std::scientific << std::setprecision(6) << beamCurrent << std::endl;
    outFile << "# Beam current: " << beamCurrent*1e6 << " uA" << std::endl;
    outFile << "# Primary electrons simulated: " << nPrimaries << std::endl;
    outFile << "# Scale factor: " << scaleFactor << " (photons/electron * electrons/s)" << std::endl;
    outFile << "# Format: energy_MeV, rate_per_second" << std::endl;
    outFile << "#" << std::endl;
    
    auto spectrum = GetBinnedPhotonSpectrum(1.0*MeV);
    
    for (const auto& bin : spectrum) {
        G4double energyMeV = bin.first / MeV;
        G4double rate = bin.second * scaleFactor;
        
        outFile << std::fixed << std::setprecision(3) << energyMeV << ","
                << std::scientific << std::setprecision(6) << rate << std::endl;
    }
    
    outFile.close();
    G4cout << "Alplib flux file written to: " << fullPath << G4endl;
    G4cout << "  " << spectrum.size() << " energy bins, " << fPhotons.size() << " total photons" << G4endl;
}

inline void DamsaFluxCollector::RecordBremsPhoton(G4double energy, G4double /*time*/,
                                                   G4double /*x*/, G4double /*y*/, G4double /*z*/,
                                                   G4double /*px*/, G4double /*py*/, G4double /*pz*/,
                                                   G4int /*trackID*/, G4int /*eventID*/, G4double weight)
{
    // Histogram at record time (1-MeV bins); per-photon storage is not
    // retained — see the fBremsSpectrum declaration for the rationale.
    const G4int binIndex = static_cast<G4int>(energy / (1.0*MeV));
    G4AutoLock lock(&fMutex);
    fBremsSpectrum[binIndex] += static_cast<G4long>(weight);
    ++fBremsCount;
}

inline std::map<G4double, G4int> DamsaFluxCollector::GetBinnedBremsSpectrum(G4double binWidth) const
{
    // Re-bin the internal 1-MeV histogram. For binWidth = 1 MeV (the only
    // width used in practice) this reproduces the per-photon binning exactly;
    // wider bins must be integer multiples of 1 MeV.
    std::map<G4double, G4int> spectrum;

    for (const auto& bin : fBremsSpectrum) {
        const G4double energy = (bin.first + 0.5) * (1.0*MeV);
        const G4int binIndex = static_cast<G4int>(energy / binWidth);
        const G4double binCenter = (binIndex + 0.5) * binWidth;
        spectrum[binCenter] += static_cast<G4int>(bin.second);
    }

    return spectrum;
}

inline void DamsaFluxCollector::WriteAlplibBremsFlux(const G4String& filename, G4double nPrimaries,
                                                     G4double beamCurrent) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream outFile(fullPath);
    
    if (!outFile.is_open()) {
        G4cout << "ERROR: Could not open file " << fullPath << " for writing!" << G4endl;
        return;
    }
    
    G4double electronsPerSecond = beamCurrent / (1.602176634e-19);
    G4double scaleFactor = electronsPerSecond / nPrimaries;
    
    outFile << "# Bremsstrahlung photon flux INSIDE target for alplib Primakoff input" << std::endl;
    outFile << "# These are hard photons (eBrem, annihil, conv) born inside tungsten" << std::endl;
    outFile << "# This is the CORRECT input spectrum for Primakoff ALP production" << std::endl;
    outFile << "# Generated by DAMSA Geant4 simulation" << std::endl;
    outFile << "# Beam mode: " << DamsaConfig::BeamModeName(DamsaConfig::gBeamMode) << std::endl;
    outFile << "# Beam current [A]: " << std::scientific << std::setprecision(6) << beamCurrent << std::endl;
    outFile << "# Beam current: " << beamCurrent*1e6 << " uA" << std::endl;
    outFile << "# Primary electrons simulated: " << nPrimaries << std::endl;
    outFile << "# Scale factor: " << scaleFactor << std::endl;
    outFile << "# Total brems photons recorded: " << fBremsCount << std::endl;
    outFile << "# Format: energy_MeV, rate_per_second" << std::endl;
    outFile << "#" << std::endl;
    
    auto spectrum = GetBinnedBremsSpectrum(1.0*MeV);
    
    for (const auto& bin : spectrum) {
        G4double energyMeV = bin.first / MeV;
        G4double rate = bin.second * scaleFactor;
        
        outFile << std::fixed << std::setprecision(3) << energyMeV << ","
                << std::scientific << std::setprecision(6) << rate << std::endl;
    }
    
    outFile.close();
    G4cout << "Alplib brems flux file written to: " << fullPath << G4endl;
    G4cout << "  " << spectrum.size() << " energy bins, " << fBremsCount << " total brems photons" << G4endl;
}

#endif
