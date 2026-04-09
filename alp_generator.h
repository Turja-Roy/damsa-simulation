#ifndef ALP_GENERATOR_H
#define ALP_GENERATOR_H

// ALP decay photon injection generator for DAMSA
// Reads alplib-exported decay photon CSV (γγ pairs from a→γγ) and fires them
// as primary particles into Geant4 for full detector response simulation.
//
// CSV format (produced by scripts/alp_signal_pipeline.py):
//   E1_MeV,px1,py1,pz1,E2_MeV,px2,py2,pz2,weight
// where px/py/pz are unit momentum direction components and weight is
// the event rate in events/s for that decay pair.
//
// Usage in action.h Build():
//   SetUserAction(new DamsaALPDecayGenerator("alp_decay_photons_ma100MeV.csv"));

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "G4ThreeVector.hh"

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <iostream>
#include <stdexcept>

struct ALPDecayEvent {
    // γ1 and γ2 4-vectors: (E_MeV, px_dir, py_dir, pz_dir) each
    double E1, px1, py1, pz1;
    double E2, px2, py2, pz2;
    double weight;  // events/day (from alplib)
};

class DamsaALPDecayGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    // csv_path: path to alplib decay photon CSV
    // vertex_z_cm: z position of the ALP production/decay vertex, in cm.
    //   Default = −45 cm = target centre (target rear at −40 cm, half-length 5 cm).
    // refire_factor: number of times each input row is re-fired (for better calo
    //   response statistics). Row weight is divided by this factor.
    explicit DamsaALPDecayGenerator(const std::string& csv_path,
                                    G4double vertex_z_cm = -45.0,
                                    G4int refire_factor = 1)
    : fVertexZ(vertex_z_cm * cm), fRefireFactor(refire_factor), fRefireCounter(0)
    {
        fInstance = this;  // Set static instance for access from stepping action
        fParticleGun = new G4ParticleGun(1);
        G4ParticleDefinition* gamma = G4ParticleTable::GetParticleTable()->FindParticle("gamma");
        fParticleGun->SetParticleDefinition(gamma);
        fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., fVertexZ));

        LoadEvents(csv_path);

        if (fEvents.empty()) {
            throw std::runtime_error("DamsaALPDecayGenerator: no events loaded from " + csv_path);
        }
        G4cout << "[ALP Generator] Loaded " << fEvents.size()
               << " decay pairs from " << csv_path
               << " (" << fRejectedRows << " zero-weight rows skipped)"
               << ", refire factor = " << fRefireFactor << G4endl;
    }

    ~DamsaALPDecayGenerator() override { delete fParticleGun; }

    void GeneratePrimaries(G4Event* event) override
    {
        const ALPDecayEvent& ev = fEvents[fCurrentEvent];
        
        // Store current event weight for retrieval by stepping action
        // (row weight divided by refire factor to preserve total)
        fCurrentEventWeight = ev.weight / G4double(fRefireFactor);

        // Fire γ1
        fParticleGun->SetParticleEnergy(ev.E1 * MeV);
        fParticleGun->SetParticleMomentumDirection(G4ThreeVector(ev.px1, ev.py1, ev.pz1));
        fParticleGun->GeneratePrimaryVertex(event);

        // Fire γ2 (as a second vertex in the same event)
        fParticleGun->SetParticleEnergy(ev.E2 * MeV);
        fParticleGun->SetParticleMomentumDirection(G4ThreeVector(ev.px2, ev.py2, ev.pz2));
        fParticleGun->GeneratePrimaryVertex(event);
        
        // Advance refire counter; move to next row when all refires done
        if (++fRefireCounter >= fRefireFactor) {
            fRefireCounter = 0;
            if (++fCurrentEvent >= fEvents.size()) fCurrentEvent = 0;
        }
    }
    
    // Get the weight for the current event (for use by stepping action)
    G4double GetCurrentEventWeight() const { return fCurrentEventWeight; }
    
    // Static instance pointer for access from stepping action
    static DamsaALPDecayGenerator* Instance() { return fInstance; }

    G4double GetTotalWeight() const {
        double total = 0.0;
        for (const auto& ev : fEvents) total += ev.weight;
        return total;
    }

    G4int GetNEvents() const { return static_cast<G4int>(fEvents.size()); }

private:
    G4ParticleGun* fParticleGun;
    G4double fVertexZ;
    std::vector<ALPDecayEvent> fEvents;
    size_t fCurrentEvent = 0;
    G4int  fRefireFactor = 1;     // each input row fired this many times
    G4int  fRefireCounter = 0;    // 0..fRefireFactor-1
    G4int  fRejectedRows = 0;     // number of zero-weight rows skipped during load
    G4double fCurrentEventWeight = 1.0;  // weight for current event (events/day / refire_factor)
    
    static DamsaALPDecayGenerator* fInstance;

    void LoadEvents(const std::string& csv_path)
    {
        std::ifstream f(csv_path);
        if (!f.is_open()) {
            throw std::runtime_error("DamsaALPDecayGenerator: cannot open " + csv_path);
        }

        std::string line;
        bool header_skipped = false;
        while (std::getline(f, line)) {
            // Skip comment and header lines
            if (line.empty() || line[0] == '#') continue;
            if (!header_skipped) {
                // Skip the first non-comment line if it contains letters (header)
                bool has_alpha = false;
                for (char c : line) {
                    if (std::isalpha(c)) { has_alpha = true; break; }
                }
                if (has_alpha) { header_skipped = true; continue; }
                header_skipped = true;
            }

            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            ALPDecayEvent ev;
            if (iss >> ev.E1 >> ev.px1 >> ev.py1 >> ev.pz1
                    >> ev.E2 >> ev.px2 >> ev.py2 >> ev.pz2 >> ev.weight) {
                // Skip physically meaningless rows: alplib writes zero-weight
                // entries for grid bins with no ALP production (low-E for heavy
                // masses, high-E for light masses). Keeping them only wastes
                // Geant4 events and produces zero-weight calo-face records.
                if (ev.weight <= 0.0) {
                    ++fRejectedRows;
                    continue;
                }
                fEvents.push_back(ev);
            }
        }
    }
};

// Static member initialization
inline DamsaALPDecayGenerator* DamsaALPDecayGenerator::fInstance = nullptr;

#endif // ALP_GENERATOR_H
