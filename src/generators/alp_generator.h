#ifndef ALP_GENERATOR_H
#define ALP_GENERATOR_H

// ALP decay photon injection generator for DAMSA
// Reads alplib-exported decay photon CSV (γγ pairs from a→γγ) and fires them
// as primary particles into Geant4 for full detector response simulation.
//
// CSV format (produced by scripts/pipeline/alp_signal_pipeline.py):
//   E1_MeV,px1,py1,pz1,E2_MeV,px2,py2,pz2,weight[,decay_z_m]
// where px/py/pz are momentum components, weight is events/day for that
// decay pair, and decay_z_m (optional) is the sampled ALP decay-vertex z in
// metres downstream of the target centre (legacy 9-column files: 0).
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
    double weight;   // events/day (from alplib)
    double decayZ_m; // decay-vertex z offset from target centre [m]
                     // (optional 10th CSV column; 0 = fire from target centre)
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
    : fVertexZ(vertex_z_cm * cm), fRefireFactor(refire_factor)
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
        // Index the CSV row by GLOBAL event ID, not a per-instance counter.
        // In MT mode each worker thread owns its own generator instance; a
        // sequential per-instance counter would make every thread start again
        // at row 0 (early rows fired once per thread, late rows never fired).
        // eventID-based indexing fires each row exactly fRefireFactor times
        // regardless of how events are scheduled across threads.
        const std::size_t idx =
            (static_cast<std::size_t>(event->GetEventID()) / fRefireFactor)
            % fEvents.size();
        const ALPDecayEvent& ev = fEvents[idx];

        // Store current event weight for retrieval by the stepping action.
        // Thread-local: stepping for this event runs on this same thread.
        // (row weight divided by refire factor to preserve total)
        fgCurrentEventWeight = ev.weight / G4double(fRefireFactor);

        // Per-event decay vertex: target centre + sampled ALP flight distance
        // (decay_z_m column; 0 for legacy 9-column CSVs). ALPs fly along +z
        // (alplib fires theta_ALP = 0), so the vertex stays on the beam axis.
        fParticleGun->SetParticlePosition(
            G4ThreeVector(0., 0., fVertexZ + ev.decayZ_m * m));

        // Fire γ1
        fParticleGun->SetParticleEnergy(ev.E1 * MeV);
        fParticleGun->SetParticleMomentumDirection(G4ThreeVector(ev.px1, ev.py1, ev.pz1));
        fParticleGun->GeneratePrimaryVertex(event);

        // Fire γ2 (as a second vertex in the same event)
        fParticleGun->SetParticleEnergy(ev.E2 * MeV);
        fParticleGun->SetParticleMomentumDirection(G4ThreeVector(ev.px2, ev.py2, ev.pz2));
        fParticleGun->GeneratePrimaryVertex(event);
    }

    // Weight of the event currently being processed ON THIS THREAD.
    // fInstance is shared (last constructed instance wins), so an instance
    // member would race across workers; the thread-local static does not.
    G4double GetCurrentEventWeight() const { return fgCurrentEventWeight; }
    
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
    G4int  fRefireFactor = 1;     // each input row fired this many times
    G4int  fRejectedRows = 0;     // number of zero-weight rows skipped during load

    static DamsaALPDecayGenerator* fInstance;
    // Per-thread current event weight (events over exposure / refire_factor).
    static thread_local G4double fgCurrentEventWeight;

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
                // Optional 10th column: decay-vertex z [m from target centre].
                // Legacy 9-column CSVs fall back to the fixed vertex.
                if (!(iss >> ev.decayZ_m)) ev.decayZ_m = 0.0;
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
inline thread_local G4double DamsaALPDecayGenerator::fgCurrentEventWeight = 1.0;

#endif // ALP_GENERATOR_H
