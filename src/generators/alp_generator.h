#ifndef ALP_GENERATOR_H
#define ALP_GENERATOR_H

// ALP decay photon injection generator for DAMSA
// Reads exported decay photons (γγ pairs from a→γγ) and fires them as primary
// particles into Geant4 for full detector response simulation.
//
// Two input formats, chosen by extension:
//   *.root  TTree "alp_decays", written by tools/damsa_alp_signal  [preferred]
//   *.csv   the pre-migration format from scripts/pipeline/alp_signal_pipeline.py
//
// Both carry the same columns:
//   E1_MeV,px1,py1,pz1,E2_MeV,px2,py2,pz2,weight_evts_per_day[,decay_z_m]
// where px/py/pz are momentum components, weight is events/day for that decay
// pair, and decay_z_m is the sampled ALP decay-vertex z in metres downstream of
// the target centre (legacy 9-column CSVs: 0).
//
// The TTree is ~4x smaller (243 MB vs 995 MB at ma=100 MeV) and is what the
// pipeline now produces; the CSV reader is kept so existing files still load.
//
// Usage in action.h Build():
//   SetUserAction(new DamsaALPDecayGenerator("alp_decay_photons_ma100MeV.root"));

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "G4ThreeVector.hh"

#include "damsa_io.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <iostream>
#include <mutex>
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
    // path: decay photon file, .root (TTree) or .csv
    // vertex_z_cm: z position of the ALP production/decay vertex, in cm.
    //   Default = −45 cm = target centre (target rear at −40 cm, half-length 5 cm).
    // refire_factor: number of times each input row is re-fired (for better calo
    //   response statistics). Row weight is divided by this factor.
    explicit DamsaALPDecayGenerator(const std::string& path,
                                    G4double vertex_z_cm = -45.0,
                                    G4int refire_factor = 1)
    : fVertexZ(vertex_z_cm * cm), fRefireFactor(refire_factor)
    {
        fInstance = this;  // Set static instance for access from stepping action
        fParticleGun = new G4ParticleGun(1);
        G4ParticleDefinition* gamma = G4ParticleTable::GetParticleTable()->FindParticle("gamma");
        fParticleGun->SetParticleDefinition(gamma);
        fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., fVertexZ));

        fEvents = &LoadShared(path, fRejectedRows);

        if (fEvents->empty()) {
            throw std::runtime_error("DamsaALPDecayGenerator: no events loaded from " + path);
        }
        G4cout << "[ALP Generator] Loaded " << fEvents->size()
               << " decay pairs from " << path
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
            % fEvents->size();
        const ALPDecayEvent& ev = (*fEvents)[idx];

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
        for (const auto& ev : *fEvents) total += ev.weight;
        return total;
    }

    G4int GetNEvents() const { return static_cast<G4int>(fEvents->size()); }

private:
    G4ParticleGun* fParticleGun;
    G4double fVertexZ;
    const std::vector<ALPDecayEvent>* fEvents = nullptr;   // borrowed, see LoadShared
    G4int  fRefireFactor = 1;     // each input row fired this many times
    G4int  fRejectedRows = 0;     // number of zero-weight rows skipped during load

    static DamsaALPDecayGenerator* fInstance;
    // Per-thread current event weight (events over exposure / refire_factor).
    static thread_local G4double fgCurrentEventWeight;

    // Geant4 MT calls Build() on every worker, so one generator is constructed
    // per thread. Loading per instance would multiply the I/O and the memory (a
    // 4M-row file is ~300 MB) by the thread count, and concurrent reads of the
    // same ROOT file are not safe -- doing so aborts with "gInterpreter not
    // initialized". Load once, share the immutable result.
    //
    // The path is the same for every worker (set once in DamsaConfig), so the
    // first caller wins; a mismatch means the caller changed it mid-run.
    static const std::vector<ALPDecayEvent>& LoadShared(const std::string& path,
                                                        G4int& rejectedOut)
    {
        static std::vector<ALPDecayEvent> events;
        static std::string loadedPath;
        static G4int rejected = 0;
        static std::once_flag once;

        std::call_once(once, [&] {
            loadedPath = path;
            LoadInto(path, events, rejected);
        });

        if (path != loadedPath) {
            throw std::runtime_error(
                "DamsaALPDecayGenerator: already loaded '" + loadedPath +
                "', cannot also load '" + path + "' in the same process");
        }
        rejectedOut = rejected;
        return events;
    }

    // Dispatch on extension. TTree is what damsa_alp_signal writes; the CSV
    // reader stays for the pre-migration alp_decay_photons_ma*MeV.csv files
    // (including legacy 9-column ones without decay_z_m).
    static void LoadInto(const std::string& path,
                         std::vector<ALPDecayEvent>& out, G4int& rejected)
    {
        if (path.size() > 5 && path.compare(path.size() - 5, 5, ".root") == 0)
            LoadRoot(path, out, rejected);
        else
            LoadCsv(path, out, rejected);
    }

    static void LoadRoot(const std::string& path,
                         std::vector<ALPDecayEvent>& out, G4int& rejected)
    {
        // Streamed, not ReadNTuple(): a 4M-row file would otherwise exist twice
        // over while the returned vector is copied into fEvents.
        damsa::io::NTupleReader<damsa::io::AlpDecayRow> reader(path);
        const auto n = reader.Entries();
        out.reserve(n);

        for (std::uint64_t i = 0; i < n; ++i) {
            const auto& r = reader.At(i);
            // Same rejection as the CSV path: alplib emits zero-weight rows for
            // grid bins with no ALP production, and firing them only wastes
            // Geant4 events.
            if (r.weight <= 0.0) { ++rejected; continue; }
            ALPDecayEvent ev;
            ev.E1 = r.E1; ev.px1 = r.px1; ev.py1 = r.py1; ev.pz1 = r.pz1;
            ev.E2 = r.E2; ev.px2 = r.px2; ev.py2 = r.py2; ev.pz2 = r.pz2;
            ev.weight   = r.weight;
            ev.decayZ_m = r.decayZ_m;
            out.push_back(ev);
        }
    }

    static void LoadCsv(const std::string& csv_path,
                        std::vector<ALPDecayEvent>& out, G4int& rejected)
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
                    ++rejected;
                    continue;
                }
                out.push_back(ev);
            }
        }
    }
};

// Static member initialization
inline DamsaALPDecayGenerator* DamsaALPDecayGenerator::fInstance = nullptr;
inline thread_local G4double DamsaALPDecayGenerator::fgCurrentEventWeight = 1.0;

#endif // ALP_GENERATOR_H
