#ifndef DETECTOR_H
#define DETECTOR_H

// Timed calorimeter readout (Random-2-photons plan §5.2-5.3).
//
// DamsaCaloSD sits on every CsI crystal and accumulates, per event and per
// cell: energy, earliest deposit time, and energy-weighted mean time. The SD
// object is per worker thread (ConstructSDandField runs per worker), so its
// per-event map needs no lock; Geant4 calls Initialize() / EndOfEvent() around
// every event, which is where it resets and flushes. Only EndOfEvent touches
// the global, mutex-protected DamsaCaloHitCollector.
//
// True times only. Time resolution and thresholds for the physics study are
// applied offline, so they can be scanned without re-running Geant4.

#include "G4VSensitiveDetector.hh"
#include "G4Step.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4AutoLock.hh"

#include <map>
#include <set>
#include <string>
#include <vector>

#include "damsa_config.h"
#include "damsa_io.h"

// Global collector, written once at end of run by the master (run.h).
class DamsaCaloHitCollector {
public:
    static DamsaCaloHitCollector* Instance() {
        static DamsaCaloHitCollector inst;
        return &inst;
    }

    // Keeps whole events only: once gCaloHitEventLimit events are stored,
    // later events are dropped entirely rather than truncated.
    void AddEvent(std::vector<damsa::io::CaloHitRow>&& rows) {
        if (rows.empty()) return;
        G4AutoLock lock(&fMutex);
        const long limit = DamsaConfig::gCaloHitEventLimit;
        if (limit >= 0 && fEvents >= limit) return;
        ++fEvents;
        fRows.insert(fRows.end(), rows.begin(), rows.end());
    }

    void Reset() { G4AutoLock lock(&fMutex); fRows.clear(); fEvents = 0; }

    void WriteNTuple(const std::string& filename) const {
        const std::string path = "output/" + filename;
        damsa::io::WriteNTuple(path, fRows);
        G4cout << "Calo hits TTree written to: " << path << " (" << fRows.size()
               << " cells, " << fEvents << " events)" << G4endl;
    }

    void WriteCSV(const std::string& filename) const {
        const std::string path = "output/" + filename;
        damsa::io::EnsureParentDir(path);
        std::ofstream out(path);
        if (!out.is_open()) { G4cout << "ERROR: Could not open file " << path << G4endl; return; }
        out << damsa::io::kCaloHitCsvHeader << "\n";
        for (const auto& r : fRows) damsa::io::WriteCaloHitCsvRow(out, r);
    }

private:
    DamsaCaloHitCollector() = default;
    inline static G4Mutex fMutex = G4MUTEX_INITIALIZER;
    std::vector<damsa::io::CaloHitRow> fRows;
    long fEvents = 0;
};

class DamsaCaloSD : public G4VSensitiveDetector {
public:
    explicit DamsaCaloSD(const G4String& name) : G4VSensitiveDetector(name) {}

    void Initialize(G4HCofThisEvent*) override { fCells.clear(); }

    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        const G4double edep = step->GetTotalEnergyDeposit();
        if (edep <= 0) return false;
        // Copy number of the crystal placement = cellID (construction.cpp).
        const G4int cell = step->GetPreStepPoint()->GetTouchable()->GetCopyNumber(0);
        const G4double t = 0.5 * (step->GetPreStepPoint()->GetGlobalTime()
                                + step->GetPostStepPoint()->GetGlobalTime());
        auto [it, fresh] = fCells.try_emplace(cell, Cell{0, t, 0});
        Cell& c = it->second;
        c.edep += edep;
        c.tEsum += edep * t;
        if (!fresh && t < c.tFirst) c.tFirst = t;
        return true;
    }

    void EndOfEvent(G4HCofThisEvent*) override {
        const G4int evt = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
        std::vector<damsa::io::CaloHitRow> rows;
        rows.reserve(fCells.size());
        for (const auto& [cell, c] : fCells) {
            if (c.edep < DamsaConfig::gCaloHitMinEdep_MeV * MeV) continue;
            damsa::io::CaloHitRow r;
            r.eventID    = evt;
            r.cellID     = cell;
            r.edep_MeV   = c.edep / MeV;
            r.t_first_ns = c.tFirst / ns;
            r.t_mean_ns  = (c.tEsum / c.edep) / ns;
            rows.push_back(r);
        }
        DamsaCaloHitCollector::Instance()->AddEvent(std::move(rows));
    }

private:
    struct Cell { G4double edep, tFirst, tEsum; };
    std::map<G4int, Cell> fCells;
};

#endif
