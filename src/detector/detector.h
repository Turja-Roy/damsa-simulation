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

// Global per-event row collector (one instance per row type), written once at
// end of run by the master (run.h). Keeps whole events only: once `limit`
// events are stored, later events are dropped entirely rather than truncated.
// limit < 0 = keep all.
template <class Row>
class DamsaEventRows {
public:
    static DamsaEventRows* Instance() {
        static DamsaEventRows inst;
        return &inst;
    }

    void AddEvent(std::vector<Row>&& rows, long limit) {
        if (rows.empty()) return;
        G4AutoLock lock(&fMutex);
        if (limit >= 0 && fEvents >= limit) return;
        ++fEvents;
        fRows.insert(fRows.end(), rows.begin(), rows.end());
    }

    // Unlocked peek, good enough to skip recording work once the cap is reached.
    bool Full(long limit) const { return limit >= 0 && fEvents >= limit; }

    void Reset() { G4AutoLock lock(&fMutex); fRows.clear(); fEvents = 0; }

    void WriteNTuple(const std::string& filename) const {
        const std::string path = "output/" + filename;
        damsa::io::WriteNTuple(path, fRows);
        G4cout << filename << " written to: " << path << " (" << fRows.size()
               << " rows, " << fEvents << " events)" << G4endl;
    }

    void WriteCSV(const std::string& filename, const char* header,
                  void (*emit)(std::ostream&, const Row&)) const {
        const std::string path = "output/" + filename;
        damsa::io::EnsureParentDir(path);
        std::ofstream out(path);
        if (!out.is_open()) { G4cout << "ERROR: Could not open file " << path << G4endl; return; }
        out << header << "\n";
        for (const auto& r : fRows) emit(out, r);
    }

private:
    DamsaEventRows() = default;
    inline static G4Mutex fMutex = G4MUTEX_INITIALIZER;
    std::vector<Row> fRows;
    long fEvents = 0;
};

using DamsaCaloHitCollector    = DamsaEventRows<damsa::io::CaloHitRow>;
using DamsaTrackerHitCollector = DamsaEventRows<damsa::io::TrackerHitRow>;
using DamsaCaloTrajCollector   = DamsaEventRows<damsa::io::CaloTrajRow>;

// Si tracker planes (charged-particle veto, Calo-Reco-Explained §2.14): one row
// per (plane, track) with summed energy and energy-weighted position. The
// plane index is the placement copy number (construction.cpp). Uses the same
// event cap as the calo hits.
class DamsaTrackerSD : public G4VSensitiveDetector {
public:
    explicit DamsaTrackerSD(const G4String& name) : G4VSensitiveDetector(name) {}

    void Initialize(G4HCofThisEvent*) override { fHits.clear(); }

    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        const G4double edep = step->GetTotalEnergyDeposit();
        if (edep <= 0) return false;
        const G4int plane = step->GetPreStepPoint()->GetTouchable()->GetCopyNumber(0);
        const G4Track* tr = step->GetTrack();
        const G4ThreeVector pos = 0.5 * (step->GetPreStepPoint()->GetPosition() +
                                         step->GetPostStepPoint()->GetPosition());
        auto [it, fresh] = fHits.try_emplace({plane, tr->GetTrackID()},
                                             Acc{0, {0, 0, 0}, step->GetPreStepPoint()->GetGlobalTime(),
                                                 tr->GetDefinition()->GetPDGEncoding()});
        Acc& a = it->second;
        a.edep += edep;
        a.xyz += edep * pos;
        a.t = std::min(a.t, step->GetPreStepPoint()->GetGlobalTime());
        return true;
    }

    void EndOfEvent(G4HCofThisEvent*) override {
        const G4int evt = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
        std::vector<damsa::io::TrackerHitRow> rows;
        for (const auto& [key, a] : fHits) {
            damsa::io::TrackerHitRow r;
            r.eventID = evt; r.plane = key.first; r.trackID = key.second; r.pdg = a.pdg;
            const G4ThreeVector p = a.xyz / a.edep;
            r.x_mm = p.x() / mm; r.y_mm = p.y() / mm; r.z_mm = p.z() / mm;
            r.edep_MeV = a.edep / MeV; r.t_ns = a.t / ns;
            rows.push_back(r);
        }
        DamsaTrackerHitCollector::Instance()->AddEvent(std::move(rows), DamsaConfig::gCaloHitEventLimit);
    }

private:
    struct Acc { G4double edep; G4ThreeVector xyz; G4double t; G4int pdg; };
    std::map<std::pair<G4int, G4int>, Acc> fHits;
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
        DamsaCaloHitCollector::Instance()->AddEvent(std::move(rows), DamsaConfig::gCaloHitEventLimit);
    }

private:
    struct Cell { G4double edep, tFirst, tEsum; };
    std::map<G4int, Cell> fCells;
};

#endif
