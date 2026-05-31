#ifndef PI0DECAYDATA_H
#define PI0DECAYDATA_H

#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4AutoLock.hh"
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sys/stat.h>

// Complete record for one pi0 -> gamma+gamma decay
struct Pi0Decay {
    G4int    eventID;
    G4int    pi0TrackID;
    G4int    gamma1TrackID;
    G4int    gamma2TrackID;
    G4double vx, vy, vz;       // decay vertex (Geant4 units = mm)
    G4double e1, e2;            // daughter gamma kinetic energies (Geant4 units = MeV)
    G4double px1, py1, pz1;    // gamma1 momentum direction (unit vector, lab frame)
    G4double px2, py2, pz2;    // gamma2 momentum direction (unit vector, lab frame)
    G4double openingAngle;      // between gamma1 and gamma2 (radians, lab frame)
    G4bool   gamma1AtCalo;      // gamma1 physically crossed physScoringCaloEntrance
    G4bool   gamma2AtCalo;
    G4bool   gamma1GeomAccept;  // gamma1 direction projects geometrically to calo aperture (no material)
    G4bool   gamma2GeomAccept;
    G4double caloEnergyMeV;     // total energy deposited in physECAL by all pi0 descendants (MeV)

    Pi0Decay()
    : eventID(0), pi0TrackID(-1), gamma1TrackID(-1), gamma2TrackID(-1),
      vx(0), vy(0), vz(0), e1(0), e2(0),
      px1(0), py1(0), pz1(1), px2(0), py2(0), pz2(1),
      openingAngle(0),
      gamma1AtCalo(false), gamma2AtCalo(false),
      gamma1GeomAccept(false), gamma2GeomAccept(false),
      caloEnergyMeV(0.0)
    {}
};

// Per-thread event-local state.
// Thread-local singleton — no mutex needed.
class DamsaPi0TrackingState {
public:
    static DamsaPi0TrackingState* Instance() {
        thread_local DamsaPi0TrackingState inst;
        return &inst;
    }
    std::set<G4int>            pi0TrackIDs;        // pi0 track IDs seen in current event
    std::map<G4int, Pi0Decay>  inProgressDecays;   // pi0TID → partial decay (waiting for 2nd gamma)
    std::map<G4int, G4int>     trackToPi0Ancestor; // trackID → ancestral pi0 trackID (all descendants)
    std::map<G4int, G4double>  pi0CaloEnergy_MeV;  // pi0TID → accumulated calo energy this event

    void Reset() {
        pi0TrackIDs.clear();
        inProgressDecays.clear();
        trackToPi0Ancestor.clear();
        pi0CaloEnergy_MeV.clear();
    }

private:
    DamsaPi0TrackingState() {}
};

// Global singleton accumulating completed pi0->gg decays from all threads.
// Thread-safe via G4Mutex.
class DamsaPi0Collector {
public:
    static DamsaPi0Collector* Instance();

    // Called when both daughter gammas identified (worker thread)
    void AddCompletedDecay(const Pi0Decay& d);

    // Called when a daughter gamma crosses physScoringCaloEntrance (worker thread)
    void MarkGammaAtCalo(G4int trackID, G4int eventID);

    // Called from EndOfEventAction to flush per-thread calo energy accumulation
    void AddCaloEnergy(G4int eventID, G4int pi0TrackID, G4double edep_MeV);

    // Called from PreUserTrackingAction when pi0 is first seen
    void IncrementPi0Produced();

    // Set calo geometry (call from Construct(), before any events)
    static void SetGeometry(G4double caloZ, G4double caloHalfXY);

    // CSV output (master thread, EndOfRunAction)
    void WriteCSV(const G4String& filename) const;
    void WriteSummaryCSV(const G4String& filename) const;

    void Reset();

    const std::vector<Pi0Decay>& GetDecays() const { return fDecays; }
    G4int GetTotalDecays()      const { return (G4int)fDecays.size(); }
    G4int GetTotalPi0Produced() const { return fTotalPi0Produced; }

private:
    DamsaPi0Collector() : fTotalPi0Produced(0) {}
    ~DamsaPi0Collector() {}
    static DamsaPi0Collector* fInstance;
    static G4Mutex            fMutex;

    std::vector<Pi0Decay> fDecays;
    // Key: (eventID, trackID) — avoids cross-event aliasing (trackIDs reset per event)
    std::map<std::pair<G4int,G4int>, G4int> fGammaToDecayIdx;  // (evtID, gamTID) → fDecays idx
    std::map<std::pair<G4int,G4int>, G4int> fPi0ToDecayIdx;    // (evtID, pi0TID) → fDecays idx

    G4int fTotalPi0Produced;

    // Calo geometry for geometric acceptance (set once before events)
    static G4double fGeoCaloZ;       // z of calo entrance scoring plane (Geant4 mm)
    static G4double fGeoCaloHalfXY;  // half-width of calo aperture (Geant4 mm)
};

// ─── Implementation ─────────────────────────────────────────────────────────

inline DamsaPi0Collector* DamsaPi0Collector::Instance()
{
    if (!fInstance) fInstance = new DamsaPi0Collector();
    return fInstance;
}

inline void DamsaPi0Collector::SetGeometry(G4double caloZ, G4double caloHalfXY)
{
    fGeoCaloZ      = caloZ;
    fGeoCaloHalfXY = caloHalfXY;
}

inline void DamsaPi0Collector::AddCompletedDecay(const Pi0Decay& d)
{
    // Compute geometric acceptance: project gamma direction from decay vertex to calo plane.
    // Uses stored lab-frame momentum direction (unit vector) and decay vertex position.
    Pi0Decay dc = d;

    if (fGeoCaloHalfXY > 0.0) {
        // gamma1
        if (dc.pz1 > 1e-9) {
            G4double t = (fGeoCaloZ - dc.vz) / dc.pz1;
            if (t > 0.0) {
                G4double xh = dc.vx + t * dc.px1;
                G4double yh = dc.vy + t * dc.py1;
                dc.gamma1GeomAccept = (std::abs(xh) <= fGeoCaloHalfXY &&
                                       std::abs(yh) <= fGeoCaloHalfXY);
            }
        }
        // gamma2
        if (dc.pz2 > 1e-9) {
            G4double t = (fGeoCaloZ - dc.vz) / dc.pz2;
            if (t > 0.0) {
                G4double xh = dc.vx + t * dc.px2;
                G4double yh = dc.vy + t * dc.py2;
                dc.gamma2GeomAccept = (std::abs(xh) <= fGeoCaloHalfXY &&
                                       std::abs(yh) <= fGeoCaloHalfXY);
            }
        }
    }

    G4AutoLock lock(&fMutex);
    G4int idx = (G4int)fDecays.size();
    fDecays.push_back(dc);
    fGammaToDecayIdx[{dc.eventID, dc.gamma1TrackID}] = idx;
    if (dc.gamma2TrackID >= 0)
        fGammaToDecayIdx[{dc.eventID, dc.gamma2TrackID}] = idx;
    fPi0ToDecayIdx[{dc.eventID, dc.pi0TrackID}] = idx;
}

inline void DamsaPi0Collector::MarkGammaAtCalo(G4int trackID, G4int eventID)
{
    G4AutoLock lock(&fMutex);
    auto it = fGammaToDecayIdx.find({eventID, trackID});
    if (it == fGammaToDecayIdx.end()) return;
    Pi0Decay& d = fDecays[it->second];
    if      (d.gamma1TrackID == trackID) d.gamma1AtCalo = true;
    else if (d.gamma2TrackID == trackID) d.gamma2AtCalo = true;
}

inline void DamsaPi0Collector::AddCaloEnergy(G4int eventID, G4int pi0TrackID, G4double edep_MeV)
{
    G4AutoLock lock(&fMutex);
    auto it = fPi0ToDecayIdx.find({eventID, pi0TrackID});
    if (it == fPi0ToDecayIdx.end()) return;  // Dalitz or missed decay — silently ignore
    fDecays[it->second].caloEnergyMeV += edep_MeV;
}

inline void DamsaPi0Collector::IncrementPi0Produced()
{
    G4AutoLock lock(&fMutex);
    fTotalPi0Produced++;
}

inline void DamsaPi0Collector::Reset()
{
    G4AutoLock lock(&fMutex);
    fDecays.clear();
    fGammaToDecayIdx.clear();
    fPi0ToDecayIdx.clear();
    fTotalPi0Produced = 0;
}

inline void DamsaPi0Collector::WriteCSV(const G4String& filename) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream out(fullPath);
    if (!out.is_open()) {
        G4cout << "ERROR: Cannot open " << fullPath << G4endl;
        return;
    }

    out << "eventID,pi0TrackID,vx_mm,vy_mm,vz_mm,"
        << "gamma1TrackID,e1_MeV,px1,py1,pz1,"
        << "gamma2TrackID,e2_MeV,px2,py2,pz2,"
        << "openingAngle_deg,pi0Energy_MeV,"
        << "gamma1AtCalo,gamma2AtCalo,"
        << "gamma1GeomAccept,gamma2GeomAccept,"
        << "caloEnergyMeV\n";

    for (const auto& d : fDecays) {
        double angleDeg = d.openingAngle * 180.0 / M_PI;
        double pi0EMeV  = (d.e1 + d.e2) / MeV;
        out << d.eventID          << ","
            << d.pi0TrackID       << ","
            << std::scientific << std::setprecision(4)
            << d.vx/mm  << "," << d.vy/mm  << "," << d.vz/mm  << ","
            << d.gamma1TrackID    << ","
            << d.e1/MeV << "," << d.px1 << "," << d.py1 << "," << d.pz1 << ","
            << d.gamma2TrackID    << ","
            << d.e2/MeV << "," << d.px2 << "," << d.py2 << "," << d.pz2 << ","
            << std::fixed << std::setprecision(4)
            << angleDeg  << ","
            << std::scientific
            << pi0EMeV   << ","
            << (d.gamma1AtCalo     ? 1 : 0) << ","
            << (d.gamma2AtCalo     ? 1 : 0) << ","
            << (d.gamma1GeomAccept ? 1 : 0) << ","
            << (d.gamma2GeomAccept ? 1 : 0) << ","
            << std::fixed << std::setprecision(4)
            << d.caloEnergyMeV << "\n";
    }

    out.close();
    G4cout << "pi0 decay CSV written to: " << fullPath
           << " (" << fDecays.size() << " decays)" << G4endl;
}

inline void DamsaPi0Collector::WriteSummaryCSV(const G4String& filename) const
{
    mkdir("output", 0755);
    std::string fullPath = "output/" + filename;
    std::ofstream out(fullPath);
    if (!out.is_open()) {
        G4cout << "ERROR: Cannot open " << fullPath << G4endl;
        return;
    }

    G4int total   = (G4int)fDecays.size();
    G4int n0 = 0, n1 = 0, n2 = 0;         // scoring-plane hits
    G4int ng0 = 0, ng1 = 0, ng2 = 0;      // geometric acceptance
    G4int nCaloSignal = 0;                 // both geom-accepted AND nonzero calo energy

    for (const auto& d : fDecays) {
        G4int hits  = (d.gamma1AtCalo     ? 1 : 0) + (d.gamma2AtCalo     ? 1 : 0);
        G4int ghits = (d.gamma1GeomAccept ? 1 : 0) + (d.gamma2GeomAccept ? 1 : 0);
        if      (hits == 0) n0++;
        else if (hits == 1) n1++;
        else                n2++;
        if      (ghits == 0) ng0++;
        else if (ghits == 1) ng1++;
        else                 ng2++;
        if (ghits == 2 && d.caloEnergyMeV > 0.0) nCaloSignal++;
    }

    out << "quantity,value\n"
        << "total_pi0_produced,"                 << fTotalPi0Produced << "\n"
        << "total_pi0_gg_decays_tracked,"         << total << "\n"
        // scoring-plane (physical survival)
        << "decays_0_photons_at_calo,"            << n0    << "\n"
        << "decays_1_photon_at_calo,"             << n1    << "\n"
        << "decays_2_photons_at_calo,"            << n2    << "\n"
        << std::fixed << std::setprecision(6)
        << "fraction_1_photon_at_calo,"           << (total > 0 ? (double)n1/total : 0.0) << "\n"
        << "fraction_2_photons_at_calo,"          << (total > 0 ? (double)n2/total : 0.0) << "\n"
        // geometric acceptance (no material effects)
        << "decays_0_photons_geom_accept,"        << ng0   << "\n"
        << "decays_1_photon_geom_accept,"         << ng1   << "\n"
        << "decays_2_photons_geom_accept,"        << ng2   << "\n"
        << "fraction_1_photon_geom_accept,"       << (total > 0 ? (double)ng1/total : 0.0) << "\n"
        << "fraction_2_photons_geom_accept,"      << (total > 0 ? (double)ng2/total : 0.0) << "\n"
        // both geom-accepted AND deposited energy in calo (signal-like)
        << "decays_both_geom_with_calo_energy,"   << nCaloSignal << "\n";

    out.close();

    G4cout << "\n=== pi0 -> gamma+gamma Summary ===" << G4endl;
    G4cout << "  Total pi0 produced:        " << fTotalPi0Produced << G4endl;
    G4cout << "  Total 2γ decays tracked:   " << total << G4endl;
    G4cout << "  --- Scoring-plane (physical, with material absorption) ---" << G4endl;
    G4cout << "  0 photons at calo:         " << n0    << G4endl;
    G4cout << "  1 photon  at calo:         " << n1    << G4endl;
    G4cout << "  2 photons at calo:         " << n2    << G4endl;
    if (total > 0) {
        G4cout << "  Double-hit fraction: " << std::fixed << std::setprecision(4)
               << 100.0*n2/total << "%" << G4endl;
    }
    G4cout << "  --- Geometric acceptance (no material effects, upper bound) ---" << G4endl;
    G4cout << "  0 photons geom-accepted:   " << ng0   << G4endl;
    G4cout << "  1 photon  geom-accepted:   " << ng1   << G4endl;
    G4cout << "  2 photons geom-accepted:   " << ng2   << G4endl;
    if (total > 0) {
        G4cout << "  Geom double-accept fraction: " << std::fixed << std::setprecision(4)
               << 100.0*ng2/total << "%" << G4endl;
    }
    G4cout << "  Both geom-accept + calo energy: " << nCaloSignal << G4endl;
    G4cout << "  Summary written to: output/" << filename << G4endl;
}

#endif
