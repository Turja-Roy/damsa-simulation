// Event-level depth-aware calo reconstruction vs truth (Calo-Reco-Explained
// §2.13): one library / gun event at a time, no beam overlay.
//
//   ./build/damsa_calo_tracks --hits output/gun_pi0_E1000_calo_hits.root \
//        --truth output/gun_pi0_E1000_calo_face_particles.root --tag pi0_E1000
//
//        [--tracker output/..._tracker_hits.root] [--traj output/..._calo_traj.root]
//
// Writes output/tracks_<tag>.csv (one row per reconstructed object: per-layer
// profile features, class EM / MIP / hadronic, Si veto, photon flag, matched
// truth and residuals), output/track_pairs_<tag>.csv (genuine truth pairs whose
// two photons passed the photon ID: reco vertex vs truth) and event displays
// plots/tracks/<tag>/evt_<id>.png (calorimeter top view; with --traj, the true
// in-calo trajectories by species). Calo-Reco-Explained §2.13-2.14.

#include "damsa_io.h"
#include "calo_reco.h"

#include <TBox.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMarker.h>
#include <TPolyLine.h>
#include <TStyle.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

struct Truth { double E, x, y, tx, ty, vx, vy, vz; int pdg, parent; };

int Match(const std::vector<Truth>& tr, double x, double y, double R, int mode)
{
    int best = -1;
    for (int i = 0; i < static_cast<int>(tr.size()); ++i) {
        const double dx = tr[i].x - x, dy = tr[i].y - y;
        const double d = mode == 0 ? std::abs(dx) : mode == 1 ? std::abs(dy) : std::hypot(dx, dy);
        if (d < R && (best < 0 || tr[i].E > tr[best].E)) best = i;
    }
    return best;
}

double Angle(double tx1, double ty1, double tx2, double ty2)
{
    const double c = (tx1 * tx2 + ty1 * ty2 + 1) /
                     (std::sqrt(tx1 * tx1 + ty1 * ty1 + 1) * std::sqrt(tx2 * tx2 + ty2 * ty2 + 1));
    return std::acos(std::min(1.0, c));
}

// Two massless photons with slopes (tx, ty): m = sqrt(2 E1 E2 (1 - cos theta12)).
double Mass(double E1, double tx1, double ty1, double E2, double tx2, double ty2)
{
    const double th = Angle(tx1, ty1, tx2, ty2);
    return std::sqrt(std::max(0.0, 2 * E1 * E2 * (1 - std::cos(th))));
}

// Validated categorical palette (dataviz): reco orange; truth gamma blue,
// e+- aqua, charged hadrons / mu violet (neutral hadrons: violet dotted).
const char* kReco = "#eb6834";
const char* kGamma = "#2a78d6";
const char* kElec = "#1baf7a";
const char* kHad = "#4a3aa7";

// Calorimeter top view: depth along x (layer 1 at the front, 1 cm per layer),
// transverse position along y, both in cm, with a grid line at every crystal
// edge. Top panel = x view (odd layers), bottom = y view (even layers), so in
// each panel the other view's layers are empty columns. Truth: the real
// trajectories inside the CsI when available (--traj), else straight photon
// lines from the calo face.
void Display(int evt, const std::vector<double>& E, const std::vector<calo::Photon3D>& ph,
             const std::vector<Truth>& tr, const std::vector<io::CaloTrajRow>* traj,
             const std::vector<calo::TrackerHit>* si, const calo::Geometry& g, const calo::RecoParams& p, double truthMin, const std::string& path)
{
    const int nLayers = static_cast<int>(E.size()) / g.nPerLayer;
    const double depth = nLayers * p.layerThick / 10.0, h = g.caloXY_mm / 20.0;   // cm
    // Depth < 0: the magnet region with the Si planes (one empty bin), drawn when
    // tracker hits are given.
    const double dLo = si ? 2 * std::floor((p.magnetZ0 - p.frontZ) / 20.0) : 0.0;   // even, so ticks fall on 2 cm
    std::vector<double> xb;
    if (si) xb.push_back(dLo);
    for (int l = 0; l <= nLayers; ++l) xb.push_back(l * p.layerThick / 10.0);
    TH2D hx(Form("hx%d", evt), "", static_cast<int>(xb.size()) - 1, xb.data(), g.nPerLayer, -h, h);
    TH2D hy(Form("hy%d", evt), "", static_cast<int>(xb.size()) - 1, xb.data(), g.nPerLayer, -h, h);
    for (int c = 0; c < static_cast<int>(E.size()); ++c) {
        if (E[c] <= 0) continue;
        const int l = c / g.nPerLayer, s = c % g.nPerLayer;
        (l % 2 ? hx : hy).Fill((l + 0.5) * p.layerThick / 10.0, g.stripPos(s) / 10.0, E[c]);
    }
    gStyle->SetTitleAlign(13);   // titles top-left: the legend sits top-right
    gStyle->SetTitleX(0.07);
    TCanvas c(Form("ev%d", evt), "", 1400, 900);
    c.Divide(1, 2, 0.002, 0.002);
    const int reco = TColor::GetColor(kReco), cg = TColor::GetColor(kGamma),
              ce = TColor::GetColor(kElec), ch = TColor::GetColor(kHad);
    const int gridCol = TColor::GetColor("#c9c8c1");
    std::vector<TObject*> keep;
    TH2D* hs[2] = {&hx, &hy};
    const char* lab[2] = {"x [cm]", "y [cm]"};
    const char* view[2] = {"x view (odd layers)", "y view (even layers)"};
    double emax = 0.05;
    for (auto* hh : hs) emax = std::max(emax, hh->GetMaximum());

    // Truth trajectories grouped per track, in step order.
    std::map<int, std::vector<const io::CaloTrajRow*>> perTrack;
    if (traj) for (const auto& r : *traj) perTrack[r.trackID].push_back(&r);

    for (int v = 0; v < 2; ++v) {
        c.cd(v + 1);
        gPad->SetRightMargin(0.12); gPad->SetLeftMargin(0.07); gPad->SetTopMargin(si && v == 0 ? 0.135 : 0.10);
        gPad->SetBottomMargin(0.14); gPad->SetLogz();
        hs[v]->SetTitle(Form("event %d -- %s;depth from the calo front [cm]  (layer n spans n-1 to n%s);%s;bar energy [MeV]",
                             evt, view[v], si ? ", Si planes at < 0" : "", lab[v]));
        hs[v]->SetMinimum(0.05);
        hs[v]->SetMaximum(emax);   // same colour scale in both panels
        hs[v]->GetXaxis()->SetNdivisions(static_cast<int>(std::lround((depth - dLo) / 2)), false);
        hs[v]->GetYaxis()->SetNdivisions(g.nPerLayer, false);
        hs[v]->GetXaxis()->SetTitleOffset(1.1);
        hs[v]->GetYaxis()->SetTitleOffset(0.6);
        hs[v]->GetXaxis()->SetTitleSize(0.05); hs[v]->GetYaxis()->SetTitleSize(0.05);
        hs[v]->GetZaxis()->SetTitleSize(0.05); hs[v]->GetZaxis()->SetTitleOffset(0.75);
        hs[v]->Draw("COLZ");
        for (int l = 0; l <= nLayers; ++l) {
            auto* L = new TLine(l * p.layerThick / 10.0, -h, l * p.layerThick / 10.0, h);
            L->SetLineColor(gridCol); L->SetLineWidth(1); L->Draw(); keep.push_back(L);
        }
        for (int s = 0; s <= g.nPerLayer; ++s) {
            const double u = -h + s * g.pitch() / 10.0;
            auto* L = new TLine(0, u, depth, u);
            L->SetLineColor(gridCol); L->SetLineWidth(1); L->Draw(); keep.push_back(L);
        }
        // Si planes (grey bars, true thickness and extent) and their hits.
        if (si) {
            for (int i = 0; i < p.nTrackers; ++i) {
                const double d = (calo::TrackerZ(i, p) - p.frontZ) / 10.0, w = p.trackerThick / 20.0;
                auto* B = new TBox(d - w, -p.trackerHalf / 10.0, d + w, p.trackerHalf / 10.0);
                B->SetFillColor(TColor::GetColor("#a8a79f")); B->Draw(); keep.push_back(B);
            }
            for (const auto& t : *si) {
                auto* M = new TMarker((calo::TrackerZ(t.plane, p) - p.frontZ) / 10.0, std::clamp((v == 0 ? t.x : t.y) / 10.0, -h, h), 20);
                M->SetMarkerColor(kBlack); M->SetMarkerSize(1.0); M->Draw(); keep.push_back(M);
            }
        }
        // Truth.
        if (traj) {
            for (const auto& [tid, pts] : perTrack) {
                if (pts.size() < 2) continue;
                auto* pl = new TPolyLine(static_cast<int>(pts.size()));
                for (std::size_t k = 0; k < pts.size(); ++k) {
                    const double d = std::clamp((pts[k]->z_mm - p.frontZ) / 10.0, dLo, depth);
                    const double u = std::clamp((v == 0 ? pts[k]->x_mm : pts[k]->y_mm) / 10.0, -h, h);
                    pl->SetPoint(static_cast<int>(k), d, u);
                }
                const int pdg = pts[0]->pdg, apdg = std::abs(pdg);
                if (pdg == 22)        { pl->SetLineColor(cg); pl->SetLineWidth(2); pl->SetLineStyle(2); }
                else if (apdg == 11)  { pl->SetLineColor(ce); pl->SetLineWidth(1); }
                else if (apdg == 2112 || apdg == 130 || apdg == 310 || apdg == 111) {
                    pl->SetLineColor(ch); pl->SetLineWidth(2); pl->SetLineStyle(3);   // neutral hadron
                } else                { pl->SetLineColor(ch); pl->SetLineWidth(3); }   // mu, pi, p, ...
                pl->Draw(); keep.push_back(pl);
            }
        }
        // Straight segments: reco objects, and truth photons when no trajectories.
        auto line = [&](double a_mm, double slope, int col, int style, int width) {
            double d1 = depth, u0 = a_mm / 10.0, u1 = u0 + slope * depth;
            if (std::abs(u1) > h) {   // leaves through the side: stop at the edge
                const double edge = u1 > 0 ? h : -h;
                if (std::abs(slope) > 1e-12) d1 = (edge - u0) / slope;
                u1 = edge;
            }
            if (d1 <= 0 || std::abs(u0) > h) return;
            auto* L = new TLine(0, u0, d1, u1);
            L->SetLineColor(col); L->SetLineWidth(width); L->SetLineStyle(style);
            L->Draw(); keep.push_back(L);
        };
        if (!traj)
            for (const auto& t : tr)
                if (t.pdg == 22 && t.E >= truthMin) line(v == 0 ? t.x : t.y, v == 0 ? t.tx : t.ty, cg, 2, 2);
        for (const auto& q : ph)
            line(v == 0 ? q.x : q.y, v == 0 ? q.tx : q.ty, reco, q.photon ? 1 : 7, q.photon ? 3 : 2);
        if (si)   // reconstructed charged tracks through the Si planes, up to the calo face
            for (const auto& t : calo::FindSiTracks(*si, p)) {
                const double a = (v == 0 ? t.x : t.y) / 10.0, b = v == 0 ? t.tx : t.ty;
                const double d0 = (calo::TrackerZ(0, p) - p.frontZ) / 10.0;
                auto* L = new TLine(d0, std::clamp(a + b * d0, -h, h), 0, std::clamp(a, -h, h));
                L->SetLineColor(reco); L->SetLineWidth(3); L->SetLineStyle(10);
                L->Draw(); keep.push_back(L);
            }
    }
    c.cd(1);
    TLegend leg(0.30, si ? 0.87 : 0.905, 0.88, 0.998);
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.040); leg.SetNColumns(3);
    TLine lr, lj, lg, le, lh, ln, lc;
    TMarker ms(0, 0, 20);
    lc.SetLineColor(reco); lc.SetLineWidth(3); lc.SetLineStyle(10);
    lr.SetLineColor(reco); lr.SetLineWidth(3);
    lj.SetLineColor(reco); lj.SetLineWidth(2); lj.SetLineStyle(7);
    lg.SetLineColor(cg); lg.SetLineWidth(2); lg.SetLineStyle(2);
    le.SetLineColor(ce); le.SetLineWidth(1);
    lh.SetLineColor(ch); lh.SetLineWidth(3);
    ln.SetLineColor(ch); ln.SetLineWidth(2); ln.SetLineStyle(3);
    leg.AddEntry(&lr, "reco photon", "l");
    leg.AddEntry(&lj, "reco rejected (MIP/hadr./charged)", "l");
    if (si) {
        leg.AddEntry(&lc, "reco Si track (charged)", "l");
        leg.AddEntry(&ms, "Si hit", "p");
    }
    leg.AddEntry(&lg, traj ? "true #gamma" : Form("true #gamma #geq %.0f MeV", truthMin), "l");
    if (traj) {
        leg.AddEntry(&le, "true e^{#pm}", "l");
        leg.AddEntry(&lh, "true #mu/#pi/p", "l");
        leg.AddEntry(&ln, "true n (neutral)", "l");
    }
    leg.Draw();
    c.SaveAs((path + ".png").c_str());
    for (auto* o : keep) delete o;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string hits, truthPath, trackerPath, trajPath, tag = "run", outDir = "output", plotDir = "plots/tracks";
    calo::RecoParams p;
    calo::Geometry g;
    double cellThr = 0.1, minE = 5.0, matchR = 35.3, truthMin = 5.0;
    int nDisplay = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--hits")               hits = nx();
        else if (s == "--truth")              truthPath = nx();
        else if (s == "--tracker")            trackerPath = nx();
        else if (s == "--traj")               trajPath = nx();
        else if (s == "--tag")                tag = nx();
        else if (s == "--out-dir")            outDir = nx();
        else if (s == "--plot-dir")           plotDir = nx();
        else if (s == "--cell-threshold-MeV") cellThr = std::stod(nx());
        else if (s == "--seed-MeV")           p.seed = std::stod(nx());
        else if (s == "--min-photon-MeV")     minE = std::stod(nx());
        else if (s == "--truth-min-MeV")      truthMin = std::stod(nx());
        else if (s == "--match-radius-mm")    matchR = std::stod(nx());
        else if (s == "--max-miss")           p.maxMiss = std::stoi(nx());
        else if (s == "--max-balance")        p.maxBalance = std::stod(nx());
        else if (s == "--max-start-gap")      p.maxStartGap = std::stod(nx());
        else if (s == "--max-depth-gap-mm")   p.maxDepthGap = std::stod(nx());
        else if (s == "--track-seed-MeV")     p.trackSeed = std::stod(nx());
        else if (s == "--em-max-start-layer") p.emMaxStartLayer = std::stoi(nx());
        else if (s == "--em-start-per-decade") p.emStartPerDecade = std::stod(nx());
        else if (s == "--sat-efrac")          p.satEfrac = std::stod(nx());
        else if (s == "--sat-radius-mm")      p.satRadius = std::stod(nx());
        else if (s == "--em-max-tmax-pull")   p.emMaxTmaxPull = std::stod(nx());
        else if (s == "--mip-min-layers")     p.mipMinLayers = std::stoi(nx());
        else if (s == "--mip-max-bars")       p.mipMaxBars = std::stod(nx());
        else if (s == "--min-veto-planes")    p.minVetoPlanes = std::stoi(nx());
        else if (s == "--si-road-mm")         p.siRoad = std::stod(nx());
        else if (s == "--charged-mask-mm")    p.chargedMask = std::stod(nx());
        else if (s == "--n-display")          nDisplay = std::stoi(nx());
        else {
            std::fprintf(stderr, "usage: %s --hits PATH --truth PATH [--tag NAME] [--tracker PATH] [--traj PATH]\n"
                                 "  [--cell-threshold-MeV 0.1 --seed-MeV 0.5 --min-photon-MeV 5\n"
                                 "   --truth-min-MeV 5 --match-radius-mm 35.3 --max-miss 2 --n-display 20\n"
                                 "   --max-balance 0.8 --max-start-gap 3 --max-depth-gap-mm 40 --track-seed-MeV 2\n"
                                 "   --em-max-start-layer 6 --em-max-tmax-pull 3 --mip-min-layers 8 --mip-max-bars 1.6\n"
                                 "   --min-veto-planes 4 --veto-road-mm 20 --charged-mask-mm 70.6 --out-dir output --plot-dir plots/tracks]\n", argv[0]);
            return 1;
        }
    }
    if (hits.empty() || truthPath.empty()) { std::fprintf(stderr, "Error: --hits and --truth required.\n"); return 1; }

    std::map<int, std::vector<io::CaloHitRow>> evHits;
    int nCells = 0;
    for (const auto& h : io::ReadNTuple<io::CaloHitRow>(hits)) {
        evHits[h.eventID].push_back(h);
        nCells = std::max(nCells, h.cellID + 1);
    }
    nCells = (nCells + g.nPerLayer - 1) / g.nPerLayer * g.nPerLayer;
    // Truth: every particle crossing the calo face (all species, so objects made
    // by pi / p / n match what really made them); neutrinos never deposit.
    std::map<int, std::vector<Truth>> evTruth;
    for (const auto& r : io::ReadNTuple<io::ParticleRow>(truthPath)) {
        const int a = std::abs(r.pdg);
        if (r.pz <= 0 || r.energy_MeV < 1 || a == 12 || a == 14 || a == 16) continue;
        if (r.parentID < 0) { std::fprintf(stderr, "Error: %s has no truth columns.\n", truthPath.c_str()); return 1; }
        evTruth[r.eventID].push_back({r.energy_MeV, r.x_mm, r.y_mm, r.px / r.pz, r.py / r.pz,
                                      r.vx_mm, r.vy_mm, r.vz_mm, r.pdg, r.parentID});
    }
    std::map<int, std::vector<calo::TrackerHit>> evTracker;
    if (!trackerPath.empty())
        for (const auto& h : io::ReadNTuple<io::TrackerHitRow>(trackerPath))
            evTracker[h.eventID].push_back({h.plane, h.x_mm, h.y_mm});
    std::map<int, std::vector<io::CaloTrajRow>> evTraj;
    if (!trajPath.empty())
        for (const auto& r : io::ReadNTuple<io::CaloTrajRow>(trajPath)) evTraj[r.eventID].push_back(r);

    const std::string dispDir = plotDir + "/" + tag;
    std::filesystem::create_directories(outDir);
    if (nDisplay > 0) std::filesystem::create_directories(dispDir);
    std::ofstream fo(outDir + "/tracks_" + tag + ".csv"), fp(outDir + "/track_pairs_" + tag + ".csv");
    fo << "eventID,E,x,y,tx,ty,sx,sy,stx,sty,nLayers,start,tmaxLayer,tmaxPull,balance,fixedDir,"
          "nLit,last,medLayerE,layerEcv,peakRatio,meanBars,widthEarly,widthLate,cls,vetoPlanes,vetoInfo,photon,"
          "charged,siX,siY,siTx,siTy,matched,ghost,true_pdg,true_parent,true_E,true_x,true_y,true_tx,true_ty,true_vz,"
          "dtheta_mrad,dx_at_vertex_mm,dy_at_vertex_mm\n";
    fp << "eventID,E1,E2,true_E1,true_E2,vtx_x,vtx_y,vtx_z,vtx_dist,true_vx,true_vy,true_vz,"
          "m_reco_dir,m_fixed_vertex,m_true\n";

    gStyle->SetOptStat(0);
    {   // Single-hue (neutral) sequential map for bar energy; lines carry the categorical colours.
        double st[] = {0.0, 1.0}, r[] = {0.94, 0.13}, gg[] = {0.94, 0.13}, b[] = {0.93, 0.13};
        TColor::CreateGradientColorTable(2, st, r, gg, b, 64);
    }

    // [0] all objects, [1] photon ID (EM and not Si-vetoed)
    long nTruthPh = 0, nFound[2] = {0, 0}, nObj[2] = {0, 0}, nMatchedGamma[2] = {0, 0}, nCls[3] = {0, 0, 0},
         nVetoed = 0, nDisp = 0;
    const double half = g.caloXY_mm / 2;
    for (const auto& [evt, hv] : evHits) {
        std::vector<double> E(nCells, 0.0), T(nCells, 0.0);
        for (const auto& h : hv)
            if (h.edep_MeV >= cellThr) { E[h.cellID] = h.edep_MeV; T[h.cellID] = h.t_mean_ns; }
        std::vector<calo::Photon3D> ph;
        for (const auto& q : calo::ReconstructDepth(E, T, g, p)) if (q.E >= minE) ph.push_back(q);
        if (!trackerPath.empty()) {
            const auto it = evTracker.find(evt);
            static const std::vector<calo::TrackerHit> none;
            std::vector<char> before(ph.size());
            for (std::size_t i = 0; i < ph.size(); ++i) before[i] = ph[i].photon;
            calo::ApplyVeto(ph, it == evTracker.end() ? none : it->second, p);
            for (std::size_t i = 0; i < ph.size(); ++i) nVetoed += before[i] && !ph[i].photon;
        }
        const auto& tr = evTruth[evt];

        std::vector<char> found[2] = {std::vector<char>(tr.size(), 0), std::vector<char>(tr.size(), 0)};
        std::vector<int> match(ph.size(), -1);
        for (std::size_t i = 0; i < ph.size(); ++i) {
            const auto& q = ph[i];
            const int m = Match(tr, q.x, q.y, matchR, 2);
            const int mx = Match(tr, q.x, q.y, matchR, 0), my = Match(tr, q.x, q.y, matchR, 1);
            const bool ghost = mx >= 0 && my >= 0 && mx != my;
            match[i] = m;
            ++nCls[q.cls];
            for (int k = 0; k < 2; ++k) {
                if (k == 1 && !q.photon) continue;
                ++nObj[k];
                if (m >= 0 && tr[m].pdg == 22) { ++nMatchedGamma[k]; found[k][m] = 1; }
            }
            fo << evt << "," << q.E << "," << q.x << "," << q.y << "," << q.tx << "," << q.ty << ","
               << q.sx << "," << q.sy << "," << q.stx << "," << q.sty << "," << q.nLayers << ","
               << q.start << "," << q.tmaxLayer << "," << q.tmaxPull << "," << q.balance << ","
               << q.fixedDir << "," << q.nLit << "," << q.last << "," << q.medLayerE << "," << q.layerEcv << ","
               << q.peakRatio << "," << q.meanBars << "," << q.widthEarly << "," << q.widthLate << ","
               << q.cls << "," << q.vetoPlanes << "," << q.vetoInfo << "," << q.photon << ","
               << q.charged << "," << q.siX << "," << q.siY << "," << q.siTx << "," << q.siTy << ","
               << (m >= 0) << "," << ghost;
            if (m >= 0) {
                const Truth& t = tr[m];
                const double dz = t.vz - p.frontZ;   // extrapolate the reco line back to the truth vertex depth
                fo << "," << t.pdg << "," << t.parent << "," << t.E << "," << t.x << "," << t.y << ","
                   << t.tx << "," << t.ty << "," << t.vz << ","
                   << 1e3 * Angle(q.tx, q.ty, t.tx, t.ty) << ","
                   << q.x + q.tx * dz - t.vx << "," << q.y + q.ty * dz - t.vy << "\n";
            } else {
                fo << ",,,,,,,,,,,\n";
            }
        }
        for (std::size_t i = 0; i < tr.size(); ++i)
            if (tr[i].pdg == 22 && tr[i].E >= truthMin && std::abs(tr[i].x) < half && std::abs(tr[i].y) < half) {
                ++nTruthPh; nFound[0] += found[0][i]; nFound[1] += found[1][i];
            }

        // Genuine truth pairs (same parent + vertex) with both photons passing the photon ID.
        for (std::size_t i = 0; i < ph.size(); ++i)
        for (std::size_t j = i + 1; j < ph.size(); ++j) {
            const int a = match[i], b = match[j];
            if (!ph[i].photon || !ph[j].photon || a < 0 || b < 0 || a == b) continue;
            const Truth &u = tr[a], &w = tr[b];
            if (u.pdg != 22 || w.pdg != 22 || u.parent != w.parent ||
                std::abs(u.vz - w.vz) > 1e-3 || std::abs(u.vx - w.vx) > 1e-3) continue;
            const auto v = calo::ClosestApproach(ph[i].x, ph[i].y, ph[i].tx, ph[i].ty,
                                                 ph[j].x, ph[j].y, ph[j].tx, ph[j].ty, p.frontZ);
            const double L = p.frontZ - p.vertexZ;
            fp << evt << "," << ph[i].E << "," << ph[j].E << "," << u.E << "," << w.E << ","
               << v.x << "," << v.y << "," << v.z << "," << v.dist << ","
               << u.vx << "," << u.vy << "," << u.vz << ","
               << Mass(ph[i].E, ph[i].tx, ph[i].ty, ph[j].E, ph[j].tx, ph[j].ty) << ","
               << calo::Mgg(ph[i].E, ph[i].x, ph[i].y, ph[j].E, ph[j].x, ph[j].y, L) << ","
               << Mass(u.E, u.tx, u.ty, w.E, w.tx, w.ty) << "\n";
        }

        // Displays: with --traj, only events that have trajectories.
        const auto tj = evTraj.find(evt);
        const bool haveTraj = tj != evTraj.end();
        if (nDisp < nDisplay && !ph.empty() && (trajPath.empty() || haveTraj)) {
            const auto si = evTracker.find(evt);
            Display(evt, E, ph, tr, haveTraj ? &tj->second : nullptr, si == evTracker.end() ? nullptr : &si->second, g, p, truthMin,
                    dispDir + "/evt_" + std::to_string(evt));
            ++nDisp;
        }
    }
    auto frac = [](long a, long b) { return b ? double(a) / double(b) : 0.0; };
    std::printf("[%s] events %zu | objects %ld: EM %ld, MIP %ld, hadronic %ld; Si-vetoed EM %ld\n",
                tag.c_str(), evHits.size(), nObj[0], nCls[calo::kEM], nCls[calo::kMIP], nCls[calo::kHadronic], nVetoed);
    std::printf("[%s] truth photons >= %.0f MeV in aperture %ld | all objects: found %.3f, gamma-purity %.3f"
                " | photon ID: %ld photons, found %.3f, gamma-purity %.3f\n",
                tag.c_str(), truthMin, nTruthPh, frac(nFound[0], nTruthPh), frac(nMatchedGamma[0], nObj[0]),
                nObj[1], frac(nFound[1], nTruthPh), frac(nMatchedGamma[1], nObj[1]));
    return 0;
}
