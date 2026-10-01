// Event-level depth-aware calo reconstruction vs truth (Calo-Reco-Explained
// §2.13): one library / gun event at a time, no beam overlay.
//
//   ./build/damsa_calo_tracks --hits output/gun_pi0_E1000_calo_hits.root \
//        --truth output/gun_pi0_E1000_calo_face_particles.root --tag pi0_E1000 [--no-energy]
//
// Writes output/tracks_<tag>.csv (one row per reconstructed photon, with its
// matched truth and residuals), output/track_pairs_<tag>.csv (genuine truth
// pairs whose two photons were both reconstructed: reco vertex vs truth) and
// event displays plots/tracks/<tag>/evt_<id>.png: (z, x) and (z, y) maps of the
// bar energies, reconstructed photon lines (solid) and truth photon lines (dashed).

#include "damsa_io.h"
#include "calo_reco.h"

#include <TCanvas.h>
#include <TColor.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLine.h>
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

void Display(int evt, const std::vector<double>& E, const std::vector<calo::Photon3D>& ph,
             const std::vector<Truth>& tr, const calo::Geometry& g, const calo::RecoParams& p,
             double truthMin, const std::string& path)
{
    const int nLayers = static_cast<int>(E.size()) / g.nPerLayer;
    const double z0 = p.frontZ, z1 = p.frontZ + nLayers * p.layerThick, h = g.caloXY_mm / 2;
    TH2D hx(Form("hx%d", evt), "", nLayers, z0, z1, g.nPerLayer, -h, h);
    TH2D hy(Form("hy%d", evt), "", nLayers, z0, z1, g.nPerLayer, -h, h);
    for (int c = 0; c < static_cast<int>(E.size()); ++c) {
        if (E[c] <= 0) continue;
        const int l = c / g.nPerLayer, s = c % g.nPerLayer;
        (l % 2 ? hx : hy).Fill(calo::LayerZ(l, p), g.stripPos(s), E[c]);
    }
    TCanvas c(Form("ev%d", evt), "", 1300, 560);
    c.Divide(2, 1, 0.002, 0.002);
    const int reco = TColor::GetColor("#eb6834"), truth = TColor::GetColor("#2a78d6");
    std::vector<TLine*> keep;
    TH2D* hs[2] = {&hx, &hy};
    const char* lab[2] = {"x [mm]  (odd layers)", "y [mm]  (even layers)"};
    for (int v = 0; v < 2; ++v) {
        c.cd(v + 1);
        gPad->SetRightMargin(0.15); gPad->SetLeftMargin(0.12); gPad->SetLogz();
        hs[v]->SetTitle(Form("event %d;z [mm] (world);%s;bar energy [MeV]", evt, lab[v]));
        hs[v]->SetMinimum(0.05);
        hs[v]->Draw("COLZ");
        for (const auto& q : ph) {
            const double a = v == 0 ? q.x : q.y, b = v == 0 ? q.tx : q.ty;
            auto* L = new TLine(z0, a, z1, a + b * (z1 - z0));
            L->SetLineColor(reco); L->SetLineWidth(2);
            L->Draw(); keep.push_back(L);
        }
        for (const auto& t : tr) {
            if (t.pdg != 22 || t.E < truthMin) continue;
            const double a = v == 0 ? t.x : t.y, b = v == 0 ? t.tx : t.ty;
            auto* L = new TLine(z0, a, z1, a + b * (z1 - z0));
            L->SetLineColor(truth); L->SetLineWidth(2); L->SetLineStyle(2);
            L->Draw(); keep.push_back(L);
        }
    }
    c.cd(1);
    TLegend leg(0.13, 0.80, 0.60, 0.92);
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.035);
    TLine lr, lt;
    lr.SetLineColor(reco); lr.SetLineWidth(2);
    lt.SetLineColor(truth); lt.SetLineWidth(2); lt.SetLineStyle(2);
    leg.AddEntry(&lr, "reconstructed photon", "l");
    leg.AddEntry(&lt, Form("truth photon #geq %.0f MeV", truthMin), "l");
    leg.Draw();
    c.SaveAs((path + ".png").c_str());
    for (auto* L : keep) delete L;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string hits, truthPath, tag = "run", outDir = "output", plotDir = "plots/tracks";
    calo::RecoParams p;
    calo::Geometry g;
    double cellThr = 0.1, minE = 5.0, matchR = 36.0, truthMin = 5.0;
    int nDisplay = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--hits")               hits = nx();
        else if (s == "--truth")              truthPath = nx();
        else if (s == "--tag")                tag = nx();
        else if (s == "--out-dir")            outDir = nx();
        else if (s == "--plot-dir")           plotDir = nx();
        else if (s == "--no-energy")          p.useEnergy = false;
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
        else if (s == "--n-display")          nDisplay = std::stoi(nx());
        else {
            std::fprintf(stderr, "usage: %s --hits PATH --truth PATH [--tag NAME] [--no-energy]\n"
                                 "  [--cell-threshold-MeV 0.1 --seed-MeV 0.5 --min-photon-MeV 5\n"
                                 "   --truth-min-MeV 5 --match-radius-mm 36 --max-miss 2 --n-display 20\n"
                                 "   --max-balance 0.8 --max-start-gap 3 --max-depth-gap-mm 40 --track-seed-MeV 2\n"
                                 "   --out-dir output --plot-dir plots/tracks]\n", argv[0]);
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
    std::map<int, std::vector<Truth>> evTruth;
    for (const auto& r : io::ReadNTuple<io::ParticleRow>(truthPath)) {
        if (r.pz <= 0 || r.energy_MeV < 1 || (r.pdg != 22 && std::abs(r.pdg) != 11)) continue;
        if (r.parentID < 0) { std::fprintf(stderr, "Error: %s has no truth columns.\n", truthPath.c_str()); return 1; }
        evTruth[r.eventID].push_back({r.energy_MeV, r.x_mm, r.y_mm, r.px / r.pz, r.py / r.pz,
                                      r.vx_mm, r.vy_mm, r.vz_mm, r.pdg, r.parentID});
    }

    const std::string dispDir = plotDir + "/" + tag + (p.useEnergy ? "" : "_noE");
    std::filesystem::create_directories(outDir);
    if (nDisplay > 0) std::filesystem::create_directories(dispDir);
    const std::string suffix = tag + (p.useEnergy ? "" : "_noE");
    std::ofstream fo(outDir + "/tracks_" + suffix + ".csv"), fp(outDir + "/track_pairs_" + suffix + ".csv");
    fo << "eventID,E,x,y,tx,ty,sx,sy,stx,sty,nLayers,start,tmaxLayer,tmaxPull,balance,fixedDir,"
          "matched,ghost,true_pdg,true_parent,true_E,true_x,true_y,true_tx,true_ty,true_vz,"
          "dtheta_mrad,dx_at_vertex_mm,dy_at_vertex_mm\n";
    fp << "eventID,E1,E2,true_E1,true_E2,vtx_x,vtx_y,vtx_z,vtx_dist,true_vx,true_vy,true_vz,"
          "m_reco_dir,m_fixed_vertex,m_true\n";

    gStyle->SetOptStat(0);
    // Single-hue (neutral) sequential map for bar energy; lines carry the categorical colours.
    {
        double st[] = {0.0, 1.0}, r[] = {0.94, 0.13}, gg[] = {0.94, 0.13}, b[] = {0.93, 0.13};
        TColor::CreateGradientColorTable(2, st, r, gg, b, 64);
    }

    long nTruthPh = 0, nFound = 0, nReco = 0, nMatched = 0, nGhost = 0, nDisp = 0;
    const double half = g.caloXY_mm / 2;
    for (const auto& [evt, hv] : evHits) {
        std::vector<double> E(nCells, 0.0), T(nCells, 0.0);
        for (const auto& h : hv)
            if (h.edep_MeV >= cellThr) { E[h.cellID] = h.edep_MeV; T[h.cellID] = h.t_mean_ns; }
        std::vector<calo::Photon3D> ph;
        for (const auto& q : calo::ReconstructDepth(E, T, g, p)) if (q.E >= minE) ph.push_back(q);
        const auto& tr = evTruth[evt];

        std::vector<char> found(tr.size(), 0);
        std::vector<int> match(ph.size(), -1);
        for (std::size_t i = 0; i < ph.size(); ++i) {
            const auto& q = ph[i];
            const int m = Match(tr, q.x, q.y, matchR, 2);
            const int mx = Match(tr, q.x, q.y, matchR, 0), my = Match(tr, q.x, q.y, matchR, 1);
            const bool ghost = mx >= 0 && my >= 0 && mx != my;
            match[i] = m;
            ++nReco; nMatched += m >= 0; nGhost += ghost;
            if (m >= 0) found[m] = 1;
            fo << evt << "," << q.E << "," << q.x << "," << q.y << "," << q.tx << "," << q.ty << ","
               << q.sx << "," << q.sy << "," << q.stx << "," << q.sty << "," << q.nLayers << ","
               << q.start << "," << q.tmaxLayer << "," << q.tmaxPull << "," << q.balance << ","
               << q.fixedDir << "," << (m >= 0) << "," << ghost;
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
                ++nTruthPh; nFound += found[i];
            }

        // Genuine truth pairs (same parent + vertex) with both photons reconstructed.
        for (std::size_t i = 0; i < ph.size(); ++i)
        for (std::size_t j = i + 1; j < ph.size(); ++j) {
            const int a = match[i], b = match[j];
            if (a < 0 || b < 0 || a == b) continue;
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

        if (nDisp < nDisplay && !ph.empty()) {
            Display(evt, E, ph, tr, g, p, truthMin, dispDir + "/evt_" + std::to_string(evt));
            ++nDisp;
        }
    }
    std::printf("[%s%s] events %zu | truth photons >= %.0f MeV in aperture %ld, found %ld (%.3f) | "
                "reco photons %ld, matched %ld (%.3f), ghosts %ld (%.3f)\n",
                tag.c_str(), p.useEnergy ? "" : " no-energy", evHits.size(), truthMin, nTruthPh, nFound,
                nTruthPh ? double(nFound) / nTruthPh : 0.0, nReco, nMatched,
                nReco ? double(nMatched) / nReco : 0.0, nGhost, nReco ? double(nGhost) / nReco : 0.0);
    return 0;
}
