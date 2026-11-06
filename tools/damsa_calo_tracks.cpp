// Event-level depth-aware calo reconstruction vs truth (Calo-Reco-Explained
// §2.13): one library / gun event at a time, no beam overlay.
//
//   ./build/damsa_calo_tracks --hits output/gun_pi0_E1000_calo_hits.root \
//        --truth output/gun_pi0_E1000_calo_face_particles.root --tag pi0_E1000
//
//        [--tracker output/..._tracker_hits.root] [--traj output/..._calo_traj.root]
//        [--decays output/gun_pi0_E1000.root]
//
// Writes output/tracks_<tag>.csv (one row per reconstructed object: per-layer
// profile features, class EM / MIP / hadronic, Si veto, photon flag, matched
// truth and residuals), output/track_pairs_<tag>.csv (genuine truth pairs whose
// two photons passed the photon ID: reco vertex vs truth) and event displays
// plots/tracks/<tag>/evt_<id>.png (top view from the target to the calo back:
// truth from each creation vertex, with --traj the true in-calo trajectories by
// species; reco lines extrapolated upstream and pair vertices). Calo-Reco-Explained
// §2.13-2.14, Display-Upstream plan.
// With --pid output/pid_table.csv (damsa_track_plots --pid-table), every object
// also gets species probabilities P_gamma..P_n (Edge-HadronID-ALP plan B3),
// shown next to it in the displays as its two most likely species.
// With --decays (the file given to damsa_alp_inject, row i = event i), the two
// primary photons are drawn even when they miss the calo, each object gets the row's
// weight, displays split into <tag>/2gamma and <tag>/1gamma by photons aimed at the
// face, and track_pairs counts conversions (e+- matched from different primaries).

#include "damsa_io.h"
#include "calo_reco.h"

#include <TBox.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TH2D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMarker.h>
#include <TPolyLine.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

struct Truth { double E, x, y, tx, ty, vx, vy, vz; int pdg, parent, proc, primary; };

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

// Parent mass from a decay row. px, py, pz give only the direction: damsa_gun writes unit vectors.
double RowMass(const io::AlpDecayRow& r)
{
    const double n1 = std::hypot(r.px1, r.py1, r.pz1), n2 = std::hypot(r.px2, r.py2, r.pz2);
    const double c = (r.px1 * r.px2 + r.py1 * r.py2 + r.pz1 * r.pz2) / (n1 * n2);
    return std::sqrt(std::max(0.0, 2 * r.E1 * r.E2 * (1 - c)));
}

// Validated categorical palette (dataviz, all pairs): reco photon orange, reco charged
// magenta (CVD 7.8 against aqua: thick lines and direct labels carry it too); truth gamma
// blue, e+- aqua, charged hadrons / mu violet (neutral hadrons: violet dotted).
// Reco hadronic objects are neutral ink. Solid = measured, dotted = extrapolated.
const char* kReco = "#eb6834";
const char* kCharged = "#d81b78";
const char* kGamma = "#2a78d6";
const char* kElec = "#1baf7a";
const char* kHad = "#4a3aa7";

// Event display, top view: depth along x from the target front to the calo back
// (0 = calo front, layer n spans n-1 to n), transverse position along y, both in
// cm and to scale, with a grid line at every crystal edge. Top panel = x view
// (odd layers), bottom = y view (even layers), so in each panel the other view's
// layers are empty columns. Upstream of the calo (Display-Upstream plan):
//   truth   -- every face particle (>= 1 MeV, 1-3 per event) as a straight line from its creation
//              vertex (the magnet is off), a marker per vertex labelled with the true
//              species leaving it, plus the decay or process where it sends >= 2;
//   reco    -- photons: line through the calo, dotted back to the target; charged: one path
//              from the Si line through the calo, thin past its shower onset; hadronic: its
//              calo line; a star with m_gg [MeV] where each pair of fiducial photons comes closest.
// The transverse axis is the calo face (+-6 cm): the VDC wall (r = 10 cm) is
// off-panel, and points outside the band are clamped and drawn open.
// Inside the CsI, truth = the real trajectories when available (--traj), else
// straight photon lines from the calo face.
using Prob = std::array<double, calo::kNPid>;

// construction.cpp defaults (as damsa_eventview): W target, then the VDC up to the magnet.
constexpr double kTargetZ0 = -500, kTargetLen = 100, kCapThick = 5;   // mm, world

// Label of a primary vertex: injected samples fire the decay photons as primaries,
// so the parent comes from the sample tag.
std::string PrimaryLabel(const std::string& tag)
{
    if (tag.rfind("pi0", 0) == 0) return "#pi^{0}#rightarrow#gamma#gamma";
    if (tag.rfind("eta", 0) == 0) return "#eta#rightarrow#gamma#gamma";
    if (tag.rfind("alp_ma", 0) == 0) return "a#rightarrow#gamma#gamma (" + tag.substr(6) + " MeV)";
    return "primary";
}

void Display(int evt, const std::vector<double>& E, const std::vector<calo::Photon3D>& ph,
             const std::vector<Prob>& P, const std::vector<Truth>& tr, const std::vector<io::CaloTrajRow>* traj,
             const std::vector<calo::TrackerHit>* si, const std::vector<calo::ChargedPath>& paths,
             const io::AlpDecayRow* dec, const calo::Geometry& g, const calo::RecoParams& p, double truthMin,
             const std::string& primLabel, const std::string& info, const std::string& path)
{
    const int nLayers = static_cast<int>(E.size()) / g.nPerLayer;
    const double depth = nLayers * p.layerThick / 10.0, h = g.caloXY_mm / 20.0;   // cm
    auto D = [&](double z_mm) { return (z_mm - p.frontZ) / 10.0; };              // world z -> depth [cm]
    const double dT0 = D(kTargetZ0), dT1 = D(kTargetZ0 + kTargetLen), dM = D(p.magnetZ0);
    const double dLo = -4 * std::ceil(-dT0 / 4);   // a whole 4 cm tick before the target
    std::vector<double> xb{dLo};
    for (int l = 0; l <= nLayers; ++l) xb.push_back(l * p.layerThick / 10.0);
    TH2D hx(Form("hx%d", evt), "", static_cast<int>(xb.size()) - 1, xb.data(), g.nPerLayer, -h, h);
    TH2D hy(Form("hy%d", evt), "", static_cast<int>(xb.size()) - 1, xb.data(), g.nPerLayer, -h, h);
    for (int c = 0; c < static_cast<int>(E.size()); ++c) {
        if (E[c] <= 0) continue;
        const int l = c / g.nPerLayer, s = c % g.nPerLayer;
        (l % 2 ? hx : hy).Fill((l + 0.5) * p.layerThick / 10.0, g.stripPos(s) / 10.0, E[c]);
    }
    gStyle->SetTitleAlign(13);   // titles top-left: the legend sits top-right
    gStyle->SetTitleX(0.04);
    // Width from the depth span, so 1 cm is as long on both axes (1400 px held 56 cm).
    TCanvas c(Form("ev%d", evt), "", static_cast<int>(std::lround(25 * (depth - dLo))), 900);
    c.Divide(1, 2, 0.002, 0.002);
    const int reco = TColor::GetColor(kReco), cg = TColor::GetColor(kGamma),
              ce = TColor::GetColor(kElec), ch = TColor::GetColor(kHad), cc = TColor::GetColor(kCharged);
    const int gridCol = TColor::GetColor("#c9c8c1"), steel = TColor::GetColor("#a8a79f"),
              ink = TColor::GetColor("#3d3d3a");
    std::vector<TObject*> keep;
    TH2D* hs[2] = {&hx, &hy};
    const char* lab[2] = {"x [cm]", "y [cm]"};
    const char* view[2] = {"x view (odd layers)", "y view (even layers)"};
    double emax = 0.05;
    for (auto* hh : hs) emax = std::max(emax, hh->GetMaximum());

    // Truth trajectories grouped per track, in step order.
    std::map<int, std::vector<const io::CaloTrajRow*>> perTrack;
    if (traj) for (const auto& r : *traj) perTrack[r.trackID].push_back(&r);
    auto styleTruth = [&](TAttLine* l, int pdg) {
        const int apdg = std::abs(pdg);
        if (pdg == 22)        { l->SetLineColor(cg); l->SetLineWidth(2); l->SetLineStyle(2); }
        else if (apdg == 11)  { l->SetLineColor(ce); l->SetLineWidth(1); l->SetLineStyle(1); }
        else if (apdg == 2112 || apdg == 130 || apdg == 310 || apdg == 111) {
            l->SetLineColor(ch); l->SetLineWidth(2); l->SetLineStyle(3);   // neutral hadron
        } else                { l->SetLineColor(ch); l->SetLineWidth(3); l->SetLineStyle(1); }   // mu, pi, p, ...
    };
    // Truth vertices: particles sharing parent and vertex (a decay, a conversion pair).
    struct Vtx { double x, y, z; int n, proc; std::vector<int> pdg; };
    std::vector<Vtx> vtx;
    const double zDec = dec ? p.vertexZ + dec->decayZ_m * 1000 : 0;
    if (dec) vtx.push_back({0, 0, zDec, 2, io::Proc::Primary, {22, 22}});
    for (const auto& t : tr) {
        if (dec && t.parent == 0) continue;   // the decay row draws these, and any partner that missed
        auto it = std::find_if(vtx.begin(), vtx.end(), [&](const Vtx& w) {
            return std::abs(w.x - t.vx) < 1e-3 && std::abs(w.y - t.vy) < 1e-3 && std::abs(w.z - t.vz) < 1e-3; });
        if (it == vtx.end()) vtx.push_back({t.vx, t.vy, t.vz, 1, t.proc, {t.pdg}});
        else { ++it->n; it->pdg.push_back(t.pdg); }
    }
    auto species = [](int pdg) -> std::string {
        switch (pdg) {
            case 22:    return "#gamma";
            case 11:    return "e^{-}";     case -11:   return "e^{+}";
            case 13:    return "#mu^{-}";   case -13:   return "#mu^{+}";
            case 211:   return "#pi^{+}";   case -211:  return "#pi^{-}";   case 111: return "#pi^{0}";
            case 321:   return "K^{+}";     case -321:  return "K^{-}";
            case 130:   return "K^{0}_{L}"; case 310:   return "K^{0}_{S}";
            case 2112:  return "n";         case -2112: return "#bar{n}";
            case 2212:  return "p";         case -2212: return "#bar{p}";
            default:    return std::to_string(pdg);
        }
    };
    auto procLabel = [&](int proc) -> std::string {
        switch (proc) {
            case io::Proc::Primary: return primLabel;
            case io::Proc::eBrem:   return "brems";
            case io::Proc::Decay:   return "decay";
            case io::Proc::Conv:    return "conv";
            case io::Proc::Compt:   return "Compton";
            case io::Proc::Annihil: return "annihil";
            default:                return "other";
        }
    };

    for (int v = 0; v < 2; ++v) {
        c.cd(v + 1);
        gPad->SetRightMargin(0.07); gPad->SetLeftMargin(0.04); gPad->SetTopMargin(0.135);
        gPad->SetBottomMargin(0.14); gPad->SetLogz();
        hs[v]->SetTitle(Form("event %d -- %s%s;depth from the calo front [cm]  (CsI layer n spans n-1 to n);%s;bar energy [MeV]",
                             evt, view[v], v == 1 ? info.c_str() : "", lab[v]));   // top title shares its row with the legend
        hs[v]->SetMinimum(0.05);
        hs[v]->SetMaximum(emax);   // same colour scale in both panels
        hs[v]->GetXaxis()->SetNdivisions(static_cast<int>(std::lround((depth - dLo) / 4)), false);
        hs[v]->GetYaxis()->SetNdivisions(g.nPerLayer, false);
        hs[v]->GetXaxis()->SetTitleOffset(1.1);
        hs[v]->GetYaxis()->SetTitleOffset(0.35);
        hs[v]->GetYaxis()->SetTickLength(0.006); hs[v]->GetZaxis()->SetTickLength(0.006);   // fractions of the wide pad
        hs[v]->GetXaxis()->SetTitleSize(0.05); hs[v]->GetYaxis()->SetTitleSize(0.05);
        hs[v]->GetZaxis()->SetTitleSize(0.05); hs[v]->GetZaxis()->SetTitleOffset(0.45);
        hs[v]->Draw("COLZ");
        // Upstream geometry: W target, VDC end caps, region names.
        auto box = [&](double d0, double d1, int col) {
            auto* B = new TBox(d0, -h, d1, h);
            B->SetFillColor(col); B->Draw(); keep.push_back(B);
        };
        box(dT0, dT1, TColor::GetColor("#e4e3dd"));
        box(dT1, dT1 + kCapThick / 10.0, steel);
        box(dM - kCapThick / 10.0, dM, steel);
        if (v == 0) {   // the CsI needs no name: its grid shows it, and a label there sits on showers
            const std::pair<double, const char*> region[] = {
                {dT0, "W target"}, {dT1, "VDC (vacuum)"}, {dM, si ? "magnet, Si" : "magnet"}};
            for (const auto& [d, name] : region) {
                auto* T = new TLatex(d + 0.4, 0.86 * h, name);
                T->SetTextSize(0.04); T->SetTextColor(ink); T->SetTextAlign(13);
                T->Draw(); keep.push_back(T);
            }
        }
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
                const double d = D(calo::TrackerZ(i, p)), w = p.trackerThick / 20.0;
                auto* B = new TBox(d - w, -p.trackerHalf / 10.0, d + w, p.trackerHalf / 10.0);
                B->SetFillColor(steel); B->Draw(); keep.push_back(B);
            }
            for (const auto& t : *si) {
                auto* M = new TMarker(D(calo::TrackerZ(t.plane, p)), std::clamp((v == 0 ? t.x : t.y) / 10.0, -h, h), 20);
                M->SetMarkerColor(kBlack); M->SetMarkerSize(1.0); M->Draw(); keep.push_back(M);
            }
        }
        // A straight segment, clipped to the transverse band |u| <= h.
        auto seg = [&](double d0, double u0, double d1, double u1, int col, int style, int width) -> TLine* {
            double a = 0, b = 1;
            const double du = u1 - u0;
            if (std::abs(du) < 1e-12) { if (std::abs(u0) > h) return nullptr; }
            else {
                double t0 = (-h - u0) / du, t1 = (h - u0) / du;
                if (t0 > t1) std::swap(t0, t1);
                a = std::max(a, t0); b = std::min(b, t1);
                if (a >= b) return nullptr;
            }
            auto* L = new TLine(d0 + a * (d1 - d0), u0 + a * du, d0 + b * (d1 - d0), u0 + b * du);
            L->SetLineColor(col); L->SetLineWidth(width); L->SetLineStyle(style);
            L->Draw(); keep.push_back(L);
            return L;
        };
        // A marker, open when clamped into the panel.
        auto mark = [&](double d, double u, int style, int openStyle, int col, double size) {
            const double dc = std::clamp(d, dLo, depth), uc = std::clamp(u, -h, h);
            auto* M = new TMarker(dc, uc, dc == d && uc == u ? style : openStyle);
            M->SetMarkerColor(col); M->SetMarkerSize(size); M->Draw(); keep.push_back(M);
            return std::pair<double, double>{dc, uc};
        };
        // Truth: in the CsI, then upstream from each creation vertex to the calo face.
        if (traj) {
            for (const auto& [tid, pts] : perTrack) {
                if (pts.size() < 2) continue;
                auto* pl = new TPolyLine(static_cast<int>(pts.size()));
                for (std::size_t k = 0; k < pts.size(); ++k) {
                    const double d = std::clamp(D(pts[k]->z_mm), dLo, depth);
                    const double u = std::clamp((v == 0 ? pts[k]->x_mm : pts[k]->y_mm) / 10.0, -h, h);
                    pl->SetPoint(static_cast<int>(k), d, u);
                }
                styleTruth(pl, pts[0]->pdg);
                pl->Draw(); keep.push_back(pl);
            }
        }
        for (const auto& t : tr)
            if (t.vz < p.frontZ && !(dec && t.parent == 0))
                if (auto* L = seg(std::max(D(t.vz), dLo), (v == 0 ? t.vx : t.vy) / 10.0, 0, (v == 0 ? t.x : t.y) / 10.0, 1, 1, 1))
                    styleTruth(L, t.pdg);
        // Primary photons from the decay row: to the face or the panel's edge, or to their conversion.
        if (dec) {
            auto ray = [&](double px, double py, double pz) {
                if (std::abs(pz) < 1e-6 * std::sqrt(px * px + py * py + pz * pz)) return;   // sideways: off the panel at once
                const double tx = px / pz, ty = py / pz;
                double zEnd = pz > 0 ? p.frontZ : p.frontZ + dLo * 10;
                for (const auto& t : tr)
                    if (t.proc == io::Proc::Conv && t.vz > zDec && t.vz < zEnd &&
                        std::hypot(tx * (t.vz - zDec) - t.vx, ty * (t.vz - zDec) - t.vy) < 2) zEnd = t.vz;
                if (auto* L = seg(D(zDec), 0, D(zEnd), (v == 0 ? tx : ty) * (zEnd - zDec) / 10.0, 1, 1, 1)) styleTruth(L, 22);
            };
            ray(dec->px1, dec->py1, dec->pz1);
            ray(dec->px2, dec->py2, dec->pz2);
        }
        for (const auto& w : vtx) {
            const auto [d, u] = mark(D(w.z), (v == 0 ? w.x : w.y) / 10.0, 33, 27, ink, 1.6);
            // True species leaving the vertex; the process too when it sends >= 2 of them.
            std::vector<int> pdgs = w.pdg;
            std::sort(pdgs.begin(), pdgs.end());
            std::string text;
            for (int q : pdgs) text += species(q);
            if (w.n >= 2) text += " (" + procLabel(w.proc) + ")";
            auto* T = new TLatex(d + 0.4, u + (u > 0 ? -0.35 : 0.35), text.c_str());
            T->SetTextSize(0.04); T->SetTextColor(ink); T->SetTextAlign(u > 0 ? 13 : 11);
            T->Draw(); keep.push_back(T);
        }
        // Straight segments: reco objects, and truth photons in the CsI when no trajectories.
        auto line = [&](double a_mm, double slope, int col, int style, int width, double dEnd) {
            seg(0, a_mm / 10.0, dEnd, a_mm / 10.0 + slope * dEnd, col, style, width);
        };
        if (!traj)
            for (const auto& t : tr)
                if (t.pdg == 22 && t.E >= truthMin) line(v == 0 ? t.x : t.y, v == 0 ? t.tx : t.ty, cg, 2, 2, depth);
        std::vector<const calo::ChargedPath*> pathOf(ph.size(), nullptr);
        for (const auto& c : paths) pathOf[c.obj] = &c;
        for (std::size_t k = 0; k < ph.size(); ++k) {
            const auto& q = ph[k];
            if (pathOf[k]) continue;
            const double a = (v == 0 ? q.x : q.y) / 10.0, b = v == 0 ? q.tx : q.ty;
            line(a * 10, b, q.photon ? reco : ink, 1, q.photon ? 3 : 2, (q.last + 1) * p.layerThick / 10.0);   // face to last layer
            if (!q.photon) continue;
            seg(dT0, a + b * dT0, 0, a, reco, 3, 1);   // extrapolated back to the target
            if (q.converted && q.siFirstPlane >= 0) {   // measured in the Si
                const double d0 = D(calo::TrackerZ(q.siFirstPlane, p));
                seg(d0, a + b * d0, 0, a, reco, 1, 3);
            }
        }
        // Si lines, measured from their first plane to the face; dotted back to the target.
        auto siLine = [&](double a, double b, int firstPlane, int col) {
            const double d0 = D(calo::TrackerZ(firstPlane, p));
            seg(d0, a + b * d0, 0, a, col, 1, 3);
            seg(dT0, a + b * dT0, d0, a + b * d0, col, 3, 1);
        };
        if (si)
            for (const auto& t : calo::FindSiTracks(*si, p))
                siLine((v == 0 ? t.x : t.y) / 10.0, v == 0 ? t.tx : t.ty, t.firstPlane, t.conv ? reco : cc);
        // Charged paths: from the Si line at the face along the filtered points, thin past the onset.
        for (const auto& c : paths) {
            const auto& q = ph[c.obj];
            const double a = (q.charged ? (v == 0 ? q.siX : q.siY) : (v == 0 ? q.x : q.y)) / 10.0;
            if (q.charged && q.stub) siLine(a, v == 0 ? q.siTx : q.siTy, q.siFirstPlane, cc);
            double d0 = 0, u0 = a, uOnset = a;
            for (const auto& pt : c.view[v == 0 ? 1 : 0]) {
                const double d1 = (pt.layer + 0.5) * p.layerThick / 10.0, u1 = pt.u / 10.0;
                seg(d0, u0, d1, u1, cc, 1, c.onset >= 0 && pt.layer > c.onset ? 2 : 3);
                if (pt.layer <= c.onset) uOnset = u1;
                d0 = d1; u0 = u1;
            }
            if (c.end == calo::kEndShower) mark((c.onset + 0.5) * p.layerThick / 10.0, uOnset, 34, 28, cc, 1.5);
        }
        // Reco pair vertices: closest approach of each pair of fiducial photons, with m_gg.
        // It is a 3D point: in a view where the two lines are nearly parallel, the star
        // need not sit on their crossing.
        for (std::size_t i = 0; i < ph.size(); ++i)
            for (std::size_t j = i + 1; j < ph.size(); ++j) {
                const auto &a = ph[i], &b = ph[j];
                if (!a.photon || !b.photon || !a.fiducial || !b.fiducial) continue;
                const auto w = calo::ClosestApproach(a.x, a.y, a.tx, a.ty, b.x, b.y, b.tx, b.ty, p.frontZ);
                const auto [d, u] = mark(D(w.z), (v == 0 ? w.x : w.y) / 10.0, 29, 30, reco, 2.2);
                auto* T = new TLatex(d + 0.4, u + (u > 0 ? -0.35 : 0.35),
                                     Form("%.0f", Mass(a.E, a.tx, a.ty, b.E, b.tx, b.ty)));
                T->SetTextSize(0.04); T->SetTextColor(ink); T->SetTextAlign(u > 0 ? 13 : 11);
                T->Draw(); keep.push_back(T);
            }
        // Direct labels at each object's end: what a path did, its two most probable species.
        static const char* tex[calo::kNPid] = {"#gamma", "e", "#pi", "p", "n"};
        for (std::size_t k = 0; k < ph.size(); ++k) {
            const auto& q = ph[k];
            std::string text;
            double d = (q.last + 1) * p.layerThick / 10.0;
            double u = ((v == 0 ? q.x : q.y) + (v == 0 ? q.tx : q.ty) * d * 10) / 10.0;
            if (const auto* c = pathOf[k]) {
                text = q.charged ? (q.stub ? "charged (Si stub)" : "charged") : "MIP (no Si)";
                switch (c->end) {
                    case calo::kEndBack:   text += ", exits back"; break;
                    case calo::kEndSide:   text += ", exits side"; break;
                    case calo::kEndShower: text += Form(", shower at %d cm", c->onset); break;
                    default:               text += Form(", stops at %d cm", c->last + 1);
                }
                const auto& w = c->view[v == 0 ? 1 : 0];
                if (!w.empty()) { d = (w.back().layer + 1) * p.layerThick / 10.0; u = w.back().u / 10.0; }
            } else if (q.converted) {
                text = q.stub ? "#gamma, converted (Si stub)" : "#gamma, converted";
            } else if (q.cls == calo::kHadronic) {
                text = "hadronic";
            } else if (!q.photon) {
                text = "near charged track";
            }
            if (k < P.size()) {
                int a = 0, b = 1;
                for (int c = 0; c < calo::kNPid; ++c) if (P[k][c] > P[k][a]) a = c;
                if (a == b) b = 0;
                for (int c = 0; c < calo::kNPid; ++c) if (c != a && P[k][c] > P[k][b]) b = c;
                text += (text.empty() ? "" : "  ") + std::string(Form("%s %.2f | %s %.2f", tex[a], P[k][a], tex[b], P[k][b]));
            }
            if (text.empty()) continue;
            // Beside the track, toward the panel's centre: on it, the text is unreadable.
            auto* T = new TLatex(std::min(d, depth - 18.0) + 0.3, std::clamp(u, -0.9 * h, 0.9 * h) + (u > 0 ? -0.8 : 0.8), text.c_str());
            T->SetTextSize(0.04); T->SetTextColor(ink); T->SetTextAlign(12);
            T->Draw(); keep.push_back(T);
        }
    }
    c.cd(1);
    TLegend leg(0.30, 0.87, 0.93, 0.998);
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.040); leg.SetNColumns(5);
    TLine lr, lj, lx, lg, le, lh, ln, lc;
    TMarker ms(0, 0, 20), mv(0, 0, 29), mt(0, 0, 33);
    lc.SetLineColor(cc); lc.SetLineWidth(3);
    lr.SetLineColor(reco); lr.SetLineWidth(3);
    lj.SetLineColor(ink); lj.SetLineWidth(2);
    lx.SetLineColor(reco); lx.SetLineWidth(1); lx.SetLineStyle(3);
    mv.SetMarkerColor(reco); mv.SetMarkerSize(2.2);
    mt.SetMarkerColor(ink); mt.SetMarkerSize(1.6);
    styleTruth(&lg, 22); styleTruth(&le, 11); styleTruth(&lh, 211); styleTruth(&ln, 2112);
    leg.AddEntry(&lr, "reco photon", "l");
    leg.AddEntry(&lx, "reco photon, extrapolated", "l");
    leg.AddEntry(&mv, "reco pair vertex, m_{#gamma#gamma} [MeV]", "p");
    leg.AddEntry(&lc, "reco charged track (Si + calo)", "l");
    leg.AddEntry(&lj, "reco hadronic / other", "l");
    if (si) leg.AddEntry(&ms, "Si hit", "p");
    leg.AddEntry(&mt, "true vertex", "p");
    leg.AddEntry(&lg, traj ? "true #gamma" : Form("true #gamma (in the CsI: #geq %.0f MeV)", truthMin), "l");
    leg.AddEntry(&le, "true e^{#pm}", "l");
    leg.AddEntry(&lh, "true #mu/#pi/p", "l");
    leg.AddEntry(&ln, "true n (neutral)", "l");
    leg.Draw();
    c.SaveAs((path + ".png").c_str());
    for (auto* o : keep) delete o;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string hits, truthPath, trackerPath, trajPath, pidPath, decaysPath, tag = "run", outDir = "output", plotDir = "plots/tracks";
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
        else if (s == "--decays")             decaysPath = nx();
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
        else if (s == "--frag-efrac")         p.fragEfrac = std::stod(nx());
        else if (s == "--sat-radius-mm")      p.satRadius = std::stod(nx());
        else if (s == "--em-max-tmax-pull")   p.emMaxTmaxPull = std::stod(nx());
        else if (s == "--mip-min-layers")     p.mipMinLayers = std::stoi(nx());
        else if (s == "--mip-min-frac")       p.mipMinFrac = std::stod(nx());
        else if (s == "--mip-min-side-exit")  p.mipMinSideExit = std::stoi(nx());
        else if (s == "--min-veto-planes")    p.minVetoPlanes = std::stoi(nx());
        else if (s == "--min-stub-planes")    p.minStubPlanes = std::stoi(nx());
        else if (s == "--stub-road-mm")       p.stubRoad = std::stod(nx());
        else if (s == "--conv-min-planes")    p.convMinPlanes = std::stoi(nx());
        else if (s == "--path-onset-hard")    p.pathOnsetHard = std::stoi(nx());
        else if (s == "--si-road-mm")         p.siRoad = std::stod(nx());
        else if (s == "--charged-mask-mm")    p.chargedMask = std::stod(nx());
        else if (s == "--n-display")          nDisplay = std::stoi(nx());
        else if (s == "--fid-margin-mm")      p.fidMargin = std::stod(nx());
        else if (s == "--pid")                pidPath = nx();
        else {
            std::fprintf(stderr, "usage: %s --hits PATH --truth PATH [--tag NAME] [--tracker PATH] [--traj PATH] [--decays PATH]\n"
                                 "  [--cell-threshold-MeV 0.1 --seed-MeV 0.5 --min-photon-MeV 5\n"
                                 "   --truth-min-MeV 5 --match-radius-mm 35.3 --max-miss 2 --n-display 20\n"
                                 "   --max-balance 0.8 --max-start-gap 3 --max-depth-gap-mm 40 --track-seed-MeV 2\n"
                                 "   --em-max-start-layer 6 --em-max-tmax-pull 3 --mip-min-layers 8 --mip-min-frac 0.7 --mip-min-side-exit 5\n"
                                 "   --frag-efrac 0.03 --min-veto-planes 4 --min-stub-planes 2 --stub-road-mm 15 --conv-min-planes 3\n"
                                 "   --path-onset-hard 3\n"
                                 "   --charged-mask-mm 70.6 --out-dir output --plot-dir plots/tracks\n"
                                 "   --fid-margin-mm 10 --pid output/pid_table.csv]\n", argv[0]);
            return 1;
        }
    }
    if (hits.empty() || truthPath.empty()) { std::fprintf(stderr, "Error: --hits and --truth required.\n"); return 1; }
    calo::PidTable pid;
    if (!pidPath.empty()) pid = calo::LoadPid(pidPath);

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
                                      r.vx_mm, r.vy_mm, r.vz_mm, r.pdg, r.parentID, r.proc, r.primaryID});
    }
    // Row i = event i, as damsa_alp_inject fires them (it skips zero-weight rows too).
    std::vector<io::AlpDecayRow> decays;
    if (!decaysPath.empty())
        for (const auto& r : io::ReadNTuple<io::AlpDecayRow>(decaysPath)) if (r.weight > 0) decays.push_back(r);
    // Row i must be what event i fired: a re-run subsample with old hits would label everything wrongly.
    if (!decays.empty()) {
        long nPrim = 0, nSame = 0;
        for (const auto& [evt, v] : evTruth)
            for (const auto& t : v)
                if (t.parent == 0 && t.pdg == 22 && evt < static_cast<int>(decays.size())) {
                    const auto& r = decays[evt];
                    ++nPrim;
                    nSame += std::min(std::abs(t.E - r.E1), std::abs(t.E - r.E2)) < 1e-3 * t.E;
                }
        if (nSame < 0.5 * nPrim) {
            std::fprintf(stderr, "Error: %s does not match %s: only %ld of %ld primaries have a row's energy.\n",
                         decaysPath.c_str(), truthPath.c_str(), nSame, nPrim);
            return 1;
        }
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
    if (nDisplay > 0) {
        std::filesystem::create_directories(dispDir);
        if (!decays.empty()) {
            std::filesystem::create_directories(dispDir + "/2gamma");
            std::filesystem::create_directories(dispDir + "/1gamma");
        }
    }
    std::ofstream fo(outDir + "/tracks_" + tag + ".csv"), fp(outDir + "/track_pairs_" + tag + ".csv");
    fo << "eventID,weight,E,x,y,tx,ty,sx,sy,stx,sty,nLayers,start,tmaxLayer,tmaxPull,balance,fixedDir,"
          "nLit,last,medLayerE,layerEcv,peakRatio,meanBars,fracMip,widthEarly,widthLate,cls,vetoPlanes,vetoInfo,photon,"
          "charged,stub,converted,siX,siY,siTx,siTy,xCore,yCore,edgeX,edgeY,fiducial,pathEnd,pathOnset,pathLast,"
          "P_gamma,P_e,P_pi,P_p,P_n,matched,ghost,true_pdg,true_parent,true_E,true_x,true_y,true_tx,true_ty,true_vz,"
          "dtheta_mrad,dx_at_vertex_mm,dy_at_vertex_mm\n";
    fp << "eventID,weight,E1,E2,true_E1,true_E2,vtx_x,vtx_y,vtx_z,vtx_dist,true_vx,true_vy,true_vz,"
          "m_reco_dir,m_fixed_vertex,m_true,fid1,fid2\n";

    gStyle->SetOptStat(0);
    {   // Single-hue (neutral) sequential map for bar energy; lines carry the categorical colours.
        double st[] = {0.0, 1.0}, r[] = {0.94, 0.13}, gg[] = {0.94, 0.13}, b[] = {0.93, 0.13};
        TColor::CreateGradientColorTable(2, st, r, gg, b, 64);
    }

    // [0] all objects, [1] photon ID (EM and not Si-vetoed)
    long nTruthPh = 0, nFound[2] = {0, 0}, nObj[2] = {0, 0}, nMatchedGamma[2] = {0, 0}, nCls[3] = {0, 0, 0},
         nVetoed = 0, nStub = 0, nConv = 0, nDisp[3] = {0, 0, 0};
    const double half = g.caloXY_mm / 2;
    // Primary photons aimed at the calo face, from the decay row.
    auto onFace = [&](const io::AlpDecayRow& r) {
        const double L = p.frontZ - (p.vertexZ + r.decayZ_m * 1000);
        auto hit = [&](double px, double py, double pz) {
            return pz > 0 && std::abs(px / pz * L) < half && std::abs(py / pz * L) < half;
        };
        return hit(r.px1, r.py1, r.pz1) + hit(r.px2, r.py2, r.pz2);
    };
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
            std::vector<calo::TrackerHit> left;
            calo::ApplyVeto(ph, calo::FindSiTracks(it == evTracker.end() ? none : it->second, p, &left), p);
            calo::ApplyStubs(ph, left, p);
            for (std::size_t i = 0; i < ph.size(); ++i) {
                nVetoed += before[i] && !ph[i].photon;
                nStub += ph[i].stub && ph[i].charged;
                nConv += ph[i].converted;
            }
        }
        const auto paths = calo::ChargedPaths(ph, E, T, g, p);
        std::vector<const calo::ChargedPath*> pathOf(ph.size(), nullptr);
        for (const auto& c : paths) pathOf[c.obj] = &c;
        const io::AlpDecayRow* row = evt < static_cast<int>(decays.size()) ? &decays[evt] : nullptr;
        const auto& tr = evTruth[evt];
        std::vector<Prob> P;
        if (!pidPath.empty()) for (const auto& q : ph) P.push_back(calo::PidProb(calo::PidFeatures(q), q.E, pid));

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
            const auto* c = pathOf[i];
            fo << evt << "," << (row ? row->weight : 1.0) << "," << q.E << "," << q.x << "," << q.y << "," << q.tx << "," << q.ty << ","
               << q.sx << "," << q.sy << "," << q.stx << "," << q.sty << "," << q.nLayers << ","
               << q.start << "," << q.tmaxLayer << "," << q.tmaxPull << "," << q.balance << ","
               << q.fixedDir << "," << q.nLit << "," << q.last << "," << q.medLayerE << "," << q.layerEcv << ","
               << q.peakRatio << "," << q.meanBars << "," << q.fracMip << "," << q.widthEarly << "," << q.widthLate << ","
               << q.cls << "," << q.vetoPlanes << "," << q.vetoInfo << "," << q.photon << ","
               << q.charged << "," << q.stub << "," << q.converted << "," << q.siX << "," << q.siY << "," << q.siTx << "," << q.siTy << ","
               << q.xCore << "," << q.yCore << "," << q.edgeX << "," << q.edgeY << "," << q.fiducial << ","
               << (c ? c->end : -1) << "," << (c ? c->onset : -1) << "," << (c ? c->last : -1) << ",";
            for (int c = 0; c < calo::kNPid; ++c) fo << (P.empty() ? "" : std::to_string(P[i][c])) << ",";
            fo << (m >= 0) << "," << ghost;
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
            // A decay row fires one pair per event, so different primaries are that pair even if one converted.
            const bool fromRow = row && u.primary != w.primary;
            if (!fromRow && (u.pdg != 22 || w.pdg != 22 || u.parent != w.parent ||
                             std::abs(u.vz - w.vz) > 1e-3 || std::abs(u.vx - w.vx) > 1e-3)) continue;
            const double zTrue = row ? p.vertexZ + row->decayZ_m * 1000 : u.vz;
            const double mTrue = row ? RowMass(*row) : Mass(u.E, u.tx, u.ty, w.E, w.tx, w.ty);
            const auto v = calo::ClosestApproach(ph[i].x, ph[i].y, ph[i].tx, ph[i].ty,
                                                 ph[j].x, ph[j].y, ph[j].tx, ph[j].ty, p.frontZ);
            const double L = p.frontZ - p.vertexZ;
            fp << evt << "," << (row ? row->weight : 1.0) << "," << ph[i].E << "," << ph[j].E << "," << u.E << "," << w.E << ","
               << v.x << "," << v.y << "," << v.z << "," << v.dist << ","
               << (row ? 0.0 : u.vx) << "," << (row ? 0.0 : u.vy) << "," << zTrue << ","
               << Mass(ph[i].E, ph[i].tx, ph[i].ty, ph[j].E, ph[j].tx, ph[j].ty) << ","
               << calo::Mgg(ph[i].E, ph[i].x, ph[i].y, ph[j].E, ph[j].x, ph[j].y, L) << ","
               << mTrue << "," << ph[i].fiducial << "," << ph[j].fiducial << "\n";
        }

        // Displays: with --traj, only events that have trajectories.
        const auto tj = evTraj.find(evt);
        const bool haveTraj = tj != evTraj.end();
        // With a decay row: half the displays each for 2 and for 1 photons aimed at the face.
        const int topo = row ? onFace(*row) : 0;
        const bool wanted = row ? topo > 0 && nDisp[topo] < nDisplay / 2 : nDisp[0] < nDisplay;
        if (wanted && !ph.empty() && (trajPath.empty() || haveTraj)) {
            static const std::vector<calo::TrackerHit> noHits;   // with --tracker, always draw the Si region
            const auto si = evTracker.find(evt);
            const std::string info = row ? Form(" -- parent E = %.0f MeV, %d/2 #gamma aimed at the face", row->E1 + row->E2, topo) : "";
            Display(evt, E, ph, P, tr, haveTraj ? &tj->second : nullptr,
                    trackerPath.empty() ? nullptr : si == evTracker.end() ? &noHits : &si->second, paths, row, g, p, truthMin,
                    PrimaryLabel(tag), info, dispDir + (row ? "/" + std::to_string(topo) + "gamma" : "") + "/evt_" + std::to_string(evt));
            ++nDisp[topo];
        }
    }
    auto frac = [](long a, long b) { return b ? double(a) / double(b) : 0.0; };
    std::printf("[%s] events %zu | objects %ld: EM %ld, MIP %ld, hadronic %ld; Si-vetoed EM %ld (stub %ld), converted %ld\n",
                tag.c_str(), evHits.size(), nObj[0], nCls[calo::kEM], nCls[calo::kMIP], nCls[calo::kHadronic], nVetoed, nStub, nConv);
    std::printf("[%s] truth photons >= %.0f MeV in aperture %ld | all objects: found %.3f, gamma-purity %.3f"
                " | photon ID: %ld photons, found %.3f, gamma-purity %.3f\n",
                tag.c_str(), truthMin, nTruthPh, frac(nFound[0], nTruthPh), frac(nMatchedGamma[0], nObj[0]),
                nObj[1], frac(nFound[1], nTruthPh), frac(nMatchedGamma[1], nObj[1]));
    return 0;
}
