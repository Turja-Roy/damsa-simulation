// Detector event display: gamma-gamma decay vertices with their photon tracks,
// in 3D and in side projection.
// Replaces scripts/visualization/pi0_vertices_g4view.py and alp_signal_g4view.py.
//
//   ./build/damsa_eventview --pi0 output/pi0_decays.root      --out-dir plots/eventview
//   ./build/damsa_eventview --alp output/alp_decay_photons_ma100MeV.root
//
// Geometry mirrors construction.cpp defaults; override with the flags if the
// scan moved something. z is the beam axis, in mm, world frame.

#include "damsa_io.h"
#include "plotting.h"

#include <TBox.h>
#include <TCanvas.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPolyLine3D.h>
#include <TPolyMarker3D.h>
#include <TView3D.h>
#include <TH2F.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace io = damsa::io;

namespace {

struct Geometry {
    double buildStart = -500.0, targetLen = 100.0, targetXY = 50.0;
    double vdcLen = 300.0, magnetLen = 120.0, caloLen = 440.0, caloXY = 120.0;

    double tgt0() const { return buildStart; }
    double tgt1() const { return buildStart + targetLen; }
    double vdc0() const { return tgt1(); }
    double vdc1() const { return tgt1() + vdcLen; }
    double mag0() const { return vdc1(); }
    double mag1() const { return vdc1() + magnetLen; }
    double calo0() const { return mag1(); }
    double calo1() const { return mag1() + caloLen; }
    double caloHalf() const { return caloXY / 2.0; }
};

struct Track {
    double vx, vy, vz;        // vertex [mm]
    double px, py, pz;        // direction (unit)
    int    which;             // 1 or 2, for colour
    bool   reachesCalo;
};

// Where to stop drawing: the calo face if the track gets there, else a stub.
void Endpoint(const Track& t, const Geometry& g, double stub,
              double& ex, double& ey, double& ez)
{
    if (t.pz > 0) {
        const double dz = g.calo0() - t.vz;
        if (dz > 0) {
            const double s = dz / t.pz;
            ex = t.vx + s * t.px; ey = t.vy + s * t.py; ez = g.calo0();
            return;
        }
    }
    ex = t.vx + stub * t.px; ey = t.vy + stub * t.py; ez = t.vz + stub * t.pz;
}

void DrawSpans(const Geometry& g)
{
    struct Span { const char* name; double z0, z1; Color_t col; };
    const Span spans[] = {
        {"W target",     g.tgt0(),  g.tgt1(),  kGray + 2},
        {"VDC",          g.vdc0(),  g.vdc1(),  kAzure - 9},
        {"Magnet+trk",   g.mag0(),  g.mag1(),  kOrange - 3},
        {"CsI ECAL",     g.calo0(), g.calo1(), kMagenta - 4},
    };
    for (const auto& s : spans) {
        auto* b = new TBox(s.z0, -g.caloHalf() * 1.6, s.z1, g.caloHalf() * 1.6);
        b->SetFillColorAlpha(s.col, 0.25);
        b->SetLineColor(s.col);
        b->Draw("l");
        auto* t = new TLatex(0.5 * (s.z0 + s.z1), g.caloHalf() * 1.45, s.name);
        t->SetTextAlign(22); t->SetTextSize(0.028); t->SetTextColor(s.col);
        t->Draw();
    }
}

}  // namespace

int main(int argc, char** argv)
{
    std::string pi0Path, alpPath, outDir = "plots/eventview";
    long maxTracks = 1500;
    std::uint64_t seed = 12345;
    Geometry g;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (s == "--pi0")         pi0Path = nx();
        else if (s == "--alp")         alpPath = nx();
        else if (s == "--out-dir")     outDir = nx();
        else if (s == "--max-tracks")  maxTracks = std::stol(nx());
        else if (s == "--seed")        seed = std::stoull(nx());
        else if (s == "--vdc-length")  g.vdcLen = std::stod(nx());
        else if (s == "--calo-xy")     g.caloXY = std::stod(nx());
        else if (s == "--target-len")  g.targetLen = std::stod(nx());
        else {
            std::fprintf(stderr,
                "Usage: %s (--pi0 FILE | --alp FILE) [--out-dir DIR]\n"
                "  --max-tracks N   subsample for legibility (default 1500)\n"
                "  --vdc-length MM  --calo-xy MM  --target-len MM\n", argv[0]);
            return 1;
        }
    }
    if (pi0Path.empty() && alpPath.empty()) {
        std::fprintf(stderr, "Error: pass --pi0 or --alp\n"); return 1;
    }

    // ── Collect tracks ──────────────────────────────────────────────────────
    std::vector<Track> tracks;
    std::string tag;

    if (!pi0Path.empty()) {
        tag = "pi0";
        for (const auto& d : io::ReadNTuple<io::Pi0Row>(pi0Path)) {
            tracks.push_back({d.vx_mm, d.vy_mm, d.vz_mm, d.px1, d.py1, d.pz1, 1,
                              d.gamma1AtCalo == 1});
            tracks.push_back({d.vx_mm, d.vy_mm, d.vz_mm, d.px2, d.py2, d.pz2, 2,
                              d.gamma2AtCalo == 1});
        }
    } else {
        tag = "alp";
        // decay_z_m is measured from the target CENTRE, which is where
        // damsa_alp_inject fires from (gALPVertexZ_cm default -45 cm).
        const double vertexZ0_mm = g.tgt0() + g.targetLen / 2.0;
        for (const auto& r : io::ReadNTuple<io::AlpDecayRow>(alpPath)) {
            if (r.weight <= 0) continue;
            const double vz = vertexZ0_mm + r.decayZ_m * 1000.0;
            const double m1 = std::sqrt(r.px1 * r.px1 + r.py1 * r.py1 + r.pz1 * r.pz1);
            const double m2 = std::sqrt(r.px2 * r.px2 + r.py2 * r.py2 + r.pz2 * r.pz2);
            if (m1 < 1e-12 || m2 < 1e-12) continue;
            tracks.push_back({0, 0, vz, r.px1 / m1, r.py1 / m1, r.pz1 / m1, 1, false});
            tracks.push_back({0, 0, vz, r.px2 / m2, r.py2 / m2, r.pz2 / m2, 2, false});
        }
    }
    if (tracks.empty()) { std::fprintf(stderr, "Error: no tracks to draw\n"); return 1; }
    std::printf("[view] %zu tracks from %s\n", tracks.size(),
                pi0Path.empty() ? alpPath.c_str() : pi0Path.c_str());

    // Subsample for legibility; thousands of overlapping lines show nothing.
    if (static_cast<long>(tracks.size()) > maxTracks) {
        std::mt19937_64 rng(seed);
        std::shuffle(tracks.begin(), tracks.end(), rng);
        tracks.resize(maxTracks);
        std::printf("[view] subsampled to %ld tracks\n", maxTracks);
    }

    std::filesystem::create_directories(outDir);
    SetPublicationStyle();
    const double stub = 250.0;

    // ── Side view: z along the beam, y transverse ──────────────────────────
    {
        auto* c = new TCanvas("c_side", "", 1100, 600);
        auto* frame = new TH2F("frame", Form("%s decay vertices, side view;z [mm];y [mm]",
                                             tag.c_str()),
                               10, g.tgt0() - 50, g.calo1() + 50,
                               10, -g.caloHalf() * 1.6, g.caloHalf() * 1.6);
        frame->SetStats(0);
        frame->Draw();
        DrawSpans(g);

        // Calorimeter aperture, the acceptance boundary the tracks must land in.
        for (double sgn : {-1.0, 1.0}) {
            auto* l = new TLine(g.calo0(), sgn * g.caloHalf(), g.calo1(), sgn * g.caloHalf());
            l->SetLineColor(kMagenta - 4); l->SetLineStyle(2); l->Draw();
        }

        for (const auto& t : tracks) {
            double ex, ey, ez; Endpoint(t, g, stub, ex, ey, ez);
            auto* l = new TLine(t.vz, t.vy, ez, ey);
            l->SetLineColorAlpha(t.which == 1 ? kRed + 1 : kOrange + 7, 0.35);
            l->Draw();
        }
        SaveCanvas(c, outDir + "/" + tag + "_vertices_side");
        delete c;
    }

    // ── 3D view ─────────────────────────────────────────────────────────────
    {
        auto* c = new TCanvas("c_3d", "", 900, 800);
        auto* view = TView3D::CreateView(1);
        const double hxy = g.caloHalf() * 1.6;
        view->SetRange(g.tgt0() - 50, -hxy, -hxy, g.calo1() + 50, hxy, hxy);

        for (const auto& t : tracks) {
            double ex, ey, ez; Endpoint(t, g, stub, ex, ey, ez);
            auto* p = new TPolyLine3D(2);
            p->SetPoint(0, t.vz, t.vx, t.vy);
            p->SetPoint(1, ez, ex, ey);
            p->SetLineColorAlpha(t.which == 1 ? kRed + 1 : kOrange + 7, 0.35);
            p->Draw();
        }
        // Decay vertices themselves.
        auto* m = new TPolyMarker3D(tracks.size() / 2);
        for (std::size_t i = 0, k = 0; i + 1 < tracks.size(); i += 2, ++k)
            m->SetPoint(k, tracks[i].vz, tracks[i].vx, tracks[i].vy);
        m->SetMarkerColor(kCyan + 1); m->SetMarkerStyle(20); m->SetMarkerSize(0.3);
        m->Draw();

        view->ShowAxis();
        SaveCanvas(c, outDir + "/" + tag + "_vertices_3d");
        delete c;
    }

    // ── Vertex z distribution: where the decays actually happen ────────────
    {
        auto* c = new TCanvas("c_vz", "", 800, 600);
        auto* h = new TH1D("h_vz", Form("%s decay vertex z;z [mm];decays", tag.c_str()),
                           120, g.tgt0() - 50, g.calo1() + 50);
        for (std::size_t i = 0; i < tracks.size(); i += 2) h->Fill(tracks[i].vz);
        StyleHist1D(h, kAzure + 2, "z [mm]", "decays");
        h->Draw("HIST");
        DrawSpans(g);
        h->Draw("HIST SAME");
        SaveCanvas(c, outDir + "/" + tag + "_vertex_z");
        delete c;
    }

    std::printf("[view] wrote %s/%s_vertices_{side,3d}.png and %s_vertex_z.png\n",
                outDir.c_str(), tag.c_str(), tag.c_str());
    return 0;
}
