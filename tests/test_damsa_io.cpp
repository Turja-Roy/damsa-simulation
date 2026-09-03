// Self-check for damsa_io.h. Build: see the damsa_io_test target in CMakeLists.
// Covers the round-trip that Phase 1 depends on: what goes into an RNTuple must
// come back bit-identical, or the CSV-vs-RNTuple cross-check is meaningless.

#include "damsa_io.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace damsa::io;

static void TestParticleRoundTrip()
{
    const std::string path = "test_out/particles.root";
    std::vector<ParticleRow> in;
    for (int i = 0; i < 1000; ++i) {
        ParticleRow r;
        r.pdg = (i % 3 == 0) ? 22 : 2112;
        r.trackID = i;  r.eventID = i / 10;
        r.energy_MeV = 1.0 + i * 0.12345678901234;   // enough digits to catch float truncation
        r.time_ns = i * 0.001;
        r.x_mm = -i * 0.5; r.y_mm = i * 0.25; r.z_mm = 20.0;
        r.px = 0.01 * i; r.py = -0.02 * i; r.pz = 0.998;
        r.weight = 1.0;
        in.push_back(r);
    }
    WriteNTuple(path, in);

    const auto out = ReadNTuple<ParticleRow>(path);
    assert(out.size() == in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        assert(out[i].pdg == in[i].pdg);
        assert(out[i].trackID == in[i].trackID);
        assert(out[i].eventID == in[i].eventID);
        // Exact: both sides are float64. A tolerance here would hide a float32 regression.
        assert(out[i].energy_MeV == in[i].energy_MeV);
        assert(out[i].x_mm == in[i].x_mm);
        assert(out[i].pz == in[i].pz);
    }
    std::printf("  particle round-trip: %zu rows exact\n", out.size());
}

static void TestAlpDecayStreaming()
{
    const std::string path = "test_out/alp.root";
    {
        NTupleWriter<AlpDecayRow> w(path);
        for (int i = 0; i < 500; ++i) {
            AlpDecayRow r;
            r.E1 = 0.29 + i; r.px1 = 0.143; r.py1 = -0.245; r.pz1 = -0.058;
            r.E2 = 1.21 + i; r.px2 = -0.143; r.py2 = 0.245; r.pz2 = 1.176;
            r.weight = 5.3579067392e10;
            r.decayZ_m = 0.448 + i * 1e-4;
            w.Fill(r);
        }
        assert(w.Entries() == 500);
    }
    const auto out = ReadNTuple<AlpDecayRow>(path);
    assert(out.size() == 500);
    assert(out[499].weight == 5.3579067392e10);
    assert(out[0].px2 == -out[0].px1);
    std::printf("  alp decay streaming: %zu rows exact\n", out.size());
}

static void TestCsv()
{
    const std::string path = "test_out/t.csv";
    { std::ofstream f(path);
      f << "# Beam mode: LESALaser\n"
        << "# Beam current [A]: 1.125247e-08\n"
        << "pdg,energy_MeV,pz\n"
        << "22,4.32,0.998\n"
        << "\n"                       // blank line must be skipped
        << "2112,150.5,0.1\n"; }

    const auto csv = ReadCsv(path);
    assert(csv.size() == 2);
    assert(csv.has("energy_MeV") && !csv.has("nope"));
    assert(csv.get(0, "energy_MeV") == 4.32);
    assert(csv.get(1, "pdg") == 2112);
    assert(csv.get(0, "nope", -1.0) == -1.0);          // fallback
    assert(csv.comment("Beam mode") == "LESALaser");
    assert(std::abs(std::stod(csv.comment("Beam current [A]")) - 1.125247e-08) < 1e-20);
    std::printf("  csv reader: ok\n");
}

static void TestBremsFlux()
{
    const std::string path = "test_out/flux.csv";
    { std::ofstream f(path);
      f << "# Bremsstrahlung photon flux INSIDE target\n"
        << "# Beam mode: LESALaser\n"
        << "# Beam current [A]: 1.125247e-08\n"
        << "# Format: energy_MeV, rate_per_second\n"
        << "1.500,3.0e+10\n"
        << "2.500,1.0e+10\n"
        << "0.000,5.0e+09\n"      // e <= 0  -> dropped, matching the Python
        << "3.500,0.0\n"; }       // rate<=0 -> dropped, matching the Python

    const auto flux = ReadBremsFlux(path);
    assert(flux.size() == 2);
    assert(flux.beamMode == "LESALaser");
    assert(std::abs(flux.beamCurrent_A - 1.125247e-08) < 1e-20);
    assert(std::abs(flux.totalRate() - 4.0e10) < 1.0);
    std::printf("  brems flux reader: %zu bins, mode=%s\n",
                flux.size(), flux.beamMode.c_str());
}

static void TestEnsureParentDir()
{
    // The bug this replaces: mkdir("output", 0755) is not recursive, so a
    // nested prefix silently produced no file at all (CLAUDE.md gotcha).
    const std::string nested = "test_out/a/b/c/deep.root";
    WriteNTuple(nested, std::vector<AlpDecayRow>{AlpDecayRow{}});
    assert(std::filesystem::exists(nested));
    std::printf("  nested path creation: ok\n");
}

int main()
{
    std::filesystem::remove_all("test_out");
    std::printf("damsa_io self-check\n");
    TestParticleRoundTrip();
    TestAlpDecayStreaming();
    TestCsv();
    TestBremsFlux();
    TestEnsureParentDir();
    std::filesystem::remove_all("test_out");
    std::printf("all passed\n");
    return 0;
}
