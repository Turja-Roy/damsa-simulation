#!/usr/bin/env python3
"""Merge per-chunk DAMSA library outputs into the canonical files that
AUDIT.md steps 2-6 consume.

Three files, three different merge rules (see AUDIT.md / CLAUDE.md):

  alplib_brems_flux.csv   binned "energy_MeV,rate_per_second" where
                          rate = count * (I/e) / N_electrons.  Cannot be
                          concatenated: reconstruct integer counts per bin
                          from each chunk, sum them, re-normalize by the
                          TOTAL electrons.  (Reproduces FluxData.h
                          WriteAlplibBremsFlux exactly.)

  pi0_decays.csv          raw per-pi0 rows; eventID is column 0.
  calo_face_particles.csv raw per-particle rows; eventID is the LAST column.
                          Concatenate, but OFFSET eventID per chunk so the
                          overlay's per-electron grouping (hash on eventID)
                          never collapses two different electrons that both
                          happened to be event 0 in their chunk.

Run from repo root after run_library_local.sh.
"""
import argparse, csv, glob, os, re, sys

OUT = "output"
E_CHARGE = 1.602176634e-19  # C, matches FluxData.h

def chunk_prefixes(nchunks):
    return [f"chunk{i:02d}_" for i in range(nchunks)]


def parse_flux_header(path):
    """Return (beam_current_A, n_primaries, header_lines) from a chunk flux CSV."""
    cur = nprim = None
    header = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            header.append(line.rstrip("\n"))
            m = re.search(r"Beam current \[A\]:\s*([0-9.eE+-]+)", line)
            if m:
                cur = float(m.group(1))
            m = re.search(r"Primary electrons simulated:\s*([0-9.eE+-]+)", line)
            if m:
                nprim = float(m.group(1))
    if cur is None or nprim is None:
        sys.exit(f"ERROR: could not parse current/primaries from {path}")
    return cur, nprim, header


def merge_flux(prefixes):
    files = [os.path.join(OUT, p + "alplib_brems_flux.csv") for p in prefixes]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        sys.exit("ERROR: no per-chunk alplib_brems_flux.csv found")

    counts = {}            # bin_center_MeV -> summed integer count
    total_electrons = 0.0
    beam_current = None
    template_header = None

    for path in files:
        cur, nprim, header = parse_flux_header(path)
        beam_current = cur
        template_header = header
        total_electrons += nprim
        eps = cur / E_CHARGE                 # electrons per second
        scale = eps / nprim                  # this chunk's photons->rate factor
        with open(path) as f:
            for row in f:
                if row.startswith("#") or row.startswith("energy") or not row.strip():
                    continue
                e_str, rate_str = row.split(",")
                center = float(e_str)
                rate = float(rate_str)
                cnt = int(round(rate / scale)) # recover raw photon count
                counts[center] = counts.get(center, 0) + cnt

    eps = beam_current / E_CHARGE
    scale = eps / total_electrons
    outpath = os.path.join(OUT, "alplib_brems_flux.csv")
    with open(outpath, "w") as f:
        f.write("# Bremsstrahlung photon flux INSIDE target for alplib Primakoff input\n")
        f.write("# MERGED from %d chunks by merge_library.py\n" % len(files))
        # keep the beam-mode / current lines from a chunk so the pipeline
        # header-reader still finds them
        for h in template_header:
            if ("Beam mode" in h) or ("Beam current" in h):
                f.write(h + "\n")
        f.write("# Primary electrons simulated: %d\n" % int(total_electrons))
        f.write("# Scale factor: %g\n" % scale)
        f.write("# Format: energy_MeV, rate_per_second\n#\n")
        for center in sorted(counts):
            f.write("%.3f,%.6e\n" % (center, counts[center] * scale))
    print(f"[flux] {len(files)} chunks, {int(total_electrons)} electrons, "
          f"{len(counts)} bins -> {outpath}")


def merge_raw(prefixes, basename, eventid_col):
    """Concatenate raw per-particle CSVs, offsetting eventID per chunk."""
    files = [(p, os.path.join(OUT, p + basename)) for p in prefixes]
    files = [(p, f) for p, f in files if os.path.exists(f)]
    if not files:
        print(f"[raw ] no chunk files for {basename}, skipping")
        return

    outpath = os.path.join(OUT, basename)
    offset = 0
    header_written = False
    rows_out = 0
    with open(outpath, "w", newline="") as out:
        w = csv.writer(out)
        for _, path in files:
            local_max = -1
            with open(path, newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                if header is None:
                    continue
                if not header_written:
                    w.writerow(header)
                    header_written = True
                # resolve eventID column index by name if present
                idx = eventid_col
                if "eventID" in header:
                    idx = header.index("eventID")
                for row in r:
                    if not row:
                        continue
                    ev = int(row[idx])
                    local_max = max(local_max, ev)
                    row[idx] = str(ev + offset)
                    w.writerow(row)
                    rows_out += 1
            if local_max >= 0:
                offset += local_max + 1   # next chunk starts past this one
    print(f"[raw ] {basename}: {len(files)} chunks, {rows_out} rows, "
          f"eventID span 0..{offset-1} -> {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchunks", type=int, required=True)
    args = ap.parse_args()
    prefixes = chunk_prefixes(args.nchunks)

    merge_flux(prefixes)
    merge_raw(prefixes, "pi0_decays.csv", eventid_col=0)
    merge_raw(prefixes, "calo_face_particles.csv", eventid_col=-1)


if __name__ == "__main__":
    main()
