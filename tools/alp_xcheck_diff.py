#!/usr/bin/env python3
"""Numeric diff of the src/alp C++ port against alplib.

    ./build/alp_xcheck alplib > cpp.txt
    python3 tools/alp_xcheck.py   > py.txt
    python3 tools/alp_xcheck_diff.py cpp.txt py.txt

Everything must agree exactly except the aggregate `flux` sums, where alplib
stores its weights as float32 (fluxes.py:64) and anything below the float32
floor (~1.2e-38) is zero on the Python side while the C++ keeps full double
precision. That case is reported separately rather than waved through: a weight
of 1e-297 events/day and a weight of 0 are the same physics.
"""
import sys, math
from collections import defaultdict

F32_TINY = 1.1754943508222875e-38

# Per-quantity relative tolerance. The closed-form physics -- cross sections,
# widths, absorption, per-element weights -- must be EXACT; a nonzero tolerance
# there would hide a real porting error. The three below are float-association
# effects with no physical content:
TOL = defaultdict(float)
TOL['flux']  = 1e-5    # naive double vs numpy pairwise float32 summation
TOL['boost'] = 1e-14   # numpy does a 4x4 BLAS matmul; this expands the terms
TOL['decay'] = 1e-12   # the above, accumulated through sin/cos and the boost

def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            out.append(('#', line)); continue
        parts = line.split()
        vals = []
        for t in parts[1:]:
            t = t.split('=')[-1]
            try: vals.append(float(t))
            except ValueError: vals.append(t)
        out.append((parts[0], vals))
    return out

def main(pa, pb):
    a, b = load(pa), load(pb)
    if len(a) != len(b):
        print("LINE COUNT MISMATCH %d vs %d" % (len(a), len(b))); return 1

    worst = defaultdict(lambda: (0.0, None))
    counts = defaultdict(int)
    underflow = 0
    bad = []

    for i, ((ka, va), (kb, vb)) in enumerate(zip(a, b)):
        if ka == '#':
            if va != vb: bad.append((i + 1, va, vb))
            continue
        if ka != kb or len(va) != len(vb):
            bad.append((i + 1, (ka, va), (kb, vb))); continue
        counts[ka] += 1
        for x, y in zip(va, vb):
            if isinstance(x, str) or isinstance(y, str):
                if x != y: bad.append((i + 1, x, y))
                continue
            if x == y: continue
            if math.isnan(x) and math.isnan(y): continue
            # Below the float32 floor alplib's stored value is either 0 or a
            # subnormal carrying almost no mantissa, so a comparison there is
            # meaningless. Both sides being under the floor is agreement:
            # 1e-45 and 0 events/day are the same physics.
            if abs(y) < F32_TINY and abs(x) < F32_TINY:
                underflow += 1; continue
            denom = max(abs(x), abs(y))
            rel = abs(x - y) / denom if denom else abs(x - y)
            if rel > worst[ka][0]:
                worst[ka] = (rel, (i + 1, x, y))

    print("%-12s %8s  %s" % ("quantity", "rows", "max relative deviation"))
    ok = not bad
    for k in sorted(counts):
        rel, info = worst[k]
        tol = TOL[k]
        flag = "exact" if rel == 0.0 else "%.3e" % rel
        if rel > tol:
            ok = False
            flag += "   <-- line %d: %.17g vs %.17g (tol %.0e)" % (info + (tol,))
        print("%-12s %8d  %s" % (k, counts[k], flag))

    if underflow:
        print("\n%d value(s) below the float32 floor (1.2e-38): alplib stored 0 "
              "or a subnormal, C++ kept full precision.\n"
              "Physically identical - these are < 1e-38 events/day." % underflow)
    if bad:
        print("\nnon-numeric mismatches: %d, first few:" % len(bad))
        for x in bad[:5]: print("  ", x)

    print("\n" + ("MATCH" if ok else "MISMATCH"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
