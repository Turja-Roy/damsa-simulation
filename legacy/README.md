# legacy/

Python (and one C++ file) superseded by the C++ migration. **Kept, not deleted**:
these are the reference implementations every ported stage was cross-checked
against, and `tools/alp_signal_xcheck.py` still imports from here.

Nothing in `jobs/` or `scripts/*.sh` calls these any more except the two
cross-checks in `jobs/01_verify.sbatch`, which deliberately run the Python side
to diff against the C++.

## What replaced what

| legacy | replacement |
|---|---|
| `python/pipeline/alp_signal_pipeline.py` | `damsa_alp_signal` |
| `python/pipeline/pileup_overlay.py` | `damsa_pileup` |
| `python/local/merge_library.py` | `damsa_merge` |
| `python/optimization/joint_pareto_scan.py` | `damsa_joint_scan` |
| `python/optimization/combine_pareto.py` | `damsa_combine_pareto` |
| `python/analysis/project_target_length.py` | `damsa_target_projection` |
| `python/visualization/final_report.py` | `damsa_report` |
| `python/visualization/alplib_signal_plots.py`, `root_config_plots.py` | `damsa_signal_plots` |
| `python/visualization/pi0_vertices_g4view.py`, `alp_signal_g4view.py` | `damsa_eventview` |
| `python/analysis/print_weight_breakdown.py`, `validate_opening_angles.py`, `angle_validation/angle_diagnostics.py` | `damsa_validate` |
| `cpp/subsample_alp_csv.cpp` | `damsa_subsample` (reads/writes TTree too) |

`python/optimization/fast_pareto_scan.py` and `projected_pareto_scan.py` were
already broken before the migration — both import `scripts.alplib.alplib_signal_plots`,
a path that has never existed — so they were never ported. `auto_coupling()` was
the only live thing in them and now lives in `src/alp/scan.h`.

## Known bug in the scan scripts

`joint_pareto_scan.py`, `project_target_length.py` and `final_report.py` all call
`flux_obj.propagate(decay_width)`. `FluxPrimakoffIsotropic.propagate`'s first
parameter is `new_coupling`, not a width (`alplib/fluxes.py:151`), so the decay
width is used as a coupling. The C++ ports reproduce this by default so the two
sides stay diffable, and take `--fix-weights` to use the intended physics.
See the commit message for `feat(scan)`.

## Still live, not legacy

`scripts/optimization/bayesian_optimization.py`, `run_optimization.py`,
`scripts/runner/{geant4_runner,objectives}.py` and
`scripts/visualization/visualization.py` are the BoTorch/pymoo branch. It was
deliberately left in Python and has no C++ replacement, so it stays under
`scripts/`. Note it cannot currently run: it imports `optimization_problem`,
which exists nowhere in the repo, swallowed by a `try/except`.
