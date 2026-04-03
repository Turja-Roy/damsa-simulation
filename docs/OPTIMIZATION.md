# DAMSA Optimization Framework

A comprehensive multi-objective optimization framework for the DAMSA (DArk Messenger Searches at an Accelerator) simulation. This framework optimizes detector geometry parameters to maximize ALP signal while minimizing background contributions.

## Overview

### What This Does

The optimization framework finds optimal values for:
- **Target thickness (Z)**: Depth along beam axis
- **Target cross-section (XY)**: Width and height (always square: X = Y)
- **Vacuum chamber length (gap)**: Length of the vacuum decay region
- **Timing window**: Start (t_min) and end (t_max) times

**Important Notes:**
- The target always has a **square cross-section** (target_x = target_y)
- The **gap parameter** represents the vacuum decay chamber length only - the magnet region length stays constant. Changing vacuum chamber length adjusts positions of downstream components (trackers, calorimeter)

To maximize these objectives:
- **Signal rate**: ALP decays detected (MAXIMIZE)
- **Neutron background**: Neutrons in timing window (MINIMIZE)  
- **Photon background**: EM punch-through photons (MINIMIZE)
- **Photon separability**: How well backgrounds can be separated from signal (MAXIMIZE)

### Key Features

1. **Hybrid Optimization**: Combines Bayesian optimization (BoTorch) for efficient exploration with NSGA-II (pymoo) for Pareto front refinement
2. **Physics-Based Objectives**: Signal from alplib, backgrounds from Geant4
3. **Caching**: Results cached to avoid redundant simulations
4. **Visualization**: Comprehensive plotting and analysis tools
5. **Flexible Configuration**: Command-line interface with full control

---

## Installation

### Python Dependencies

```bash
# Install core dependencies
pip install numpy scipy pandas matplotlib seaborn

# For Bayesian optimization (recommended)
pip install torch botorch

# For NSGA-II/III optimization (recommended)  
pip install pymoo
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### Geant4

Geant4 must be installed separately. See [Geant4 Installation Guide](https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/index.html).

Build the DAMSA simulation:
```bash
mkdir -p build && cd build
cmake ..
make
```

---

## Quick Start

### 1. Test with Mock Simulation

```bash
cd scripts
python run_optimization.py --mode hybrid --n-bo-iter 5 --n-nsga-gen 10 --quiet
```

### 2. Run Full Optimization

```bash
# Hybrid mode (recommended)
python scripts/run_optimization.py \
    --mode hybrid \
    --n-bo-iter 50 \
    --n-nsga-gen 100 \
    --n-events 1000

# BO only (smaller budget)
python scripts/run_optimization.py \
    --mode bo-only \
    --n-bo-iter 100 \
    --n-events 1000

# NSGA-II only
python scripts/run_optimization.py \
    --mode nsga-only \
    --n-nsga-gen 200 \
    --n-events 1000
```

### 3. Analyze Results

```bash
python scripts/visualization.py optimization_results/YYYYMMDD_HHMMSS/
```

---

## Usage Guide

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Optimization mode: `hybrid`, `bo-only`, `nsga-only` | `hybrid` |
| `--n-bo-init` | Initial BO samples | 20 |
| `--n-bo-iter` | BO iterations | 50 |
| `--n-nsga-pop` | NSGA population size | 50 |
| `--n-nsga-gen` | NSGA generations | 100 |
| `--axion-mass` | ALP mass in MeV | 100 |
| `--coupling` | ALP-photon coupling | 1e-4 |
| `--exposure` | Exposure time in days | 30 |
| `--n-events` | Events per simulation | 1000 |
| `--no-mock` | Use real Geant4 instead of mock | False |
| `--objectives` | Objectives to optimize | signal, neutron, photon_bkg, separability |
| `--output-dir` | Output directory | auto-generated |
| `--quiet` | Reduce verbosity | False |

### Example Commands

#### Different Physics Scenarios

```bash
# Light ALP (50 MeV)
python scripts/run_optimization.py --axion-mass 50 --coupling 1e-5

# Heavy ALP (200 MeV)
python scripts/run_optimization.py --axion-mass 200 --coupling 1e-4

# Longer exposure
python scripts/run_optimization.py --exposure 90
```

#### Different Optimization Budgets

```bash
# Quick test (5 minutes)
python scripts/run_optimization.py --mode nsga-only --n-nsga-gen 20

# Medium run (1 hour)
python scripts/run_optimization.py --mode hybrid --n-bo-iter 30 --n-nsga-gen 50

# Production (several hours)
python scripts/run_optimization.py --mode hybrid --n-bo-iter 100 --n-nsga-gen 200 --n-events 5000
```

---

## Output Files

### Directory Structure

```
optimization_results/YYYYMMDD_HHMMSS/
├── pareto_front.npz          # Pareto front (X, F, G)
├── pareto_solutions.csv       # Solutions in CSV format
├── evaluation_history.json   # All evaluations
├── summary.json              # Summary statistics
├── config.json               # Configuration used
└── phase1_bo/                # (hybrid mode only)
    ├── checkpoint_*.json
    ├── pareto_front.npz
    └── summary.json
```

### Analysis Output

```
optimization_results/YYYYMMDD_HHMMSS/analysis/
├── pareto_2d_*.png           # 2D projections
├── pareto_3d.png             # 3D visualization
├── pareto_matrix.png         # Pairwise projections
├── parallel_coords.png       # Parallel coordinates
├── variable_distributions.png
├── convergence.png           # Convergence plot
├── objective_correlations.csv
├── ranked_solutions.csv     # Ranked by composite FoM
└── summary_stats.json
```

---

## Understanding the Results

### Decision Variables

| Variable | Range | Description |
|----------|-------|-------------|
| target_z | 5-20 cm | Target thickness along beam |
| target_xy | 5-10 cm | Target width and height (square cross-section) |
| gap | 20-60 cm | Vacuum decay chamber length (NOT total distance) |
| t_min | 0-10 ns | Timing window start |
| t_max | 1-100 ns | Timing window end |

**Note on Gap Parameter:** The `gap` represents the vacuum decay chamber length only. The magnet region length is fixed. When you change the vacuum chamber length, downstream components (silicon trackers, calorimeter) are repositioned accordingly.

### Interpreting Objectives

All objectives are minimized in the optimization:

| Objective | Meaning | Target |
|-----------|---------|--------|
| signal (neg) | Negative signal rate | Maximize (most negative = best) |
| neutron | Neutrons in window | Minimize |
| photon_bkg | EM punch-through | Minimize |
| separability (neg) | Negative separability | Maximize (most negative = best) |

### Trade-off Analysis

The framework produces a **Pareto front** - a set of non-dominated solutions where improving one objective requires worsening another. Key trade-offs:

- **Signal vs Background**: Thicker targets produce more signal but more background
- **Gap vs Timing**: Larger gaps give better time separation but weaker signal
- **Separability vs Rate**: Tight angular cuts improve separability but reduce acceptance

Use the `rank_solutions()` function or look at `ranked_solutions.csv` to find the best compromise.

---

## Architecture

### Module Overview

```
scripts/
├── geant4_runner.py          # Simulation interface
├── objectives.py             # Objective calculations  
├── optimization_problem.py   # pymoo integration
├── bayesian_optimization.py  # BoTorch optimization
├── run_optimization.py       # Main pipeline
├── visualization.py         # Analysis tools
├── flux_converter.py        # Geant4 → alplib format
└── damsa_alplib_analysis.py  # ALP signal calculation
```

### Data Flow

```
Geant4 Simulation
       ↓
  photon_flux.csv / background.csv
       ↓
geant4_runner.py → SimulationResult
       ↓
objectives.py → ObjectiveValues
       ↓
optimization_problem.py / bayesian_optimization.py
       ↓
Pareto Front
       ↓
visualization.py → Plots & Analysis
```

---

## Troubleshooting

### Missing Dependencies

```bash
# Core
pip install numpy scipy pandas matplotlib seaborn

# BoTorch (BO)
pip install torch botorch

# pymoo (NSGA-II)
pip install pymoo
```

### Out of Memory

Reduce simulation events or batch size:
```bash
python run_optimization.py --n-events 500 --n-bo-init 10
```

### Slow Convergence

Increase BO iterations or use warm-start:
```bash
python run_optimization.py --n-bo-iter 100
```

### Cache Issues

Clear cached results:
```bash
rm -rf optimization_cache/
```

---

## Advanced Usage

### Custom Objectives

Edit `objectives.py` to add custom physics:

```python
class CustomObjectiveFunctions(ObjectiveFunctions):
    def _calculate_custom_signal(self, sim_result):
        # Your physics here
        return custom_value
```

### Parallel Execution

Run multiple simulations in parallel:
```bash
# Modify geant4_runner.py
runner = Geant4Runner(n_workers=4)
```

### Custom Visualization

```python
from visualization import plot_pareto_2d, analyze_tradeoffs

# Load results
results = load_optimization_results('results/')

# Custom plot
plot_pareto_2d(results['F'], obj_indices=(0, 1))
```

---

## References

- BoTorch: https://botorch.org
- pymoo: https://pymoo.org
- alplib: https://github.com/axionl/alplib
- Geant4: https://geant4.web.cern.ch

---

## License

DAMSA Collaboration
