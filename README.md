# DAMSA Simulation Repository

DAMSA stands for **DArk Messenger Searches at an Accelerator**. It is a proposed short-baseline experiment searching for dark-sector particles and other short-lived new-physics signatures.

This repository contains the Geant4 + ROOT simulation and optimization workflows used to study detector performance for DAMSA, with a strong focus on:

- background characterization
- detector/beamline geometry optimization
- objective-driven scans for staged DAMSA studies

## Experiment Context

The DAMSA experiment concept and physics goals are documented in the CDR white paper:

- **DAMSA Experiment Conceptual Design White Paper**, arXiv:2601.15255v2  
  https://arxiv.org/abs/2601.15255v2

From the CDR abstract: DAMSA targets MeV-to-sub-GeV dark-sector messengers and is designed to probe short-lived processes with an ultra-short baseline.

## Current Branch Focus

This branch (`8GeV-e/optimization/T1-2`) is focused on **geometry optimization and background studies for an 8 GeV electron beam configuration at the SLAC LESA facility**.

## What Is In This Repository

- `damsa` - main Geant4 simulation executable
- `damsa_opt` - optimization-oriented batch runner
- `damsa_alp_inject` - ALP injection workflow executable
- Python tooling for optimization, pipelines, and analysis

## Repository Layout

- `src/` - detector, physics, generators, actions, and data components
- `macros/` - Geant4 macro files for run and visualization modes
- `scripts/` - optimization, pipeline, runner, and visualization scripts
- `Reading/` - DAMSA reference material (including CDR PDF)

## Dependency

- [alplib](https://github.com/athompson-git/alplib.git) - ALP production and decay library used for signal generation and injection studies

## Prerequisites

- CMake >= 3.10
- Geant4 (UI/vis-enabled build recommended; current codes are tuned for MultiThread mode)
- ROOT (`root-config` available in `PATH`)
- C++11 compiler

Optional Python environment:

- Python 3.10+
- packages in `requirements.txt`

## Build

```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

## Run

Interactive visualization:

```bash
./build/damsa
```

Batch run with macro:

```bash
./build/damsa macros/run.mac
```

Optimization batch example:

```bash
./build/damsa_opt --target-z 10 --target-xy 5 --gap 47 --n-events 10000 --output-dir output
```

ALP injection run:

```bash
./build/damsa_alp_inject macros/run_alp.mac
```

## Outputs

- ROOT output files (event and detector-level records)
- CSV/JSON files for optimization and post-processing
- plots and analysis artifacts in `plots/` and `output/`

## Python Workflows

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Main script groups:

- `scripts/optimization/` - fast scans, Pareto workflows, Bayesian optimization
- `scripts/pipeline/` - flux conversion and ALP signal pipeline
- `scripts/runner/` - Geant4 orchestration and objective evaluation
