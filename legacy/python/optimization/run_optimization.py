#!/usr/bin/env python3
"""
DAMSA hybrid optimization pipeline: Bayesian optimization (BoTorch) for
sample-efficient exploration, then NSGA-II for Pareto front refinement
warm-started from the BO data.

Usage:
    python run_optimization.py --mode hybrid --n-bo-iter 50 --n-nsga-gen 100
"""

import argparse
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Local imports
from scripts.runner.geant4_runner import Geant4Runner, MockGeant4Runner
from scripts.runner.objectives import ObjectiveFunctions, ObjectiveValues, FigureOfMerit

# Conditional imports - optimization_problem was removed
try:
    from optimization_problem import (
        OptimizationConfig, 
        DAMSAOptimizationProblem,
        run_nsga2_optimization,
        run_nsga3_optimization,
        save_optimization_results,
        PYMOO_AVAILABLE
    )
    OptimizationConfig = OptimizationConfig
except ImportError:
    OptimizationConfig = None
    DAMSAOptimizationProblem = None
    run_nsga2_optimization = None
    run_nsga3_optimization = None
    save_optimization_results = None
    PYMOO_AVAILABLE = False

# Conditional imports
try:
    from scripts.optimization.bayesian_optimization import (
        BOConfig,
        MultiObjectiveBayesianOptimizer,
        run_bayesian_optimization,
        warm_start_from_history,
        BOTORCH_AVAILABLE
    )
except ImportError:
    BOTORCH_AVAILABLE = False
    BOConfig = None  # Placeholder when BoTorch unavailable


def create_output_directory(base_dir: str = "optimization_results") -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_hybrid_optimization(
    opt_config: OptimizationConfig,
    bo_config: BOConfig,
    nsga_pop_size: int = 50,
    nsga_n_gen: int = 100,
    output_dir: Path = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run hybrid BO + NSGA-II optimization.
    
    Strategy:
    1. Run Bayesian optimization for efficient exploration
    2. Use BO results to warm-start NSGA-II population
    3. Run NSGA-II for Pareto front refinement
    """
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch required for hybrid mode. Use --mode nsga-only")
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo required for hybrid mode. Use --mode bo-only")
    
    if output_dir is None:
        output_dir = create_output_directory()
    
    start_time = time.time()
    
    if verbose:
        print("=" * 70)
        print("DAMSA Hybrid Optimization Pipeline")
        print("=" * 70)
        print(f"\nPhase 1: Bayesian Optimization")
        print(f"  Initial samples: {bo_config.n_initial}")
        print(f"  BO iterations: {bo_config.n_iterations}")
        print(f"\nPhase 2: NSGA-II Refinement")
        print(f"  Population size: {nsga_pop_size}")
        print(f"  Generations: {nsga_n_gen}")
        print(f"\nOutput: {output_dir}")
        print("=" * 70)
    
    # Phase 1: Bayesian Optimization
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: Bayesian Optimization (Exploration)")
        print("=" * 70)
    
    bo_config.output_dir = str(output_dir / "phase1_bo")
    
    bo_results = run_bayesian_optimization(
        opt_config=opt_config,
        bo_config=bo_config,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nBO Phase Complete:")
        print(f"  Evaluations: {bo_results['n_evals']}")
        print(f"  Pareto solutions: {bo_results['n_pareto']}")
        print(f"  Hypervolume: {bo_results['hypervolume']:.4e}")
    
    # Phase 2: NSGA-II with warm-start
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: NSGA-II Refinement (Population-based)")
        print("=" * 70)
    
    # Create initial population from BO results
    # Use all BO evaluations as starting knowledge
    bo_X = bo_results['all_X']
    bo_Y = bo_results['all_Y']
    
    # Create problem with warm-start
    problem = DAMSAOptimizationProblem(opt_config, verbose=verbose)
    
    # Add BO history to problem
    for i in range(len(bo_X)):
        problem.eval_history.append({
            'eval_id': i + 1,
            'X': bo_X[i].tolist(),
            'F': bo_Y[i].tolist(),
            'G': [0.0, 0.0],  # Assume feasible from BO
            'source': 'bo_warmstart'
        })
    problem.n_evals = len(bo_X)
    
    # Run NSGA-II
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize
    from pymoo.core.population import Population
    from pymoo.core.evaluator import Evaluator
    
    # Create initial population from best BO points
    # Select diverse subset of BO evaluations
    n_init = min(nsga_pop_size, len(bo_X))
    
    # Use Pareto front + random selection
    pareto_X = bo_results['X']
    n_pareto = len(pareto_X)
    
    if n_pareto >= n_init:
        init_X = pareto_X[:n_init]
    else:
        # Fill with random BO points
        remaining = n_init - n_pareto
        random_idx = np.random.choice(len(bo_X), size=min(remaining, len(bo_X)), replace=False)
        init_X = np.vstack([pareto_X, bo_X[random_idx]])[:n_init]
    
    # Pad with random if needed
    if len(init_X) < nsga_pop_size:
        n_random = nsga_pop_size - len(init_X)
        random_X = np.random.uniform(
            opt_config.xl, opt_config.xu, 
            size=(n_random, opt_config.n_var)
        )
        init_X = np.vstack([init_X, random_X])
    
    # Configure NSGA-II with custom initial population
    algorithm = NSGA2(
        pop_size=nsga_pop_size,
        sampling=init_X,  # Use BO solutions as initial population
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", nsga_n_gen)
    
    nsga_start = time.time()
    
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=verbose,
        save_history=True
    )
    
    nsga_time = time.time() - nsga_start
    
    # Combine results
    total_time = time.time() - start_time
    
    combined_results = {
        'X': result.X,
        'F': result.F,
        'G': result.G,
        'n_evals_total': problem.n_evals,
        'n_evals_bo': bo_results['n_evals'],
        'n_evals_nsga': problem.n_evals - bo_results['n_evals'],
        'n_pareto_final': len(result.X) if result.X is not None else 0,
        'n_pareto_bo': bo_results['n_pareto'],
        'hypervolume_bo': bo_results['hypervolume'],
        'run_time_total': total_time,
        'run_time_bo': bo_results['run_time_seconds'],
        'run_time_nsga': nsga_time,
        'config': opt_config,
        'bo_config': bo_config,
        'nsga_config': {
            'pop_size': nsga_pop_size,
            'n_gen': nsga_n_gen
        },
        'history': problem.eval_history,
        'bo_results': bo_results
    }
    
    # Save results
    save_hybrid_results(combined_results, output_dir)
    
    if verbose:
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"\nTotal evaluations: {problem.n_evals}")
        print(f"  BO phase: {bo_results['n_evals']}")
        print(f"  NSGA-II phase: {problem.n_evals - bo_results['n_evals']}")
        print(f"\nPareto solutions: {len(result.X) if result.X is not None else 0}")
        print(f"Total time: {total_time:.1f}s")
        print(f"\nResults saved to: {output_dir}")
    
    return combined_results


def save_hybrid_results(results: Dict[str, Any], output_dir: Path):
    """Save hybrid optimization results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save Pareto front
    if results['X'] is not None:
        np.savez(
            output_dir / "pareto_front.npz",
            X=results['X'],
            F=results['F'],
            G=results.get('G', np.array([]))
        )
        
        # Save as CSV
        import pandas as pd
        
        # Use config var_names if available
        if hasattr(results.get('config'), 'var_names'):
            var_names = results['config'].var_names
        else:
            var_names = ['target_z_cm', 'target_xy_cm', 'gap_cm', 't_min_ns', 't_max_ns']
        
        obj_names = results['config'].objectives if hasattr(results.get('config'), 'objectives') else \
                   ['obj_' + str(i) for i in range(results['F'].shape[1])]
        
        df = pd.DataFrame(results['X'], columns=var_names)
        for i, name in enumerate(obj_names):
            df[name] = results['F'][:, i]
        
        df.to_csv(output_dir / "pareto_solutions.csv", index=False)
    
    # Save history
    with open(output_dir / "evaluation_history.json", 'w') as f:
        json.dump(results.get('history', []), f, indent=2)
    
    # Save summary
    summary = {
        'n_evals_total': results['n_evals_total'],
        'n_evals_bo': results['n_evals_bo'],
        'n_evals_nsga': results['n_evals_nsga'],
        'n_pareto_final': results['n_pareto_final'],
        'n_pareto_bo': results['n_pareto_bo'],
        'hypervolume_bo': results['hypervolume_bo'],
        'run_time_total': results['run_time_total'],
        'run_time_bo': results['run_time_bo'],
        'run_time_nsga': results['run_time_nsga']
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save config
    if hasattr(results.get('config'), 'save'):
        results['config'].save(str(output_dir / "config.json"))
    
    print(f"Results saved to: {output_dir}")


def run_bo_only_optimization(
    opt_config: OptimizationConfig,
    bo_config: BOConfig,
    output_dir: Path = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run Bayesian optimization only."""
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch required. Install with: pip install botorch")
    
    if output_dir is None:
        output_dir = create_output_directory()
    
    bo_config.output_dir = str(output_dir)
    
    if verbose:
        print("=" * 70)
        print("DAMSA Bayesian Optimization")
        print("=" * 70)
        print(f"Initial samples: {bo_config.n_initial}")
        print(f"BO iterations: {bo_config.n_iterations}")
        print(f"Output: {output_dir}")
        print("=" * 70)
    
    results = run_bayesian_optimization(
        opt_config=opt_config,
        bo_config=bo_config,
        verbose=verbose
    )
    
    return results


def run_nsga_only_optimization(
    opt_config: OptimizationConfig,
    pop_size: int = 50,
    n_gen: int = 100,
    use_nsga3: bool = False,
    output_dir: Path = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run NSGA-II/III optimization only."""
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo required. Install with: pip install pymoo")
    
    if output_dir is None:
        output_dir = create_output_directory()
    
    if verbose:
        alg_name = "NSGA-III" if use_nsga3 else "NSGA-II"
        print("=" * 70)
        print(f"DAMSA {alg_name} Optimization")
        print("=" * 70)
        print(f"Population size: {pop_size}")
        print(f"Generations: {n_gen}")
        print(f"Output: {output_dir}")
        print("=" * 70)
    
    if use_nsga3:
        results = run_nsga3_optimization(
            config=opt_config,
            pop_size=pop_size,
            n_gen=n_gen,
            verbose=verbose
        )
    else:
        results = run_nsga2_optimization(
            config=opt_config,
            pop_size=pop_size,
            n_gen=n_gen,
            verbose=verbose
        )
    
    save_optimization_results(results, str(output_dir))
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DAMSA Multi-Objective Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hybrid optimization (recommended)
  python run_optimization.py --mode hybrid --n-bo-iter 50 --n-nsga-gen 100
  
  # BO only (for small budgets)
  python run_optimization.py --mode bo-only --n-bo-iter 100
  
  # NSGA-II only (no BoTorch needed)
  python run_optimization.py --mode nsga-only --n-nsga-gen 200
  
  # With custom physics parameters
  python run_optimization.py --mode hybrid --axion-mass 50 --coupling 1e-5
  
  # Use real Geant4 (not mock)
  python run_optimization.py --mode hybrid --no-mock --n-events 10000
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str, default='hybrid',
        choices=['hybrid', 'bo-only', 'nsga-only'],
        help='Optimization mode (default: hybrid)'
    )
    
    # BO parameters
    parser.add_argument(
        '--n-bo-init', type=int, default=20,
        help='Number of initial BO samples (default: 20)'
    )
    parser.add_argument(
        '--n-bo-iter', type=int, default=50,
        help='Number of BO iterations (default: 50)'
    )
    
    # NSGA parameters
    parser.add_argument(
        '--n-nsga-pop', type=int, default=50,
        help='NSGA population size (default: 50)'
    )
    parser.add_argument(
        '--n-nsga-gen', type=int, default=100,
        help='NSGA generations (default: 100)'
    )
    parser.add_argument(
        '--use-nsga3', action='store_true',
        help='Use NSGA-III instead of NSGA-II'
    )
    
    # Physics parameters
    parser.add_argument(
        '--axion-mass', type=float, default=100.0,
        help='ALP mass in MeV (default: 100)'
    )
    parser.add_argument(
        '--coupling', type=float, default=1e-3,
        help='ALP-photon coupling (default: 1e-3)'
    )
    parser.add_argument(
        '--exposure', type=float, default=30.0,
        help='Exposure time in days (default: 30)'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--no-mock', action='store_true',
        help='Use real Geant4 instead of mock runner'
    )
    parser.add_argument(
        '--n-events', type=int, default=1000,
        help='Events per simulation (default: 1000)'
    )
    
    # Objectives
    parser.add_argument(
        '--objectives', type=str, nargs='+',
        default=['signal', 'neutron', 'photon_bkg', 'separability'],
        help='Objectives to optimize'
    )
    
    # Output
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (auto-generated if not specified)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    opt_config = OptimizationConfig(
        use_mock=not args.no_mock,
        n_events=args.n_events,
        axion_mass_MeV=args.axion_mass,
        axion_coupling=args.coupling,
        exposure_days=args.exposure,
        objectives=args.objectives
    )
    
    # Create BO config only if BoTorch is available
    bo_config = None
    if BOTORCH_AVAILABLE and BOConfig is not None:
        bo_config = BOConfig(
            n_initial=args.n_bo_init,
            n_iterations=args.n_bo_iter
        )
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory()
    
    verbose = not args.quiet
    
    # Print configuration
    if verbose:
        print("\nConfiguration:")
        print(f"  Mode: {args.mode}")
        print(f"  Mock runner: {opt_config.use_mock}")
        print(f"  Events per sim: {opt_config.n_events}")
        print(f"  Objectives: {opt_config.objectives}")
        print(f"  ALP mass: {opt_config.axion_mass_MeV} MeV")
        print(f"  Coupling: {opt_config.axion_coupling}")
        print(f"  Exposure: {opt_config.exposure_days} days")
        print(f"  Output: {output_dir}\n")
    
    # Run optimization
    try:
        if args.mode == 'hybrid':
            if bo_config is None:
                raise ImportError("BoTorch required for hybrid mode. Use --mode nsga-only")
            results = run_hybrid_optimization(
                opt_config=opt_config,
                bo_config=bo_config,
                nsga_pop_size=args.n_nsga_pop,
                nsga_n_gen=args.n_nsga_gen,
                output_dir=output_dir,
                verbose=verbose
            )
        
        elif args.mode == 'bo-only':
            if bo_config is None:
                raise ImportError("BoTorch required for bo-only mode. Use --mode nsga-only")
            results = run_bo_only_optimization(
                opt_config=opt_config,
                bo_config=bo_config,
                output_dir=output_dir,
                verbose=verbose
            )
        
        elif args.mode == 'nsga-only':
            results = run_nsga_only_optimization(
                opt_config=opt_config,
                pop_size=args.n_nsga_pop,
                n_gen=args.n_nsga_gen,
                use_nsga3=args.use_nsga3,
                output_dir=output_dir,
                verbose=verbose
            )
        
        # Print summary
        if verbose and results.get('X') is not None:
            print("\n" + "=" * 70)
            print("FINAL PARETO FRONT")
            print("=" * 70)
            
            n_show = min(5, len(results['X']))
            var_names = ['target_z', 'target_xy', 'gap', 't_min', 't_max']
            
            print(f"\nShowing {n_show} of {len(results['X'])} Pareto solutions:\n")
            
            for i in range(n_show):
                print(f"Solution {i+1}:")
                print(f"  Variables: ", end="")
                for j, name in enumerate(var_names):
                    print(f"{name}={results['X'][i,j]:.2f} ", end="")
                print()
                print(f"  Objectives: {results['F'][i]}")
                print()
    
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nAvailable modes depend on installed packages:")
        print(f"  BoTorch available: {BOTORCH_AVAILABLE}")
        print(f"  pymoo available: {PYMOO_AVAILABLE}")
        
        if not BOTORCH_AVAILABLE and args.mode in ['hybrid', 'bo-only']:
            print("\nInstall BoTorch: pip install botorch")
        if not PYMOO_AVAILABLE and args.mode in ['hybrid', 'nsga-only']:
            print("\nInstall pymoo: pip install pymoo")
        
        return 1
    
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
