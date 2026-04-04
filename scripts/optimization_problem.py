#!/usr/bin/env python3
"""
DAMSA Multi-Objective Optimization Problem Definition

This module defines the optimization problem for pymoo NSGA-II
and BoTorch Bayesian optimization.

Decision Variables:
    - target_z: Target thickness (cm)
    - target_xy: Target width and height (cm) - square cross-section
    - gap: Vacuum decay chamber length (cm) - NOT total distance
    - t_min: Timing window start (ns)
    - t_max: Timing window end (ns)

Note on Gap Parameter:
    The 'gap' parameter represents the vacuum decay chamber length only.
    The magnet region length stays constant. Changing the vacuum chamber
    length adjusts the positions of downstream components (trackers,
    calorimeter) accordingly.

Objectives (all to minimize):
    - f1: -signal_rate (maximize signal)
    - f2: neutron_background (minimize)
    - f3: photon_background (minimize)
    - f4: -photon_separability (maximize separability)

Constraints:
    - Target dimensions within physical limits
    - Gap + target/2 fits in experimental hall
    - Valid timing window (t_max > t_min)

Author: DAMSA Collaboration
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time

# pymoo imports
try:
    from pymoo.core.problem import Problem, ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    # Create placeholder base class that accepts any arguments
    class Problem:
        def __init__(self, *args, **kwargs):
            pass
    class ElementwiseProblem:
        def __init__(self, *args, **kwargs):
            pass
    print("Warning: pymoo not installed. NSGA-II optimization unavailable.")
    print("Install with: pip install pymoo")

# Local imports
from geant4_runner import Geant4Runner, MockGeant4Runner, SimulationResult
from objectives import ObjectiveFunctions, ObjectiveValues, FigureOfMerit


@dataclass
class OptimizationConfig:
    """
    Configuration for the optimization problem.
    
    Note: The target has a square cross-section (target_x = target_y),
    so we use a single 'target_xy_range' parameter for both dimensions.
    
    The 'gap' parameter represents the vacuum decay chamber length only.
    The magnet region length is fixed; changing the vacuum chamber length
    adjusts positions of downstream components (trackers, calorimeter).
    """
    # Variable bounds
    target_z_range: Tuple[float, float] = (5.0, 20.0)    # cm, thickness along beam
    target_xy_range: Tuple[float, float] = (5.0, 10.0)   # cm, square cross-section (x=y)
    gap_range: Tuple[float, float] = (20.0, 60.0)        # cm, vacuum decay chamber length
    t_min_range: Tuple[float, float] = (0.0, 10.0)       # ns
    t_max_range: Tuple[float, float] = (1.0, 100.0)      # ns
    
    # Constraints
    max_hall_length: float = 100.0  # cm, experimental hall constraint (adjusted for new ranges)
    min_timing_window: float = 1.0  # ns, minimum timing window width
    
    # Physics parameters
    beam_current_uA: float = 62.5
    axion_mass_MeV: float = 100.0
    axion_coupling: float = 1e-3
    exposure_days: float = 30.0
    
    # Simulation parameters
    n_events: int = 1000
    use_mock: bool = True  # Use mock runner for testing
    
    # Optimization parameters
    objectives: List[str] = None  # Which objectives to use
    
    def __post_init__(self):
        if self.objectives is None:
            # Default: 4 objectives
            self.objectives = ['signal', 'neutron', 'photon_bkg', 'separability']
    
    @property
    def n_var(self) -> int:
        """Number of decision variables (5: target_z, target_xy, gap, t_min, t_max)."""
        return 5
    
    @property
    def n_obj(self) -> int:
        return len(self.objectives)
    
    @property
    def var_names(self) -> List[str]:
        """Names of decision variables."""
        return ['target_z_cm', 'target_xy_cm', 'gap_cm', 't_min_ns', 't_max_ns']
    
    @property
    def xl(self) -> np.ndarray:
        """Lower bounds."""
        return np.array([
            self.target_z_range[0],
            self.target_xy_range[0],
            self.gap_range[0],
            self.t_min_range[0],
            self.t_max_range[0]
        ])
    
    @property
    def xu(self) -> np.ndarray:
        """Upper bounds."""
        return np.array([
            self.target_z_range[1],
            self.target_xy_range[1],
            self.gap_range[1],
            self.t_min_range[1],
            self.t_max_range[1]
        ])
    
    def save(self, path: str):
        """Save configuration to JSON."""
        config_dict = {
            'target_z_range': self.target_z_range,
            'target_xy_range': self.target_xy_range,
            'gap_range': self.gap_range,
            't_min_range': self.t_min_range,
            't_max_range': self.t_max_range,
            'max_hall_length': self.max_hall_length,
            'min_timing_window': self.min_timing_window,
            'beam_current_uA': self.beam_current_uA,
            'axion_mass_MeV': self.axion_mass_MeV,
            'axion_coupling': self.axion_coupling,
            'exposure_days': self.exposure_days,
            'n_events': self.n_events,
            'use_mock': self.use_mock,
            'objectives': self.objectives
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'OptimizationConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Handle legacy configs with separate target_x_range/target_y_range
        if 'target_x_range' in config_dict and 'target_xy_range' not in config_dict:
            config_dict['target_xy_range'] = config_dict.pop('target_x_range')
            config_dict.pop('target_y_range', None)
        return cls(**config_dict)


class DAMSAOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for DAMSA detector.
    
    Inherits from pymoo Problem class for use with NSGA-II/III.
    
    Parameters
    ----------
    config : OptimizationConfig
        Problem configuration
    runner : Geant4Runner or MockGeant4Runner
        Simulation runner
    verbose : bool
        Print progress
    """
    
    def __init__(self, 
                 config: OptimizationConfig = None,
                 runner = None,
                 verbose: bool = True):
        
        self.config = config or OptimizationConfig()
        self.verbose = verbose
        
        # Setup runner
        if runner is not None:
            self.runner = runner
        elif self.config.use_mock:
            self.runner = MockGeant4Runner(verbose=False)
        else:
            self.runner = Geant4Runner(
                n_events=self.config.n_events,
                verbose=False
            )
        
        # Setup objective function calculator
        self.obj_func = ObjectiveFunctions(
            beam_current_uA=self.config.beam_current_uA,
            axion_mass_MeV=self.config.axion_mass_MeV,
            axion_coupling=self.config.axion_coupling,
            exposure_days=self.config.exposure_days
        )
        
        # Evaluation counter
        self.n_evals = 0
        self.eval_history = []
        
        # Initialize pymoo Problem
        super().__init__(
            n_var=self.config.n_var,
            n_obj=self.config.n_obj,
            n_ieq_constr=2,  # Hall length + timing window
            xl=self.config.xl,
            xu=self.config.xu
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives and constraints for population.
        
        Parameters
        ----------
        X : np.ndarray
            Population matrix (pop_size, n_var)
            Variables: [target_z, target_xy, gap, t_min, t_max]
        out : dict
            Output dictionary for objectives ("F") and constraints ("G")
        """
        pop_size = X.shape[0]
        
        F = np.zeros((pop_size, self.config.n_obj))
        G = np.zeros((pop_size, 2))
        
        for i in range(pop_size):
            target_z, target_xy, gap, t_min, t_max = X[i]
            
            # Run simulation (target_x = target_y = target_xy for square target)
            try:
                sim_result = self.runner.run(
                    target_z=target_z,
                    target_x=target_xy,
                    target_y=target_xy,
                    gap=gap
                )
                
                # Calculate objectives
                objectives = self.obj_func.calculate(sim_result, t_min, t_max)
                
                # Convert to array (minimization form)
                F[i] = objectives.to_array(self.config.objectives)
                
            except Exception as e:
                if self.verbose:
                    print(f"Evaluation {i} failed: {e}")
                # Penalize failed evaluations
                F[i] = np.array([1e10] * self.config.n_obj)
            
            # Constraints (g(x) <= 0 means satisfied)
            # C1: Hall length constraint
            G[i, 0] = (target_z / 2 + gap) - self.config.max_hall_length
            
            # C2: Valid timing window
            G[i, 1] = (t_min + self.config.min_timing_window) - t_max
            
            self.n_evals += 1
            
            # Log evaluation
            self.eval_history.append({
                'eval_id': self.n_evals,
                'X': X[i].tolist(),
                'F': F[i].tolist(),
                'G': G[i].tolist()
            })
            
            if self.verbose and self.n_evals % 10 == 0:
                print(f"Evaluations: {self.n_evals}")
        
        out["F"] = F
        out["G"] = G
    
    def evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate single point (for Bayesian optimization).
        
        Parameters
        ----------
        x : np.ndarray
            Decision variables [target_z, target_xy, gap, t_min, t_max]
            Note: target_xy is used for both target_x and target_y (square target)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (objectives, constraints)
        """
        X = x.reshape(1, -1)
        out = {}
        self._evaluate(X, out)
        return out["F"][0], out["G"][0]
    
    def save_history(self, path: str):
        """Save evaluation history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.eval_history, f, indent=2)


class DAMSAElementwiseProblem(ElementwiseProblem):
    """
    Elementwise version for parallel evaluation.
    
    Each individual is evaluated separately, allowing parallel execution.
    """
    
    def __init__(self, config: OptimizationConfig = None, verbose: bool = False):
        self.config = config or OptimizationConfig()
        self.verbose = verbose
        
        # Setup components (will be initialized in worker processes)
        self._runner = None
        self._obj_func = None
        
        super().__init__(
            n_var=self.config.n_var,
            n_obj=self.config.n_obj,
            n_ieq_constr=2,
            xl=self.config.xl,
            xu=self.config.xu
        )
    
    def _get_runner(self):
        """Lazy initialization of runner (for multiprocessing)."""
        if self._runner is None:
            if self.config.use_mock:
                self._runner = MockGeant4Runner(verbose=False)
            else:
                self._runner = Geant4Runner(
                    n_events=self.config.n_events,
                    verbose=False
                )
        return self._runner
    
    def _get_obj_func(self):
        """Lazy initialization of objective function."""
        if self._obj_func is None:
            self._obj_func = ObjectiveFunctions(
                beam_current_uA=self.config.beam_current_uA,
                axion_mass_MeV=self.config.axion_mass_MeV,
                axion_coupling=self.config.axion_coupling,
                exposure_days=self.config.exposure_days
            )
        return self._obj_func
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate single individual."""
        target_z, target_xy, gap, t_min, t_max = x
        
        runner = self._get_runner()
        obj_func = self._get_obj_func()
        
        try:
            # Square target: target_x = target_y = target_xy
            sim_result = runner.run(
                target_z=target_z,
                target_x=target_xy,
                target_y=target_xy,
                gap=gap
            )
            
            objectives = obj_func.calculate(sim_result, t_min, t_max)
            out["F"] = objectives.to_array(self.config.objectives)
            
        except Exception as e:
            out["F"] = np.array([1e10] * self.config.n_obj)
        
        # Constraints
        out["G"] = np.array([
            (target_z / 2 + gap) - self.config.max_hall_length,
            (t_min + self.config.min_timing_window) - t_max
        ])


def run_nsga2_optimization(
    config: OptimizationConfig = None,
    pop_size: int = 50,
    n_gen: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run NSGA-II optimization.
    
    Parameters
    ----------
    config : OptimizationConfig
        Problem configuration
    pop_size : int
        Population size
    n_gen : int
        Number of generations
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Optimization results
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo is required for NSGA-II optimization")
    
    config = config or OptimizationConfig()
    problem = DAMSAOptimizationProblem(config, verbose=verbose)
    
    # Configure NSGA-II
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Termination
    termination = get_termination("n_gen", n_gen)
    
    # Run optimization
    start_time = time.time()
    
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
        save_history=True
    )
    
    run_time = time.time() - start_time
    
    # Extract results
    return {
        'X': result.X,  # Pareto-optimal decision variables
        'F': result.F,  # Pareto-optimal objectives
        'G': result.G,  # Constraint values
        'n_gen': result.algorithm.n_gen,
        'n_evals': problem.n_evals,
        'run_time_seconds': run_time,
        'config': config,
        'history': problem.eval_history
    }


def run_nsga3_optimization(
    config: OptimizationConfig = None,
    pop_size: int = None,
    n_gen: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run NSGA-III optimization (better for many objectives).
    
    Parameters
    ----------
    config : OptimizationConfig
        Problem configuration
    pop_size : int
        Population size (auto-calculated if None)
    n_gen : int
        Number of generations
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Optimization results
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo is required for NSGA-III optimization")
    
    config = config or OptimizationConfig()
    problem = DAMSAOptimizationProblem(config, verbose=verbose)
    
    # Reference directions for NSGA-III
    n_obj = config.n_obj
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    if pop_size is None:
        pop_size = len(ref_dirs)
    
    # Configure NSGA-III
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Termination
    termination = get_termination("n_gen", n_gen)
    
    # Run optimization
    start_time = time.time()
    
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
        save_history=True
    )
    
    run_time = time.time() - start_time
    
    return {
        'X': result.X,
        'F': result.F,
        'G': result.G,
        'n_gen': result.algorithm.n_gen,
        'n_evals': problem.n_evals,
        'run_time_seconds': run_time,
        'config': config,
        'history': problem.eval_history
    }


def save_optimization_results(results: Dict[str, Any], output_dir: str):
    """
    Save optimization results to files.
    
    Parameters
    ----------
    results : dict
        Results from optimization
    output_dir : str
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save Pareto front
    np.savez(
        output_dir / "pareto_front.npz",
        X=results['X'],
        F=results['F'],
        G=results.get('G', np.array([]))
    )
    
    # Save config
    if isinstance(results.get('config'), OptimizationConfig):
        results['config'].save(str(output_dir / "config.json"))
    
    # Save history
    history_path = output_dir / "evaluation_history.json"
    with open(history_path, 'w') as f:
        json.dump(results.get('history', []), f, indent=2)
    
    # Save summary CSV
    if results['X'] is not None and len(results['X']) > 0:
        import pandas as pd
        
        # Use config var_names if available, otherwise default
        if results.get('config') and hasattr(results['config'], 'var_names'):
            var_names = results['config'].var_names
        else:
            var_names = ['target_z_cm', 'target_xy_cm', 'gap_cm', 't_min_ns', 't_max_ns']
        
        obj_names = results['config'].objectives if results.get('config') else \
                    ['obj_' + str(i) for i in range(results['F'].shape[1])]
        
        df = pd.DataFrame(results['X'], columns=var_names)
        for i, name in enumerate(obj_names):
            df[name] = results['F'][:, i]
        
        df.to_csv(output_dir / "pareto_solutions.csv", index=False)
    
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    print("Testing DAMSA Optimization Problem...")
    
    # Create configuration
    config = OptimizationConfig(
        use_mock=True,
        n_events=100,
        objectives=['signal', 'neutron', 'photon_bkg', 'separability']
    )
    
    print(f"\nConfiguration:")
    print(f"  Variables: {config.n_var} - {config.var_names}")
    print(f"  Objectives: {config.n_obj} - {config.objectives}")
    print(f"  Bounds:")
    print(f"    Lower: {config.xl}")
    print(f"    Upper: {config.xu}")
    
    # Test single evaluation
    # Variables: [target_z, target_xy, gap, t_min, t_max]
    print("\nTesting single evaluation...")
    problem = DAMSAOptimizationProblem(config, verbose=False)
    
    x_test = np.array([10.0, 5.0, 100.0, 0.0, 10.0])
    F, G = problem.evaluate_single(x_test)
    
    print(f"  Input: {x_test}")
    print(f"  Objectives: {F}")
    print(f"  Constraints: {G}")
    
    # Test NSGA-II (small run)
    if PYMOO_AVAILABLE:
        print("\nTesting NSGA-II (5 generations)...")
        results = run_nsga2_optimization(
            config=config,
            pop_size=10,
            n_gen=5,
            verbose=False
        )
        
        print(f"  Evaluations: {results['n_evals']}")
        print(f"  Pareto solutions: {len(results['X'])}")
        print(f"  Run time: {results['run_time_seconds']:.1f}s")
        
        # Save results
        save_optimization_results(results, "optimization_test_output")
    else:
        print("\npymoo not installed - skipping NSGA-II test")
    
    print("\nTest complete!")
