#!/usr/bin/env python3
"""
Bayesian Optimization Module for DAMSA Using BoTorch

This module implements multi-objective Bayesian optimization using BoTorch
for efficient exploration of the DAMSA parameter space.

Key Features:
- Multi-objective optimization with qEHVI (Expected Hypervolume Improvement)
- Constraint handling via constrained acquisition functions
- Warm-starting from prior evaluations
- Integration with Geant4 runner and objective functions

Decision Variables (5 total):
    - target_z: Target thickness (cm)
    - target_xy: Target width and height (cm) - square cross-section
    - gap: Vacuum decay chamber length (cm)
    - t_min: Timing window start (ns)
    - t_max: Timing window end (ns)

Strategy:
1. Initial phase: Use Bayesian optimization for efficient exploration
2. After sufficient data: Hand off to NSGA-II for Pareto front refinement

Author: DAMSA Collaboration
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import time
import warnings

# BoTorch imports
try:
    import botorch
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.multi_objective.monte_carlo import (
        qExpectedHypervolumeImprovement,
        qNoisyExpectedHypervolumeImprovement
    )
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.box_decompositions.dominated import (
        DominatedPartitioning
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.utils.transforms import normalize, unnormalize
    
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    warnings.warn("BoTorch not installed. Install with: pip install botorch")

# Local imports
from geant4_runner import Geant4Runner, MockGeant4Runner, SimulationResult
from objectives import ObjectiveFunctions, ObjectiveValues, FigureOfMerit
from optimization_problem import OptimizationConfig


@dataclass
class BOConfig:
    """Configuration for Bayesian optimization."""
    # Optimization parameters
    n_initial: int = 20          # Initial random samples (Latin Hypercube)
    n_iterations: int = 50       # Number of BO iterations
    batch_size: int = 1          # Candidates per iteration (>1 for parallel)
    
    # Acquisition function settings
    mc_samples: int = 128        # MC samples for acquisition function
    
    # Reference point for hypervolume (all objectives to minimize)
    # Should be "worse" than any expected objective value
    ref_point: List[float] = None  # Auto-calculated if None
    
    # Constraint handling
    constraint_threshold: float = 0.0  # g(x) <= threshold means feasible
    
    # Model settings
    use_saas: bool = False       # Use SAAS prior for high-dim
    
    # Output
    output_dir: str = "bo_output"
    save_every: int = 5          # Save checkpoint every N iterations
    
    def __post_init__(self):
        if self.ref_point is None:
            # Default reference point for 4 objectives
            # [neg_signal, neutron, photon_bkg, neg_separability]
            # All should be "worse" than realistic values
            self.ref_point = [0.0, 1e8, 1e8, 0.0]


class MultiObjectiveBayesianOptimizer:
    """
    Multi-objective Bayesian optimization for DAMSA.
    
    Uses BoTorch's qEHVI (q-Expected Hypervolume Improvement) for
    efficient multi-objective optimization with constraints.
    
    Parameters
    ----------
    opt_config : OptimizationConfig
        Problem configuration (bounds, objectives, etc.)
    bo_config : BOConfig
        Bayesian optimization configuration
    runner : Geant4Runner or MockGeant4Runner
        Simulation runner
    verbose : bool
        Print progress
    """
    
    def __init__(self,
                 opt_config: OptimizationConfig = None,
                 bo_config: BOConfig = None,
                 runner=None,
                 verbose: bool = True):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required. Install with: pip install botorch")
        
        self.opt_config = opt_config or OptimizationConfig()
        self.bo_config = bo_config or BOConfig()
        self.verbose = verbose
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        
        # Setup runner
        if runner is not None:
            self.runner = runner
        elif self.opt_config.use_mock:
            self.runner = MockGeant4Runner(verbose=False)
        else:
            self.runner = Geant4Runner(
                n_events=self.opt_config.n_events,
                verbose=False
            )
        
        # Setup objective function
        self.obj_func = ObjectiveFunctions(
            beam_current_uA=self.opt_config.beam_current_uA,
            axion_mass_MeV=self.opt_config.axion_mass_MeV,
            axion_coupling=self.opt_config.axion_coupling,
            exposure_days=self.opt_config.exposure_days
        )
        
        # Bounds as tensors (normalized to [0, 1])
        self.bounds = torch.tensor(
            [[0.0] * self.opt_config.n_var, [1.0] * self.opt_config.n_var],
            dtype=self.dtype,
            device=self.device
        )
        
        # Original bounds for unnormalization
        self.lb = torch.tensor(self.opt_config.xl, dtype=self.dtype, device=self.device)
        self.ub = torch.tensor(self.opt_config.xu, dtype=self.dtype, device=self.device)
        
        # Reference point for hypervolume
        self.ref_point = torch.tensor(
            self.bo_config.ref_point[:self.opt_config.n_obj],
            dtype=self.dtype,
            device=self.device
        )
        
        # Data storage
        self.train_X = None    # Normalized inputs
        self.train_Y = None    # Objective values (for minimization)
        self.train_C = None    # Constraint values
        
        # History
        self.history = []
        self.n_evals = 0
        
        # Output directory
        self.output_dir = Path(self.bo_config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize X from original bounds to [0, 1]."""
        return (X - self.lb) / (self.ub - self.lb)
    
    def _unnormalize(self, X_normalized: torch.Tensor) -> torch.Tensor:
        """Unnormalize X from [0, 1] to original bounds."""
        return X_normalized * (self.ub - self.lb) + self.lb
    
    def _evaluate(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate objective and constraint functions.
        
        Parameters
        ----------
        X : torch.Tensor
            Normalized input tensor of shape (n, d)
            Variables: [target_z, target_xy, gap, t_min, t_max]
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (objectives, constraints) tensors
        """
        X_orig = self._unnormalize(X)
        n_points = X.shape[0]
        
        Y = torch.zeros(n_points, self.opt_config.n_obj, 
                       dtype=self.dtype, device=self.device)
        C = torch.zeros(n_points, 2, dtype=self.dtype, device=self.device)
        
        for i in range(n_points):
            x = X_orig[i].cpu().numpy()
            target_z, target_xy, gap, t_min, t_max = x
            
            try:
                # Run simulation (square target: target_x = target_y = target_xy)
                sim_result = self.runner.run(
                    target_z=target_z,
                    target_x=target_xy,
                    target_y=target_xy,
                    gap=gap
                )
                
                # Calculate objectives
                objectives = self.obj_func.calculate(sim_result, t_min, t_max)
                obj_array = objectives.to_array(self.opt_config.objectives)
                
                Y[i] = torch.tensor(obj_array, dtype=self.dtype, device=self.device)
                
            except Exception as e:
                if self.verbose:
                    print(f"Evaluation failed: {e}")
                Y[i] = torch.tensor([1e10] * self.opt_config.n_obj,
                                   dtype=self.dtype, device=self.device)
            
            # Constraints (g(x) <= 0 means feasible)
            C[i, 0] = (target_z / 2 + gap) - self.opt_config.max_hall_length
            C[i, 1] = (t_min + self.opt_config.min_timing_window) - t_max
            
            self.n_evals += 1
            
            # Log
            self.history.append({
                'eval_id': self.n_evals,
                'X': x.tolist(),
                'Y': Y[i].cpu().numpy().tolist(),
                'C': C[i].cpu().numpy().tolist(),
                'feasible': bool((C[i] <= self.bo_config.constraint_threshold).all())
            })
            
            if self.verbose:
                print(f"Eval {self.n_evals}: Y = {Y[i].cpu().numpy()}")
        
        return Y, C
    
    def _generate_initial_data(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate initial data using Sobol sampling.
        
        Parameters
        ----------
        n_samples : int
            Number of initial samples
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (X, Y, C) tensors
        """
        if self.verbose:
            print(f"Generating {n_samples} initial samples...")
        
        # Sobol sampling in [0, 1]^d
        X = draw_sobol_samples(
            bounds=self.bounds,
            n=n_samples,
            q=1
        ).squeeze(1)
        
        Y, C = self._evaluate(X)
        
        return X, Y, C
    
    def _build_models(self, X: torch.Tensor, Y: torch.Tensor) -> ModelListGP:
        """
        Build GP models for each objective.
        
        Parameters
        ----------
        X : torch.Tensor
            Training inputs
        Y : torch.Tensor
            Training outputs
            
        Returns
        -------
        ModelListGP
            List of GP models, one per objective
        """
        models = []
        
        for i in range(Y.shape[-1]):
            # Create individual GP for each objective
            model = SingleTaskGP(
                train_X=X,
                train_Y=Y[:, i:i+1],
                outcome_transform=Standardize(m=1)
            )
            models.append(model)
        
        model_list = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
        
        # Fit models
        fit_gpytorch_mll(mll)
        
        return model_list
    
    def _get_acquisition_function(self, 
                                   model: ModelListGP,
                                   Y_train: torch.Tensor,
                                   C_train: torch.Tensor) -> qExpectedHypervolumeImprovement:
        """
        Create qEHVI acquisition function with constraint handling.
        
        Parameters
        ----------
        model : ModelListGP
            GP model for objectives
        Y_train : torch.Tensor
            Training objective values
        C_train : torch.Tensor
            Training constraint values
            
        Returns
        -------
        qExpectedHypervolumeImprovement
            Acquisition function
        """
        # Filter to feasible points for Pareto front
        feasible_mask = (C_train <= self.bo_config.constraint_threshold).all(dim=-1)
        
        if feasible_mask.sum() > 0:
            Y_feasible = Y_train[feasible_mask]
        else:
            # No feasible points yet - use all points
            Y_feasible = Y_train
        
        # Compute Pareto front
        pareto_mask = is_non_dominated(Y_feasible)
        pareto_Y = Y_feasible[pareto_mask]
        
        # Partitioning for hypervolume
        partitioning = DominatedPartitioning(
            ref_point=self.ref_point,
            Y=pareto_Y
        )
        
        # Create acquisition function
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point.tolist(),
            partitioning=partitioning,
            sampler=None  # Use default sampler
        )
        
        return acq_func
    
    def _optimize_acquisition(self, 
                              acq_func,
                              batch_size: int = 1) -> torch.Tensor:
        """
        Optimize acquisition function to get next candidates.
        
        Parameters
        ----------
        acq_func : AcquisitionFunction
            Acquisition function to optimize
        batch_size : int
            Number of candidates to generate
            
        Returns
        -------
        torch.Tensor
            Next candidates to evaluate
        """
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200}
        )
        
        return candidates
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization loop.
        
        Returns
        -------
        dict
            Optimization results including Pareto front
        """
        start_time = time.time()
        
        if self.verbose:
            print("=" * 60)
            print("Starting Multi-Objective Bayesian Optimization")
            print(f"Objectives: {self.opt_config.objectives}")
            print(f"Variables: {self.opt_config.n_var}")
            print(f"Initial samples: {self.bo_config.n_initial}")
            print(f"BO iterations: {self.bo_config.n_iterations}")
            print("=" * 60)
        
        # Generate initial data
        self.train_X, self.train_Y, self.train_C = self._generate_initial_data(
            self.bo_config.n_initial
        )
        
        # Main optimization loop
        for iteration in range(self.bo_config.n_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.bo_config.n_iterations} ---")
            
            try:
                # Build GP models
                model = self._build_models(self.train_X, self.train_Y)
                
                # Get acquisition function
                acq_func = self._get_acquisition_function(
                    model, self.train_Y, self.train_C
                )
                
                # Optimize acquisition to get next point(s)
                new_X = self._optimize_acquisition(
                    acq_func, 
                    batch_size=self.bo_config.batch_size
                )
                
                # Evaluate new point(s)
                new_Y, new_C = self._evaluate(new_X)
                
                # Update training data
                self.train_X = torch.cat([self.train_X, new_X], dim=0)
                self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)
                self.train_C = torch.cat([self.train_C, new_C], dim=0)
                
                # Log progress
                if self.verbose:
                    feasible_mask = (self.train_C <= self.bo_config.constraint_threshold).all(dim=-1)
                    n_feasible = feasible_mask.sum().item()
                    print(f"Total evaluations: {self.n_evals}, Feasible: {n_feasible}")
                
                # Save checkpoint
                if (iteration + 1) % self.bo_config.save_every == 0:
                    self._save_checkpoint(iteration + 1)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Iteration {iteration + 1} failed: {e}")
                continue
        
        # Final results
        total_time = time.time() - start_time
        
        # Extract Pareto front
        feasible_mask = (self.train_C <= self.bo_config.constraint_threshold).all(dim=-1)
        
        if feasible_mask.sum() > 0:
            Y_feasible = self.train_Y[feasible_mask]
            X_feasible = self.train_X[feasible_mask]
            
            pareto_mask = is_non_dominated(Y_feasible)
            pareto_X = X_feasible[pareto_mask]
            pareto_Y = Y_feasible[pareto_mask]
        else:
            pareto_X = self.train_X
            pareto_Y = self.train_Y
        
        # Convert back to original scale
        pareto_X_orig = self._unnormalize(pareto_X)
        
        # Compute hypervolume
        if len(pareto_Y) > 0:
            partitioning = DominatedPartitioning(
                ref_point=self.ref_point,
                Y=pareto_Y
            )
            hypervolume = partitioning.compute_hypervolume().item()
        else:
            hypervolume = 0.0
        
        results = {
            'X': pareto_X_orig.cpu().numpy(),
            'F': pareto_Y.cpu().numpy(),
            'all_X': self._unnormalize(self.train_X).cpu().numpy(),
            'all_Y': self.train_Y.cpu().numpy(),
            'all_C': self.train_C.cpu().numpy(),
            'n_evals': self.n_evals,
            'n_pareto': len(pareto_Y),
            'hypervolume': hypervolume,
            'run_time_seconds': total_time,
            'config': self.opt_config,
            'bo_config': self.bo_config,
            'history': self.history
        }
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete")
            print(f"Total evaluations: {self.n_evals}")
            print(f"Pareto solutions: {len(pareto_Y)}")
            print(f"Hypervolume: {hypervolume:.4e}")
            print(f"Run time: {total_time:.1f}s")
            print("=" * 60)
        
        # Save final results
        self._save_results(results)
        
        return results
    
    def _save_checkpoint(self, iteration: int):
        """Save optimization checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'train_X': self.train_X.cpu().numpy().tolist(),
            'train_Y': self.train_Y.cpu().numpy().tolist(),
            'train_C': self.train_C.cpu().numpy().tolist(),
            'n_evals': self.n_evals,
            'history': self.history
        }
        
        path = self.output_dir / f"checkpoint_iter{iteration}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        if self.verbose:
            print(f"Checkpoint saved: {path}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save final optimization results."""
        # Save Pareto front
        np.savez(
            self.output_dir / "pareto_front.npz",
            X=results['X'],
            F=results['F']
        )
        
        # Save all evaluations
        np.savez(
            self.output_dir / "all_evaluations.npz",
            X=results['all_X'],
            Y=results['all_Y'],
            C=results['all_C']
        )
        
        # Save history
        with open(self.output_dir / "optimization_history.json", 'w') as f:
            json.dump(results['history'], f, indent=2)
        
        # Save config
        if hasattr(results['config'], 'save'):
            results['config'].save(str(self.output_dir / "config.json"))
        
        # Save summary
        summary = {
            'n_evals': results['n_evals'],
            'n_pareto': results['n_pareto'],
            'hypervolume': results['hypervolume'],
            'run_time_seconds': results['run_time_seconds']
        }
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save Pareto solutions as CSV
        if len(results['X']) > 0:
            import pandas as pd
            
            # Use config var_names if available
            if hasattr(self.opt_config, 'var_names'):
                var_names = self.opt_config.var_names
            else:
                var_names = ['target_z_cm', 'target_xy_cm', 'gap_cm', 't_min_ns', 't_max_ns']
            obj_names = self.opt_config.objectives
            
            df = pd.DataFrame(results['X'], columns=var_names)
            for i, name in enumerate(obj_names):
                df[name] = results['F'][:, i]
            
            df.to_csv(self.output_dir / "pareto_solutions.csv", index=False)
        
        if self.verbose:
            print(f"Results saved to: {self.output_dir}")
    
    def load_checkpoint(self, path: str):
        """
        Load checkpoint to continue optimization.
        
        Parameters
        ----------
        path : str
            Path to checkpoint JSON file
        """
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.train_X = torch.tensor(
            checkpoint['train_X'],
            dtype=self.dtype,
            device=self.device
        )
        self.train_Y = torch.tensor(
            checkpoint['train_Y'],
            dtype=self.dtype,
            device=self.device
        )
        self.train_C = torch.tensor(
            checkpoint['train_C'],
            dtype=self.dtype,
            device=self.device
        )
        self.n_evals = checkpoint['n_evals']
        self.history = checkpoint['history']
        
        if self.verbose:
            print(f"Loaded checkpoint from: {path}")
            print(f"Evaluations: {self.n_evals}")


def warm_start_from_history(history: List[Dict],
                            opt_config: OptimizationConfig,
                            device: torch.device = None,
                            dtype: torch.dtype = torch.double) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training data from prior evaluation history.
    
    Useful for warm-starting BO from previous NSGA-II or grid search runs.
    
    Parameters
    ----------
    history : List[Dict]
        List of evaluation records with 'X' and 'Y' keys
    opt_config : OptimizationConfig
        Problem configuration
    device : torch.device
        Device for tensors
    dtype : torch.dtype
        Data type for tensors
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (X, Y, C) normalized tensors
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lb = torch.tensor(opt_config.xl, dtype=dtype, device=device)
    ub = torch.tensor(opt_config.xu, dtype=dtype, device=device)
    
    X_list = []
    Y_list = []
    C_list = []
    
    for record in history:
        x = np.array(record['X'])
        y = np.array(record.get('Y', record.get('F', [])))
        c = np.array(record.get('C', record.get('G', [0.0, 0.0])))
        
        if len(y) == opt_config.n_obj:
            X_list.append(x)
            Y_list.append(y)
            C_list.append(c)
    
    if len(X_list) == 0:
        return None, None, None
    
    X = torch.tensor(np.array(X_list), dtype=dtype, device=device)
    Y = torch.tensor(np.array(Y_list), dtype=dtype, device=device)
    C = torch.tensor(np.array(C_list), dtype=dtype, device=device)
    
    # Normalize X
    X_normalized = (X - lb) / (ub - lb)
    
    return X_normalized, Y, C


def run_bayesian_optimization(
    opt_config: OptimizationConfig = None,
    bo_config: BOConfig = None,
    runner=None,
    warm_start_history: List[Dict] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run Bayesian optimization.
    
    Parameters
    ----------
    opt_config : OptimizationConfig
        Problem configuration
    bo_config : BOConfig
        BO configuration
    runner : Geant4Runner or MockGeant4Runner
        Simulation runner
    warm_start_history : List[Dict], optional
        Prior evaluations for warm-starting
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Optimization results
    """
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch is required. Install with: pip install botorch")
    
    optimizer = MultiObjectiveBayesianOptimizer(
        opt_config=opt_config,
        bo_config=bo_config,
        runner=runner,
        verbose=verbose
    )
    
    # Warm start if history provided
    if warm_start_history is not None and len(warm_start_history) > 0:
        X, Y, C = warm_start_from_history(
            warm_start_history,
            optimizer.opt_config,
            optimizer.device,
            optimizer.dtype
        )
        if X is not None:
            optimizer.train_X = X
            optimizer.train_Y = Y
            optimizer.train_C = C
            optimizer.n_evals = len(X)
            if verbose:
                print(f"Warm-started with {len(X)} prior evaluations")
    
    return optimizer.optimize()


if __name__ == "__main__":
    print("Testing Bayesian Optimization Module...")
    
    if not BOTORCH_AVAILABLE:
        print("BoTorch not installed. Skipping test.")
        print("Install with: pip install botorch")
        exit(0)
    
    # Create configurations
    opt_config = OptimizationConfig(
        use_mock=True,
        n_events=100,
        objectives=['signal', 'neutron', 'photon_bkg', 'separability']
    )
    
    bo_config = BOConfig(
        n_initial=10,
        n_iterations=5,
        batch_size=1,
        output_dir="bo_test_output"
    )
    
    print(f"\nConfiguration:")
    print(f"  Variables: {opt_config.n_var}")
    print(f"  Objectives: {opt_config.n_obj} - {opt_config.objectives}")
    print(f"  Initial samples: {bo_config.n_initial}")
    print(f"  BO iterations: {bo_config.n_iterations}")
    
    # Run optimization
    print("\nRunning Bayesian Optimization...")
    results = run_bayesian_optimization(
        opt_config=opt_config,
        bo_config=bo_config,
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"  Total evaluations: {results['n_evals']}")
    print(f"  Pareto solutions: {results['n_pareto']}")
    print(f"  Hypervolume: {results['hypervolume']:.4e}")
    print(f"  Run time: {results['run_time_seconds']:.1f}s")
    
    if len(results['X']) > 0:
        print(f"\nBest Pareto solutions (first 3):")
        for i in range(min(3, len(results['X']))):
            print(f"  X[{i}]: {results['X'][i]}")
            print(f"  F[{i}]: {results['F'][i]}")
    
    print("\nTest complete!")
