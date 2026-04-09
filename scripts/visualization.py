#!/usr/bin/env python3
"""
Visualization and Analysis Tools for DAMSA Optimization Results

This module provides:
1. Pareto front visualization (2D projections, 3D plots, parallel coordinates)
2. Decision space analysis (variable distributions, correlations)
3. Convergence plots (hypervolume over iterations)
4. Trade-off analysis between objectives
5. Solution comparison and ranking
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import json
import warnings

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not installed. Plotting unavailable.")

# For advanced plots
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# Variable and objective names (updated for square target constraint)
VAR_NAMES = ['target_z_cm', 'target_xy_cm', 'gap_cm', 't_min_ns', 't_max_ns']
VAR_LABELS = ['Target Z (cm)', 'Target XY (cm)', 'Vacuum Chamber (cm)',
              't_min (ns)', 't_max (ns)']

OBJ_NAMES_DEFAULT = ['signal', 'neutron', 'photon_bkg', 'separability']
OBJ_LABELS = {
    'signal': 'Signal Rate (neg)',
    'neutron': 'Neutron Background',
    'photon_bkg': 'Photon Background',
    'separability': 'Separability (neg)',
    'em': 'EM Background',
    'snr': 'S/√B (neg)'
}


def load_optimization_results(results_dir: str) -> Dict[str, Any]:
    """
    Load optimization results from directory.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory
        
    Returns
    -------
    dict
        Dictionary with X (variables), F (objectives), history, etc.
    """
    results_dir = Path(results_dir)
    results = {}
    
    # Load Pareto front
    pareto_file = results_dir / "pareto_front.npz"
    if pareto_file.exists():
        data = np.load(pareto_file)
        results['X'] = data['X']
        results['F'] = data['F']
        if 'G' in data:
            results['G'] = data['G']
    
    # Load all evaluations if available
    all_evals_file = results_dir / "all_evaluations.npz"
    if all_evals_file.exists():
        data = np.load(all_evals_file)
        results['all_X'] = data['X']
        results['all_Y'] = data['Y']
        if 'C' in data:
            results['all_C'] = data['C']
    
    # Load CSV version of Pareto solutions
    csv_file = results_dir / "pareto_solutions.csv"
    if csv_file.exists():
        results['pareto_df'] = pd.read_csv(csv_file)
    
    # Load history
    history_file = results_dir / "evaluation_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            results['history'] = json.load(f)
    
    # Load summary
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load config
    config_file = results_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
    
    return results


def plot_pareto_2d(
    F: np.ndarray,
    obj_indices: Tuple[int, int] = (0, 1),
    obj_names: List[str] = None,
    title: str = "Pareto Front (2D Projection)",
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    show_dominated: bool = False,
    all_F: np.ndarray = None
) -> plt.Figure:
    """
    Plot 2D projection of Pareto front.
    
    Parameters
    ----------
    F : np.ndarray
        Pareto front objectives (n_solutions, n_objectives)
    obj_indices : tuple
        Indices of objectives to plot
    obj_names : list
        Names of objectives
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    show_dominated : bool
        Show dominated solutions
    all_F : np.ndarray
        All objective values (for showing dominated)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    i, j = obj_indices
    
    # Plot dominated solutions
    if show_dominated and all_F is not None:
        ax.scatter(all_F[:, i], all_F[:, j], 
                  c='lightgray', s=20, alpha=0.5, label='Dominated')
    
    # Plot Pareto front
    ax.scatter(F[:, i], F[:, j], c='blue', s=50, label='Pareto Front')
    
    # Labels
    if obj_names is None:
        obj_names = OBJ_NAMES_DEFAULT
    
    xlabel = OBJ_LABELS.get(obj_names[i], f'Objective {i}')
    ylabel = OBJ_LABELS.get(obj_names[j], f'Objective {j}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_pareto_3d(
    F: np.ndarray,
    obj_indices: Tuple[int, int, int] = (0, 1, 2),
    obj_names: List[str] = None,
    title: str = "Pareto Front (3D)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None,
    color_by: int = None
) -> plt.Figure:
    """
    Plot 3D visualization of Pareto front.
    
    Parameters
    ----------
    F : np.ndarray
        Pareto front objectives
    obj_indices : tuple
        Indices of 3 objectives to plot
    obj_names : list
        Names of objectives
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    color_by : int
        Index of 4th objective to color by (optional)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    i, j, k = obj_indices
    
    if color_by is not None and color_by < F.shape[1]:
        colors = F[:, color_by]
        sc = ax.scatter(F[:, i], F[:, j], F[:, k], 
                       c=colors, cmap='viridis', s=50)
        plt.colorbar(sc, label=OBJ_LABELS.get(obj_names[color_by] if obj_names else f'Obj {color_by}', 
                                              f'Objective {color_by}'))
    else:
        ax.scatter(F[:, i], F[:, j], F[:, k], c='blue', s=50)
    
    if obj_names is None:
        obj_names = OBJ_NAMES_DEFAULT
    
    ax.set_xlabel(OBJ_LABELS.get(obj_names[i], f'Obj {i}'))
    ax.set_ylabel(OBJ_LABELS.get(obj_names[j], f'Obj {j}'))
    ax.set_zlabel(OBJ_LABELS.get(obj_names[k], f'Obj {k}'))
    ax.set_title(title, fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_pareto_matrix(
    F: np.ndarray,
    obj_names: List[str] = None,
    title: str = "Pareto Front - Pairwise Projections",
    figsize: Tuple[int, int] = None,
    save_path: str = None
) -> plt.Figure:
    """
    Plot matrix of all pairwise objective projections.
    
    Parameters
    ----------
    F : np.ndarray
        Pareto front objectives
    obj_names : list
        Names of objectives
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    n_obj = F.shape[1]
    
    if figsize is None:
        figsize = (3 * n_obj, 3 * n_obj)
    
    if obj_names is None:
        obj_names = OBJ_NAMES_DEFAULT[:n_obj]
    
    fig, axes = plt.subplots(n_obj, n_obj, figsize=figsize)
    
    for i in range(n_obj):
        for j in range(n_obj):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(F[:, i], bins=20, color='blue', alpha=0.7)
                ax.set_xlabel(OBJ_LABELS.get(obj_names[i], f'Obj {i}'))
            else:
                # Off-diagonal: scatter
                ax.scatter(F[:, j], F[:, i], c='blue', s=10, alpha=0.7)
                
                if i == n_obj - 1:
                    ax.set_xlabel(OBJ_LABELS.get(obj_names[j], f'Obj {j}'))
                if j == 0:
                    ax.set_ylabel(OBJ_LABELS.get(obj_names[i], f'Obj {i}'))
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_parallel_coordinates(
    X: np.ndarray,
    F: np.ndarray,
    var_names: List[str] = None,
    obj_names: List[str] = None,
    color_by: str = 'signal',
    title: str = "Parallel Coordinates Plot",
    figsize: Tuple[int, int] = (14, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Plot parallel coordinates for decision variables and objectives.
    
    Parameters
    ----------
    X : np.ndarray
        Decision variables
    F : np.ndarray
        Objectives
    var_names : list
        Variable names
    obj_names : list
        Objective names
    color_by : str
        Objective to color by
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    if var_names is None:
        var_names = VAR_NAMES[:X.shape[1]]
    if obj_names is None:
        obj_names = OBJ_NAMES_DEFAULT[:F.shape[1]]
    
    # Combine variables and objectives
    data = np.hstack([X, F])
    all_names = list(var_names) + list(obj_names)
    
    # Normalize each dimension to [0, 1]
    data_norm = np.zeros_like(data)
    for i in range(data.shape[1]):
        col = data[:, i]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            data_norm[:, i] = (col - min_val) / (max_val - min_val)
        else:
            data_norm[:, i] = 0.5
    
    # Color by objective
    if color_by in obj_names:
        color_idx = obj_names.index(color_by) + X.shape[1]
    else:
        color_idx = X.shape[1]  # First objective
    
    colors = data[:, color_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each solution
    x_coords = np.arange(len(all_names))
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    cmap = cm.viridis
    
    for i in range(len(data)):
        color = cmap(norm(colors[i]))
        ax.plot(x_coords, data_norm[i], c=color, alpha=0.5, linewidth=1)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(OBJ_LABELS.get(color_by, color_by))
    
    # Axis labels
    ax.set_xticks(x_coords)
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.set_title(title, fontsize=14)
    
    # Add vertical lines
    for x in x_coords:
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_convergence(
    history: List[Dict],
    metric: str = 'hypervolume',
    title: str = "Optimization Convergence",
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Plot convergence of optimization over iterations.
    
    Parameters
    ----------
    history : list
        Evaluation history
    metric : str
        Metric to track ('hypervolume', 'best_signal', 'n_pareto')
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    evals = [h['eval_id'] for h in history]
    
    if metric == 'best_signal':
        # Track best (most negative) signal value
        best_so_far = []
        current_best = np.inf
        for h in history:
            if 'Y' in h or 'F' in h:
                obj = h.get('Y', h.get('F', [np.inf]))[0]
                current_best = min(current_best, obj)
            best_so_far.append(current_best)
        
        ax.plot(evals, best_so_far, 'b-', linewidth=2)
        ax.set_ylabel('Best Signal (negative)', fontsize=12)
        
    elif metric == 'n_pareto':
        # This requires tracking Pareto front size over time
        # Simplified: just show evaluation count
        ax.plot(evals, evals, 'b-', linewidth=2)
        ax.set_ylabel('Total Evaluations', fontsize=12)
        
    else:
        # Default: show objective values over time
        if history and ('Y' in history[0] or 'F' in history[0]):
            obj_values = [h.get('Y', h.get('F', [0]))[0] for h in history]
            ax.plot(evals, obj_values, 'b.', alpha=0.5, markersize=3)
            ax.set_ylabel('Objective 1 Value', fontsize=12)
    
    ax.set_xlabel('Evaluation', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_variable_distributions(
    X: np.ndarray,
    var_names: List[str] = None,
    title: str = "Decision Variable Distributions",
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plot distributions of decision variables in Pareto solutions.
    
    Parameters
    ----------
    X : np.ndarray
        Decision variables
    var_names : list
        Variable names
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    if var_names is None:
        var_names = VAR_NAMES[:X.shape[1]]
    
    n_vars = X.shape[1]
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_vars):
        ax = axes[i]
        ax.hist(X[:, i], bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(VAR_LABELS[i] if i < len(VAR_LABELS) else var_names[i])
        ax.set_ylabel('Count')
        ax.axvline(X[:, i].mean(), color='red', linestyle='--', 
                   label=f'Mean: {X[:, i].mean():.2f}')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def analyze_tradeoffs(
    F: np.ndarray,
    obj_names: List[str] = None
) -> pd.DataFrame:
    """
    Analyze trade-offs between objectives.
    
    Parameters
    ----------
    F : np.ndarray
        Pareto front objectives
    obj_names : list
        Names of objectives
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix between objectives
    """
    if obj_names is None:
        obj_names = OBJ_NAMES_DEFAULT[:F.shape[1]]
    
    df = pd.DataFrame(F, columns=obj_names)
    corr = df.corr()
    
    return corr


def rank_solutions(
    X: np.ndarray,
    F: np.ndarray,
    weights: Dict[str, float] = None,
    obj_names: List[str] = None,
    method: str = 'weighted_sum'
) -> pd.DataFrame:
    """
    Rank Pareto solutions by combined metric.
    
    Parameters
    ----------
    X : np.ndarray
        Decision variables
    F : np.ndarray
        Objectives
    weights : dict
        Weights for each objective (positive = better when smaller)
    obj_names : list
        Names of objectives
    method : str
        Ranking method ('weighted_sum', 'topsis')
        
    Returns
    -------
    pd.DataFrame
        Ranked solutions with scores
    """
    if obj_names is None:
        obj_names = OBJ_NAMES_DEFAULT[:F.shape[1]]
    
    if weights is None:
        weights = {name: 1.0 for name in obj_names}
    
    # Normalize objectives
    F_norm = np.zeros_like(F)
    for i in range(F.shape[1]):
        col = F[:, i]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            F_norm[:, i] = (col - min_val) / (max_val - min_val)
        else:
            F_norm[:, i] = 0.0
    
    if method == 'weighted_sum':
        # Weighted sum (lower = better since all objectives are minimized)
        scores = np.zeros(len(F))
        for i, name in enumerate(obj_names):
            w = weights.get(name, 1.0)
            scores += w * F_norm[:, i]
    
    elif method == 'topsis':
        # TOPSIS method
        # Ideal: minimum of each objective
        # Anti-ideal: maximum of each objective
        
        ideal = F_norm.min(axis=0)
        anti_ideal = F_norm.max(axis=0)
        
        dist_to_ideal = np.sqrt(np.sum((F_norm - ideal)**2, axis=1))
        dist_to_anti = np.sqrt(np.sum((F_norm - anti_ideal)**2, axis=1))
        
        scores = dist_to_ideal / (dist_to_ideal + dist_to_anti + 1e-10)
    
    else:
        scores = F_norm[:, 0]  # Default to first objective
    
    # Create dataframe
    var_names = VAR_NAMES[:X.shape[1]]
    df = pd.DataFrame(X, columns=var_names)
    
    for i, name in enumerate(obj_names):
        df[name] = F[:, i]
    
    df['score'] = scores
    df['rank'] = df['score'].rank(ascending=True).astype(int)
    
    return df.sort_values('rank')


def generate_report(
    results_dir: str,
    output_dir: str = None,
    obj_names: List[str] = None,
    flux_file: str = None,
    alp_mass_MeV: float = 100.0,
    alp_coupling: float = 1e-3,
    generate_root_plots: bool = True,
    generate_alp_plots: bool = True
) -> None:
    """
    Generate comprehensive analysis report including ROOT and ALP signal plots.
    
    Parameters
    ----------
    results_dir : str
        Path to optimization results
    output_dir : str
        Path for output plots (defaults to results_dir/analysis)
    obj_names : list
        Names of objectives
    flux_file : str
        Path to photon flux file for ALP calculations (auto-detected if None)
    alp_mass_MeV : float
        ALP mass for signal calculations
    alp_coupling : float
        ALP-photon coupling for signal calculations
    generate_root_plots : bool
        Generate ROOT summary plots for top configurations
    generate_alp_plots : bool
        Generate ALP signal visualization plots
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for report generation")
    
    results = load_optimization_results(results_dir)
    
    if output_dir is None:
        output_dir = Path(results_dir) / "analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    X = results.get('X')
    F = results.get('F')
    
    if X is None or F is None:
        print("No Pareto front data found")
        return
    
    if obj_names is None:
        if 'config' in results and 'objectives' in results['config']:
            obj_names = results['config']['objectives']
        else:
            obj_names = OBJ_NAMES_DEFAULT[:F.shape[1]]
    
    print(f"Generating analysis report for {len(X)} Pareto solutions...")
    
    # 1. Pareto front projections
    n_obj = F.shape[1]
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            fig = plot_pareto_2d(
                F, obj_indices=(i, j), obj_names=obj_names,
                title=f"Pareto: {obj_names[i]} vs {obj_names[j]}",
                save_path=str(output_dir / f"pareto_2d_{i}_{j}.png")
            )
            plt.close(fig)
    
    # 2. Pareto matrix
    fig = plot_pareto_matrix(
        F, obj_names=obj_names,
        save_path=str(output_dir / "pareto_matrix.png")
    )
    plt.close(fig)
    
    # 3. 3D plot (if 3+ objectives)
    if n_obj >= 3:
        fig = plot_pareto_3d(
            F, obj_indices=(0, 1, 2), obj_names=obj_names,
            color_by=3 if n_obj > 3 else None,
            save_path=str(output_dir / "pareto_3d.png")
        )
        plt.close(fig)
    
    # 4. Parallel coordinates
    fig = plot_parallel_coordinates(
        X, F, obj_names=obj_names,
        save_path=str(output_dir / "parallel_coords.png")
    )
    plt.close(fig)
    
    # 5. Variable distributions
    fig = plot_variable_distributions(
        X, save_path=str(output_dir / "variable_distributions.png")
    )
    plt.close(fig)
    
    # 6. Convergence (if history available)
    if 'history' in results and results['history']:
        fig = plot_convergence(
            results['history'],
            save_path=str(output_dir / "convergence.png")
        )
        plt.close(fig)
    
    # 7. Trade-off analysis
    corr = analyze_tradeoffs(F, obj_names)
    corr.to_csv(output_dir / "objective_correlations.csv")
    
    # 8. Ranked solutions
    ranked = rank_solutions(X, F, obj_names=obj_names)
    ranked.to_csv(output_dir / "ranked_solutions.csv", index=False)
    
    # 9. Summary statistics
    summary_stats = {
        'n_solutions': len(X),
        'variable_ranges': {},
        'objective_ranges': {}
    }
    
    for i, name in enumerate(VAR_NAMES[:X.shape[1]]):
        summary_stats['variable_ranges'][name] = {
            'min': float(X[:, i].min()),
            'max': float(X[:, i].max()),
            'mean': float(X[:, i].mean()),
            'std': float(X[:, i].std())
        }
    
    for i, name in enumerate(obj_names):
        summary_stats['objective_ranges'][name] = {
            'min': float(F[:, i].min()),
            'max': float(F[:, i].max()),
            'mean': float(F[:, i].mean()),
            'std': float(F[:, i].std())
        }
    
    with open(output_dir / "summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # 10. ROOT summary plots for top configurations
    if generate_root_plots:
        print("\nGenerating ROOT summary plots...")
        try:
            generate_root_summary_plots(
                results_dir=results_dir,
                output_dir=output_dir / "root_summaries",
                top_n=5
            )
        except Exception as e:
            print(f"  ROOT plot generation failed: {e}")
    
    # 11. ALP signal visualization plots
    if generate_alp_plots:
        print("\nGenerating ALP signal plots...")
        try:
            # Auto-detect flux file if not provided
            if flux_file is None:
                flux_file = _find_flux_file(results_dir)
            
            if flux_file:
                generate_alp_signal_plots(
                    flux_file=flux_file,
                    output_dir=output_dir / "alp_signals",
                    alp_mass_MeV=alp_mass_MeV,
                    coupling=alp_coupling
                )
            else:
                print("  No flux file found, skipping ALP signal plots")
        except Exception as e:
            print(f"  ALP signal plot generation failed: {e}")
    
    print(f"\nAnalysis report saved to: {output_dir}")
    print(f"Generated files:")
    for f in sorted(output_dir.glob("**/*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}")


def _find_flux_file(results_dir: str) -> Optional[str]:
    """Auto-detect photon flux file in results directory."""
    results_dir = Path(results_dir)
    
    # Common flux file patterns
    patterns = [
        "photon_flux.csv",
        "flux_data.csv",
        "**/photon_flux*.csv",
        "**/flux*.csv",
        "../photon_flux.csv",
        "../../photon_flux.csv"
    ]
    
    for pattern in patterns:
        matches = list(results_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    # Check parent directories
    parent = results_dir.parent
    for _ in range(3):
        flux_file = parent / "photon_flux.csv"
        if flux_file.exists():
            return str(flux_file)
        parent = parent.parent
    
    return None


def generate_root_summary_plots(
    results_dir: str,
    output_dir: str,
    top_n: int = 5
) -> None:
    """
    Generate ROOT summary plots for top Pareto solutions.
    
    Parameters
    ----------
    results_dir : str
        Path to optimization results
    output_dir : str
        Output directory for ROOT plots
    top_n : int
        Number of top solutions to generate plots for
    """
    # Import ROOT plotting module
    try:
        from root_config_plots import ConfigurationPlotter, ConfigurationComparator
    except ImportError:
        print("  root_config_plots module not available")
        return
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Pareto solutions
    pareto_csv = results_dir / "pareto_solutions.csv"
    if not pareto_csv.exists():
        print(f"  {pareto_csv} not found")
        return
    
    df = pd.read_csv(pareto_csv)
    
    # Get top N solutions (by rank/score if available)
    if 'rank' in df.columns:
        top_df = df.nsmallest(top_n, 'rank')
    elif 'score' in df.columns:
        top_df = df.nsmallest(top_n, 'score')
    else:
        top_df = df.head(top_n)
    
    print(f"  Generating plots for top {len(top_df)} configurations")
    
    # Create configurations from Pareto solutions
    configs = []
    for idx, row in top_df.iterrows():
        config = {
            'target_z': row.get('target_z_cm', row.get('target_z', 10.0)),
            'target_xy': row.get('target_xy_cm', row.get('target_xy', 5.0)),
            'gap': row.get('gap_cm', row.get('gap', 40.0)),
            'config_id': idx
        }
        configs.append(config)
    
    # Look for simulation data files
    data_files = []
    for config in configs:
        # Try to find corresponding data file
        data_patterns = [
            results_dir / f"sim_data_{config['config_id']}.csv",
            results_dir / "simulation_data" / f"config_{config['config_id']}.csv",
            results_dir.parent / "flux_data.csv"
        ]
        
        data_file = None
        for pattern in data_patterns:
            if pattern.exists():
                data_file = str(pattern)
                break
        
        if data_file:
            data_files.append(data_file)
    
    if not data_files:
        print("  No simulation data files found for ROOT plots")
        # Generate placeholder plots with configuration info
        _generate_config_info_plots(configs, output_dir)
        return
    
    # Generate comparison plots
    comparator = ConfigurationComparator(configs[:len(data_files)], str(output_dir))
    comparator.load_all_data(data_files)
    comparator.create_comparison_plots()
    comparator.create_summary_table()
    comparator.save_all_to_root("config_comparison.root")
    
    print(f"  ROOT plots saved to {output_dir}")


def _generate_config_info_plots(configs: List[Dict], output_dir: Path) -> None:
    """Generate configuration summary plots without simulation data."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary figure showing configuration parameters
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    config_ids = [c.get('config_id', i) for i, c in enumerate(configs)]
    target_z = [c['target_z'] for c in configs]
    target_xy = [c['target_xy'] for c in configs]
    gap = [c['gap'] for c in configs]
    
    axes[0].bar(range(len(configs)), target_z)
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Target Z (cm)')
    axes[0].set_title('Target Thickness')
    
    axes[1].bar(range(len(configs)), target_xy)
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Target XY (cm)')
    axes[1].set_title('Target Cross-section')
    
    axes[2].bar(range(len(configs)), gap)
    axes[2].set_xlabel('Configuration')
    axes[2].set_ylabel('Gap (cm)')
    axes[2].set_title('Vacuum Chamber Length')
    
    plt.tight_layout()
    fig.savefig(output_dir / "config_parameters.png", dpi=150)
    plt.close(fig)
    
    print(f"  Configuration summary saved to {output_dir}/config_parameters.png")


def generate_alp_signal_plots(
    flux_file: str,
    output_dir: str,
    alp_mass_MeV: float = 100.0,
    coupling: float = 1e-3
) -> None:
    """
    Generate ALP signal visualization plots.
    
    Parameters
    ----------
    flux_file : str
        Path to photon flux CSV file
    output_dir : str
        Output directory for plots
    alp_mass_MeV : float
        ALP mass in MeV
    coupling : float
        ALP-photon coupling
    """
    # Import ALP plotting module
    try:
        from alplib_signal_plots import ALPSignalVisualizer
    except ImportError:
        print("  alplib_signal_plots module not available")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ALPSignalVisualizer(str(output_dir))
    
    # Generate full report
    summary = visualizer.generate_full_report(
        flux_file=flux_file,
        alp_mass_MeV=alp_mass_MeV,
        coupling=coupling
    )
    
    print(f"  ALP signal plots saved to {output_dir}")
    print(f"  Expected events: {summary.get('n_events', 'N/A')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze DAMSA optimization results")
    parser.add_argument("results_dir", help="Path to optimization results directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory for plots")
    parser.add_argument("--objectives", nargs="+", default=None, help="Objective names")
    parser.add_argument("--flux-file", "-f", default=None, 
                       help="Path to photon flux file for ALP calculations")
    parser.add_argument("--alp-mass", type=float, default=100.0,
                       help="ALP mass in MeV (default: 100)")
    parser.add_argument("--alp-coupling", type=float, default=1e-3,
                       help="ALP-photon coupling (default: 1e-3)")
    parser.add_argument("--no-root-plots", action="store_true",
                       help="Skip ROOT summary plot generation")
    parser.add_argument("--no-alp-plots", action="store_true",
                       help="Skip ALP signal plot generation")
    
    args = parser.parse_args()
    
    generate_report(
        args.results_dir,
        output_dir=args.output,
        obj_names=args.objectives,
        flux_file=args.flux_file,
        alp_mass_MeV=args.alp_mass,
        alp_coupling=args.alp_coupling,
        generate_root_plots=not args.no_root_plots,
        generate_alp_plots=not args.no_alp_plots
    )
