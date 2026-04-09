#!/usr/bin/env python3
"""
GAP Pareto Optimization Scan for DAMSA

Scans decay chamber gap (distance from target exit to calorimeter entrance)
from 42 cm to 52 cm in 1 cm steps, calculating:
- ALP signal using target exit photons (via alplib Primakoff production)
- Separability: fraction of ALP decays with opening angle >= 10 deg OR
  (opening angle < 10 deg AND both photon energies > 100 MeV)
- Weighted background at calorimeter entrance: photons + 10*neutrons

Generates Pareto front visualization: maximize separability, minimize weighted background.

Usage:
    python gap_pareto_scan.py
    python gap_pareto_scan.py --n-events 10000 --alp-mass 20 --coupling 1e-4
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import flux loading from alplib_signal_plots
from scripts.alplib_signal_plots import load_flux_for_alplib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GAP Pareto optimization scan"
    )
    parser.add_argument('--n-events', type=int, default=10000,
                        help='Number of events per gap configuration (default: 10000)')
    parser.add_argument('--alp-mass', type=float, default=10.0,
                        help='ALP mass in MeV (default: 10, must be < max photon energy ~42 MeV)')
    parser.add_argument('--target-z', type=float, default=10.0,
                        help='Target Z length in cm (default: 10)')
    parser.add_argument('--target-xy', type=float, default=5.0,
                        help='Target XY size in cm (default: 5)')
    parser.add_argument('--beam-current', type=float, default=62.5,
                        help='Beam current in uA (default: 62.5)')
    parser.add_argument('--exposure', type=float, default=0.0417,
                        help='Exposure time in days for ALP calculation (default: 0.0417 = 1 hour)')
    parser.add_argument('--coupling', type=float, default=1e-5,
                        help='ALP coupling in GeV^-1 (default: 1e-4, good statistics for optimization)')
    parser.add_argument('--angle-cut-high', type=float, default=10.0,
                        help='High opening angle cut (degrees) - separable if >= this (default: 10)')
    parser.add_argument('--energy-cut', type=float, default=100.0,
                        help='Energy cut for separability in MeV (default: 100)')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, use existing data if available')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing results')
    return parser.parse_args()


def create_output_directory(base_dir: str = "output/gap_optimization") -> Path:
    """Create output directory for gap optimization results."""
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_simulation(gap_cm: float, n_events: int, target_z: float, target_xy: float,
                   optimize: bool = True) -> Dict[str, Any]:
    """
    Run Geant4 simulation for a given gap distance.
    
    Parameters
    ----------
    gap_cm : float
        Gap distance in cm
    n_events : int
        Number of events to simulate
    target_z : float
        Target Z length in cm
    target_xy : float
        Target XY size in cm
    optimize : bool
        Use config-specific filenames
    
    Returns
    -------
    Dict with simulation results
    """
    # Build command
    cmd = [
        './build/damsa_opt',
        '--target-z', str(target_z),
        '--target-xy', str(target_xy),
        '--gap', str(gap_cm),
        '--n-events', str(n_events),
    ]
    
    if optimize:
        cmd.append('--optimize')
    
    print(f"\n{'='*60}")
    print(f"Running simulation: gap = {gap_cm} cm")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Run simulation
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"ERROR: Simulation failed for gap = {gap_cm}")
        print(result.stderr)
        return None
    
    return {
        'gap': gap_cm,
        'return_code': result.returncode
    }


def extract_simulation_results(gap_cm: float, output_dir: Path, 
                               target_z: float = 10.0, target_xy: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Extract results from simulation output files.
    
    Uses target exit photons for alplib signal calculation.
    Uses calorimeter entrance data for background and Pareto optimization.
    
    Parameters
    ----------
    gap_cm : float
        Gap distance in cm
    output_dir : Path
        Output directory path
    target_z : float
        Target Z length in cm (for config prefix)
    target_xy : float
        Target XY size in cm (for config prefix)
    
    Returns
    -------
    Dict with extracted metrics including:
        - photons_target_exit: For alplib signal calculation
        - photons_calo_entrance: For weighted background
        - neutrons_calo_entrance: For weighted background
        - weighted_background: photons + 10*neutrons at calo entrance
    """
    # Expected output files (with optimize prefix)
    config_prefix = f"Tz{target_z:.0f}_xy{target_xy:.0f}_G{gap_cm:.0f}_"
    
    photon_flux_file = output_dir / f"{config_prefix}photon_flux_target_exit.csv"
    sim_info_file = output_dir / f"{config_prefix}simulation_info.json"
    
    results = {'gap': gap_cm}
    
    # Extract target exit photons (for alplib signal)
    if photon_flux_file.exists():
        df = pd.read_csv(photon_flux_file)
        results['photons_target_exit'] = len(df)
        if 'energy_MeV' in df.columns:
            results['max_photon_energy'] = df['energy_MeV'].max()
        elif 'energy' in df.columns:
            results['max_photon_energy'] = df['energy'].max()
        else:
            results['max_photon_energy'] = 0
    else:
        print(f"WARNING: Photon flux file not found: {photon_flux_file}")
        results['photons_target_exit'] = 0
        results['max_photon_energy'] = 0
    
    # Extract simulation info (includes both target exit and calo entrance data)
    if sim_info_file.exists():
        with open(sim_info_file, 'r') as f:
            sim_info = json.load(f)
            
            # Target exit data (for alplib signal)
            results['exit_photons'] = sim_info.get('exit_photons', 0)
            results['exit_neutrons'] = sim_info.get('exit_neutrons', 0)
            
            # Calorimeter entrance data (for Pareto optimization)
            results['photons_calo_entrance'] = sim_info.get('calo_entrance_photons', 0)
            results['neutrons_calo_entrance'] = sim_info.get('calo_entrance_neutrons', 0)
            
            results['run_time'] = sim_info.get('run_time_seconds', 0)
    else:
        print(f"WARNING: Simulation info file not found: {sim_info_file}")
        # Fallback to target exit data (not ideal)
        results['exit_photons'] = results.get('photons_target_exit', 0)
        results['exit_neutrons'] = 0
        results['photons_calo_entrance'] = 0
        results['neutrons_calo_entrance'] = 0
        results['run_time'] = 0
    
    # Calculate weighted background: photons + 10*neutrons at calo entrance
    results['weighted_background'] = (
        results['photons_calo_entrance'] + 10 * results['neutrons_calo_entrance']
    )
    
    return results


def calculate_separability(opening_angles_deg: np.ndarray, 
                           photon_energies_1: np.ndarray,
                           photon_energies_2: np.ndarray,
                           weights: np.ndarray,
                           angle_cut_high: float = 10.0,
                           energy_cut_MeV: float = 100.0) -> float:
    """
    Calculate separability fraction based on opening angle and energy.
    
    Separable if:
    - opening_angle >= angle_cut_high (10 deg), OR
    - opening_angle < angle_cut_high (10 deg) AND both photon energies > energy_cut (100 MeV)
    
    Parameters
    ----------
    opening_angles_deg : array
        Opening angles between decay photons in degrees
    photon_energies_1 : array
        First photon energies in MeV
    photon_energies_2 : array
        Second photon energies in MeV
    weights : array
        Event weights for proper normalization
    angle_cut_high : float
        High opening angle cut in degrees (default: 10)
    energy_cut_MeV : float
        Energy cut in MeV for low-angle separability (default: 100)
    
    Returns
    -------
    Separability fraction (0 to 1)
    """
    if len(opening_angles_deg) == 0:
        return 0.0
    
    # Condition 1: High opening angle (easy to resolve two photons)
    high_angle = opening_angles_deg >= angle_cut_high
    
    # Condition 2: Low opening angle but both photons have high energy
    low_angle_high_energy = (
        (opening_angles_deg < angle_cut_high) & 
        (photon_energies_1 > energy_cut_MeV) & 
        (photon_energies_2 > energy_cut_MeV)
    )
    
    # Separable = either condition satisfied
    separable = high_angle | low_angle_high_energy
    
    # Weighted fraction
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0
    
    return np.sum(weights[separable]) / total_weight


def calculate_alp_signal_with_separability(
    photon_flux_file: Path, 
    n_primaries: int,
    beam_current: float, 
    exposure_days: float,
    coupling: float = 1e-5,
    alp_mass: float = 10.0,
    detector_distance_m: float = 1.0,
    angle_cut_high: float = 10.0,
    energy_cut_MeV: float = 100.0
) -> Dict[str, Any]:
    """
    Calculate ALP signal and separability using alplib.
    
    Uses target exit photon flux for ALP production via Primakoff effect.
    
    Parameters
    ----------
    photon_flux_file : Path
        Path to photon flux CSV (target exit)
    n_primaries : int
        Number of primary events simulated
    beam_current : float
        Beam current in uA
    exposure_days : float
        Exposure time in days
    coupling : float
        ALP coupling in GeV^-1 (default: 1e-5)
    alp_mass : float
        ALP mass in MeV (default: 10.0)
    detector_distance_m : float
        Detector distance in meters
    angle_cut_high : float
        High opening angle cut in degrees (default: 10)
    energy_cut_MeV : float
        Energy cut for low-angle separability
    
    Returns
    -------
    dict with:
        - n_events: Total ALP decay events
        - separability: Fraction of separable decays
        - opening_angles_deg: Array of opening angles
        - energy1_MeV: Array of first photon energies
        - energy2_MeV: Array of second photon energies
    """
    # Convert coupling from GeV^-1 to MeV^-1 (alplib uses MeV^-1)
    coupling_mev = coupling / 1000.0
    
    print(f"  ALP mass: m_a = {alp_mass:.2f} MeV")
    print(f"  Coupling: g = {coupling:.2e} GeV^-1 = {coupling_mev:.2e} MeV^-1")
    
    try:
        # Import alplib components
        import alplib.fluxes as alplib_fluxes
        import alplib.materials as alplib_materials
        import alplib.generators as alplib_generators
        
        # Load flux with beam scaling (target exit photons)
        flux_array = load_flux_for_alplib(str(photon_flux_file), n_primaries, beam_current)
        
        # Create target material
        target_material = alplib_materials.Material("W")
        detector_material = alplib_materials.Material("CsI")
        
        # Create Primakoff flux generator
        flux_obj = alplib_fluxes.FluxPrimakoffIsotropic(
            photon_flux=flux_array,
            target=target_material,
            det_dist=detector_distance_m,
            det_length=1.0,  # 1 meter detector
            det_area=0.25,  # 50x50 cm detector
            axion_mass=alp_mass,
            axion_coupling=coupling_mev,
            n_samples=1000
        )
        
        # Simulate ALP production
        flux_obj.simulate()
        
        # Propagate ALPs
        flux_obj.propagate()
        
        # Create event generator
        generator = alplib_generators.PhotonEventGenerator(flux_obj, detector_material)
        
        # Get total decay events for exposure period
        n_events = generator.decays(
            days_exposure=exposure_days,
            threshold=0.1  # MeV threshold
        )
        
        # Get decay kinematics for separability calculation
        # Use simulate_decay_4vectors to get photon 4-momenta
        try:
            p4_1, p4_2, weights = generator.simulate_decay_4vectors(
                days_exposure=exposure_days,
                n_samples=1000
            )
            
            if len(p4_1) > 0:
                # Extract energies (p0 component of 4-vector)
                energy1 = np.array([p.p0 for p in p4_1])  # MeV
                energy2 = np.array([p.p0 for p in p4_2])  # MeV
                
                # Extract momenta for opening angle calculation
                px1 = np.array([p.p1 for p in p4_1])
                py1 = np.array([p.p2 for p in p4_1])
                pz1 = np.array([p.p3 for p in p4_1])
                
                px2 = np.array([p.p1 for p in p4_2])
                py2 = np.array([p.p2 for p in p4_2])
                pz2 = np.array([p.p3 for p in p4_2])
                
                # Compute opening angle from momentum dot product
                p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)
                p2_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
                dot = px1*px2 + py1*py2 + pz1*pz2
                cos_angle = np.clip(dot / (p1_mag * p2_mag + 1e-10), -1, 1)
                opening_angles_rad = np.arccos(cos_angle)
                opening_angles_deg = np.degrees(opening_angles_rad)
                
                weights_arr = np.array(weights)
                
                # Calculate separability
                separability = calculate_separability(
                    opening_angles_deg, energy1, energy2, weights_arr,
                    angle_cut_high, energy_cut_MeV
                )
                
                return {
                    'n_events': n_events,
                    'separability': separability,
                    'opening_angles_deg': opening_angles_deg,
                    'energy1_MeV': energy1,
                    'energy2_MeV': energy2,
                    'weights': weights_arr,
                    'mean_opening_angle': np.mean(opening_angles_deg),
                    'mean_photon_energy': np.mean(np.concatenate([energy1, energy2]))
                }
            else:
                print("  WARNING: No decay events generated for separability")
                
        except AttributeError:
            # simulate_decay_4vectors not available in this alplib version
            # Fall back to theoretical estimate
            print("  WARNING: simulate_decay_4vectors not available, using theoretical separability")
        
        # Fallback: estimate separability from ALP kinematics
        # Opening angle ~ 2 * m_a / E_a for boosted decays
        # Assume typical ALP energy ~ 2 * m_a (modest boost)
        typical_alp_energy = 2 * alp_mass
        estimated_opening_angle = 2 * alp_mass / typical_alp_energy  # radians (= 1 rad = 57 deg)
        estimated_angle_deg = np.degrees(estimated_opening_angle)
        
        # Rough separability estimate based on opening angle
        if estimated_angle_deg >= angle_cut_high:
            separability = 1.0
        else:
            # Low angle - check if energies might help
            separability = 0.5  # Conservative estimate
        
        return {
            'n_events': n_events,
            'separability': separability,
            'opening_angles_deg': np.array([estimated_angle_deg]),
            'energy1_MeV': np.array([typical_alp_energy / 2]),
            'energy2_MeV': np.array([typical_alp_energy / 2]),
            'weights': np.array([1.0]),
            'mean_opening_angle': estimated_angle_deg,
            'mean_photon_energy': typical_alp_energy / 2
        }
        
    except ImportError as e:
        print(f"  WARNING: alplib not available: {e}")
        return {
            'n_events': 0.0,
            'separability': 0.0,
            'opening_angles_deg': np.array([]),
            'energy1_MeV': np.array([]),
            'energy2_MeV': np.array([]),
            'weights': np.array([]),
            'mean_opening_angle': 0.0,
            'mean_photon_energy': 0.0
        }
    except Exception as e:
        print(f"  WARNING: ALP signal calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'n_events': 0.0,
            'separability': 0.0,
            'opening_angles_deg': np.array([]),
            'energy1_MeV': np.array([]),
            'energy2_MeV': np.array([]),
            'weights': np.array([]),
            'mean_opening_angle': 0.0,
            'mean_photon_energy': 0.0
        }


def identify_pareto_front(separability: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Identify Pareto-optimal points.
    
    Pareto optimal = no other point has BOTH higher separability AND lower background.
    
    Parameters
    ----------
    separability : array
        Separability fractions (to maximize)
    background : array
        Weighted background counts (to minimize)
    
    Returns
    -------
    Boolean array indicating Pareto-optimal points
    """
    n = len(separability)
    pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if j has >= separability AND <= background
                # with at least one strict inequality
                if (separability[j] >= separability[i] and 
                    background[j] <= background[i] and
                    (separability[j] > separability[i] or background[j] < background[i])):
                    pareto[i] = False
                    break
    
    return pareto


def run_gap_pareto_scan(args):
    """Main function to run gap Pareto optimization scan."""
    
    output_dir = create_output_directory()
    
    # Gap values: 42 to 52 cm (1 cm steps)
    gap_values = list(range(42, 53))
    
    # Convert exposure to hours for display
    exposure_hours = args.exposure * 24
    
    print(f"\n{'#'*60}")
    print(f"# GAP PARETO OPTIMIZATION SCAN")
    print(f"#")
    print(f"# Target: {args.target_z} cm Z x {args.target_xy} cm XY")
    print(f"# Gap range: {gap_values[0]} to {gap_values[-1]} cm ({len(gap_values)} values)")
    print(f"# Events per config: {args.n_events}")
    print(f"# ALP mass: {args.alp_mass} MeV")
    print(f"# Coupling: {args.coupling:.2e} GeV^-1")
    print(f"# Exposure: {exposure_hours:.1f} hours ({args.exposure:.4f} days)")
    print(f"#")
    print(f"# Separability criteria:")
    print(f"#   - Opening angle >= {args.angle_cut_high} deg, OR")
    print(f"#   - Opening angle < {args.angle_cut_high} deg AND both photons > {args.energy_cut} MeV")
    print(f"#")
    print(f"# Weighted background: photons + 10*neutrons @ calo entrance")
    print(f"#")
    print(f"# Output directory: {output_dir}")
    print(f"{'#'*60}")
    
    results = []
    
    for i, gap in enumerate(gap_values):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(gap_values)}: gap = {gap} cm")
        print(f"{'='*60}")
        
        # Run simulation (unless skip_sim is set)
        if not args.skip_sim:
            sim_result = run_simulation(
                gap_cm=gap,
                n_events=args.n_events,
                target_z=args.target_z,
                target_xy=args.target_xy,
                optimize=True
            )
            
            if sim_result is None:
                print(f"ERROR: Simulation failed for gap = {gap}")
                continue
        
        # Extract results from simulation output
        sim_results = extract_simulation_results(
            gap, Path('output'), args.target_z, args.target_xy
        )
        
        if sim_results is None:
            print(f"WARNING: Could not extract results for gap = {gap}")
            continue
        
        # Calculate ALP signal with separability (using target exit photons)
        config_prefix = f"Tz{args.target_z:.0f}_xy{args.target_xy:.0f}_G{gap:.0f}_"
        photon_flux_file = Path('output') / f"{config_prefix}photon_flux_target_exit.csv"
        
        if photon_flux_file.exists():
            alp_result = calculate_alp_signal_with_separability(
                photon_flux_file,
                args.n_events,
                args.beam_current,
                args.exposure,
                coupling=args.coupling,
                alp_mass=args.alp_mass,
                detector_distance_m=gap / 100.0,  # Convert cm to m
                angle_cut_high=args.angle_cut_high,
                energy_cut_MeV=args.energy_cut
            )
        else:
            print(f"  WARNING: Photon flux file not found: {photon_flux_file}")
            alp_result = {
                'n_events': 0.0,
                'separability': 0.0,
                'mean_opening_angle': 0.0,
                'mean_photon_energy': 0.0
            }
        
        # Store ALP results
        sim_results['alp_events'] = alp_result['n_events']
        sim_results['separability'] = alp_result['separability']
        sim_results['mean_opening_angle'] = alp_result.get('mean_opening_angle', 0.0)
        sim_results['mean_photon_energy'] = alp_result.get('mean_photon_energy', 0.0)
        sim_results['alp_mass_MeV'] = args.alp_mass
        sim_results['coupling_GeV'] = args.coupling
        
        results.append(sim_results)
        
        print(f"  Results:")
        print(f"    Target exit photons: {sim_results.get('photons_target_exit', 0)}")
        print(f"    Calo entrance photons: {sim_results.get('photons_calo_entrance', 0)}")
        print(f"    Calo entrance neutrons: {sim_results.get('neutrons_calo_entrance', 0)}")
        print(f"    Weighted background: {sim_results.get('weighted_background', 0)}")
        print(f"    ALP events: {alp_result['n_events']:.4f}")
        print(f"    Separability: {alp_result['separability']:.1%}")
        print(f"    Mean opening angle: {alp_result.get('mean_opening_angle', 0):.1f} deg")
    
    if len(results) == 0:
        print("ERROR: No results collected")
        return None
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_file = output_dir / 'gap_pareto_results.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}")
    
    # Generate Pareto visualization
    generate_pareto_plot(results_df, output_dir, args)
    
    print(f"\n{'#'*60}")
    print(f"# SCAN COMPLETE")
    print(f"#")
    print(f"# Results: {csv_file}")
    print(f"# Plots: {output_dir / 'gap_pareto_front.png'}")
    print(f"{'#'*60}")
    
    return results_df


def generate_pareto_plot(results_df: pd.DataFrame, output_dir: Path, args):
    """
    Generate Pareto front visualization.
    
    Pareto objectives:
    - Maximize: separability (fraction of ALP decays distinguishable from background)
    - Minimize: weighted_background (photons + 10*neutrons at calo entrance)
    """
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot generation")
        return
    
    # Convert exposure to hours for display
    exposure_hours = args.exposure * 24
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'DAMSA Gap Pareto Optimization\n'
                 f'(Target: {args.target_z}×{args.target_xy} cm, '
                 f'm_a = {args.alp_mass:.1f} MeV, '
                 f'g = {args.coupling:.0e} GeV⁻¹, '
                 f'{exposure_hours:.1f} hr exposure)', 
                 fontsize=12, fontweight='bold')
    
    # ========== Plot 1: Pareto Front (main plot) ==========
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(
        results_df['weighted_background'],  # X: minimize (left is better)
        results_df['separability'],          # Y: maximize (up is better)
        c=results_df['gap'], cmap='viridis', s=150, edgecolors='black'
    )
    ax1.set_xlabel('Weighted Background (photons + 10×neutrons)', fontsize=11)
    ax1.set_ylabel('Separability Fraction', fontsize=11)
    ax1.set_title('Pareto Front: Separability vs Background', fontsize=11)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Gap (cm)', fontsize=10)
    
    # Highlight Pareto-optimal points
    if len(results_df) > 1:
        pareto_mask = identify_pareto_front(
            results_df['separability'].values,
            results_df['weighted_background'].values
        )
        pareto_df = results_df[pareto_mask]
        ax1.scatter(
            pareto_df['weighted_background'],
            pareto_df['separability'],
            s=300, facecolors='none', edgecolors='red', linewidths=3,
            label=f'Pareto optimal ({pareto_mask.sum()} points)'
        )
        
        # Annotate Pareto-optimal points with gap values
        for _, row in pareto_df.iterrows():
            ax1.annotate(f"{row['gap']:.0f}", 
                        (row['weighted_background'], row['separability']),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=9, color='red')
        
        ax1.legend(loc='lower right')
    
    # Add arrow indicating optimization direction
    ax1.annotate('', xy=(0.1, 0.9), xytext=(0.3, 0.7),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(0.15, 0.85, 'Better', transform=ax1.transAxes, 
             fontsize=10, color='green', fontweight='bold')
    
    # ========== Plot 2: Gap vs Separability ==========
    ax2 = axes[0, 1]
    ax2.plot(results_df['gap'], results_df['separability'], 'b-o', markersize=8, linewidth=2)
    ax2.set_xlabel('Gap Distance (cm)', fontsize=11)
    ax2.set_ylabel('Separability Fraction', fontsize=11)
    ax2.set_title('Separability vs Gap Distance', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Mark best separability
    best_sep_idx = results_df['separability'].idxmax()
    ax2.scatter(results_df.loc[best_sep_idx, 'gap'], 
                results_df.loc[best_sep_idx, 'separability'],
                s=200, facecolors='none', edgecolors='green', linewidths=3,
                label=f"Best: {results_df.loc[best_sep_idx, 'separability']:.1%}")
    ax2.legend()
    
    # ========== Plot 3: Gap vs Weighted Background ==========
    ax3 = axes[1, 0]
    
    # Stacked bar: photons and 10*neutrons
    photons = results_df['photons_calo_entrance'].values
    neutrons_weighted = 10 * results_df['neutrons_calo_entrance'].values
    gaps = results_df['gap'].values
    
    ax3.bar(gaps, photons, label='Photons', color='steelblue', alpha=0.8)
    ax3.bar(gaps, neutrons_weighted, bottom=photons, label='10×Neutrons', color='coral', alpha=0.8)
    ax3.set_xlabel('Gap Distance (cm)', fontsize=11)
    ax3.set_ylabel('Weighted Background', fontsize=11)
    ax3.set_title('Background Composition at Calorimeter Entrance', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Mark minimum background
    min_bg_idx = results_df['weighted_background'].idxmin()
    ax3.axvline(results_df.loc[min_bg_idx, 'gap'], color='green', linestyle='--', 
                linewidth=2, label=f"Min @ gap={results_df.loc[min_bg_idx, 'gap']:.0f}")
    
    # ========== Plot 4: ALP Events (reference) ==========
    ax4 = axes[1, 1]
    ax4.semilogy(results_df['gap'], results_df['alp_events'], 'g-^', markersize=10, linewidth=2)
    ax4.set_xlabel('Gap Distance (cm)', fontsize=11)
    ax4.set_ylabel('ALP Events (log scale)', fontsize=11)
    ax4.set_title(f'ALP Signal (g = {args.coupling:.0e} GeV⁻¹, {exposure_hours:.1f} hr)', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Add secondary y-axis for opening angle
    ax4b = ax4.twinx()
    ax4b.plot(results_df['gap'], results_df['mean_opening_angle'], 'r--s', 
              markersize=6, linewidth=1.5, alpha=0.7)
    ax4b.set_ylabel('Mean Opening Angle (deg)', fontsize=10, color='red')
    ax4b.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    
    plot_file = output_dir / 'gap_pareto_front.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Pareto plot saved to: {plot_file}")
    
    plt.close()
    
    # ========== Generate summary statistics ==========
    print("\n" + "="*70)
    print("GAP PARETO OPTIMIZATION SUMMARY")
    print("="*70)
    
    # Summary table
    summary_cols = ['gap', 'separability', 'weighted_background', 'alp_events', 
                    'photons_calo_entrance', 'neutrons_calo_entrance']
    available_cols = [c for c in summary_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False, float_format='%.4f'))
    
    # Find optimal configurations
    if len(results_df) > 0:
        # Best separability
        best_sep = results_df.loc[results_df['separability'].idxmax()]
        
        # Minimum background
        min_bg = results_df.loc[results_df['weighted_background'].idxmin()]
        
        # Pareto optimal points
        pareto_mask = identify_pareto_front(
            results_df['separability'].values,
            results_df['weighted_background'].values
        )
        pareto_gaps = results_df.loc[pareto_mask, 'gap'].values
        
        print(f"\n{'='*70}")
        print("OPTIMAL CONFIGURATIONS")
        print(f"{'='*70}")
        print(f"Best separability:   gap = {best_sep['gap']:.0f} cm, "
              f"sep = {best_sep['separability']:.1%}, "
              f"bg = {best_sep['weighted_background']:.0f}")
        print(f"Min background:      gap = {min_bg['gap']:.0f} cm, "
              f"sep = {min_bg['separability']:.1%}, "
              f"bg = {min_bg['weighted_background']:.0f}")
        print(f"Pareto-optimal gaps: {', '.join(f'{g:.0f}' for g in sorted(pareto_gaps))} cm")
        print(f"{'='*70}")


def main():
    args = parse_args()
    results = run_gap_pareto_scan(args)
    return 0 if results is not None else 1


if __name__ == "__main__":
    exit(main())
