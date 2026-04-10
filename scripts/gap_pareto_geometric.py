#!/usr/bin/env python3
"""
Gap Pareto Optimization with Geometric Acceptance for DAMSA

This script refactors the gap optimization to:
1. Simulate ALP events ONCE (gap-independent)
2. For each gap configuration, calculate geometric acceptance

Key features:
- Sample ALP decay positions from exponential distribution (lifetime-based)
- Check if both decay photons hit the calorimeter face
- Apply separability criteria (angle >= 20° OR angle < 20° with both E >= 100 MeV)
- Calculate weighted background from simulation

Usage:
    python scripts/gap_pareto_geometric.py
    python scripts/gap_pareto_geometric.py --n-events 10000 --alp-mass 10 --coupling 1e-5
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.alplib_signal_plots import load_flux_for_alplib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GAP Pareto optimization with geometric acceptance"
    )
    parser.add_argument('--n-events', type=int, default=10000,
                        help='Number of simulation events per gap (default: 10000)')
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
                        help='ALP coupling in GeV^-1 (default: 1e-5)')
    parser.add_argument('--angle-cut-high', type=float, default=20.0,
                        help='High opening angle cut in degrees (default: 20)')
    parser.add_argument('--energy-cut', type=float, default=100.0,
                        help='Energy cut for separability in MeV (default: 100)')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of MC samples for ALP decay (default: 10000)')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, use existing data if available')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing results')
    return parser.parse_args()


# Geometry constants from construction.cpp
TARGET_EXIT_Z = -40.0  # cm (fixed target rear face position)
CALO_SIZE_XY = 12.0  # cm (12 cm x 12 cm calorimeter face)
CALO_HALF_WIDTH = CALO_SIZE_XY / 2.0  # 6 cm


def create_output_directory(base_dir: str = "output/gap_optimization") -> Path:
    """Create output directory for gap optimization results."""
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_simulation(gap_cm: float, n_events: int, target_z: float, target_xy: float,
                   optimize: bool = True) -> Dict[str, Any]:
    """Run Geant4 simulation for a given gap distance."""
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
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"ERROR: Simulation failed for gap = {gap_cm}")
        print(result.stderr)
        return None
    
    return {'gap': gap_cm, 'return_code': result.returncode}


def extract_simulation_results(gap_cm: float, output_dir: Path, 
                               target_z: float = 10.0, target_xy: float = 5.0) -> Optional[Dict[str, Any]]:
    """Extract results from simulation output files."""
    config_prefix = f"Tz{target_z:.0f}_xy{target_xy:.0f}_G{gap_cm:.0f}_"
    
    sim_info_file = output_dir / f"{config_prefix}simulation_info.json"
    
    results = {'gap': gap_cm}
    
    if sim_info_file.exists():
        with open(sim_info_file, 'r') as f:
            sim_info = json.load(f)
            results['photons_calo_entrance'] = sim_info.get('calo_entrance_photons', 0)
            results['neutrons_calo_entrance'] = sim_info.get('calo_entrance_neutrons', 0)
    else:
        print(f"WARNING: Simulation info file not found: {sim_info_file}")
        results['photons_calo_entrance'] = 0
        results['neutrons_calo_entrance'] = 0
    
    results['weighted_background'] = (
        results['photons_calo_entrance'] + 10 * results['neutrons_calo_entrance']
    )
    
    return results


def generate_alp_decay_events(
    photon_flux_file: Path,
    n_primaries: int,
    beam_current: float,
    exposure_days: float,
    coupling: float,
    alp_mass: float,
    n_samples: int = 10000,
    nominal_gap_cm: float = 47.0
) -> Dict[str, Any]:
    """
    Generate ALP decay events once (gap-independent).
    
    This function:
    1. Loads photon flux and scales to real beam
    2. Simulates ALP production via Primakoff
    3. Gets decay 4-vectors with weights
    4. Samples decay positions from lifetime
    5. Returns list of ALP decay events
    """
    coupling_mev = coupling / 1000.0
    
    print(f"\n{'='*60}")
    print("Generating ALP decay events (once, gap-independent)")
    print(f"{'='*60}")
    print(f"  ALP mass: m_a = {alp_mass:.2f} MeV")
    print(f"  Coupling: g = {coupling:.2e} GeV^-1 = {coupling_mev:.2e} MeV^-1")
    print(f"  Exposure: {exposure_days * 24:.1f} hours ({exposure_days:.4f} days)")
    print(f"  MC samples: {n_samples}")
    
    try:
        import alplib.fluxes as alplib_fluxes
        import alplib.materials as alplib_materials
        import alplib.generators as alplib_generators
        from alplib.decay import W_gg
        from alplib.constants import METER_BY_MEV, S_PER_DAY, C_LIGHT, HBAR
        
        # Load flux with beam scaling
        flux_array = load_flux_for_alplib(str(photon_flux_file), n_primaries, beam_current)
        
        # Check photon energy threshold
        max_photon_energy = flux_array[:, 0].max()
        if alp_mass >= max_photon_energy:
            print(f"ERROR: ALP mass {alp_mass} MeV >= max photon energy {max_photon_energy:.1f} MeV")
            print(f"       No ALP production possible!")
            return {'events': [], 'n_events': 0.0, 'max_photon_energy': max_photon_energy}
        
        # Create materials
        target_material = alplib_materials.Material("W")
        
        # Set geometry to middle of decay chamber range
        # This determines the decay probability calculation
        det_dist_m = nominal_gap_cm / 100.0  # 0.47 m
        det_length_m = nominal_gap_cm / 100.0  # 0.47 m (full decay chamber)
        det_area = 0.12 * 0.12  # 0.0144 m² (12 cm x 12 cm)
        
        print(f"  Geometry: detector distance = {det_dist_m*100:.1f} cm")
        print(f"           detector length = {det_length_m*100:.1f} cm")
        print(f"           detector area = {det_area*1e4:.1f} cm²")
        
        # Create Primakoff flux generator
        # Use the standard alplib approach but we'll add decay position sampling separately
        flux_obj = alplib_fluxes.FluxPrimakoffIsotropic(
            photon_flux=flux_array,
            target=target_material,
            det_dist=det_dist_m,
            det_length=det_length_m,
            det_area=det_area,
            axion_mass=alp_mass,
            axion_coupling=coupling_mev,
            n_samples=n_samples
        )
        
        # Simulate ALP production (Primakoff conversion)
        flux_obj.simulate()
        
        n_alp_bins = len(flux_obj.axion_energy)
        print(f"  ALP energy bins: {n_alp_bins}")
        
        if n_alp_bins == 0:
            print(f"  WARNING: No ALP production (m_a too large for photon spectrum)")
            return {'events': [], 'n_events': 0.0, 'max_photon_energy': max_photon_energy}
        
        alp_energy_min = np.array(flux_obj.axion_energy).min()
        alp_energy_max = np.array(flux_obj.axion_energy).max()
        print(f"  ALP energy range: {alp_energy_min:.1f} - {alp_energy_max:.1f} MeV")
        
        # Calculate decay width and lifetime
        decay_width = W_gg(coupling_mev, alp_mass)
        print(f"  Decay width: Γ = {decay_width:.3e} MeV")
        tau_rest = HBAR / decay_width
        print(f"  Rest lifetime: τ = {tau_rest:.3e} s")
        
        # Calculate total ALP production flux
        total_alp_flux = np.sum(flux_obj.axion_flux)
        print(f"  Total ALP production flux: {total_alp_flux:.3e} ALPs/s")
        
        # Calculate decay weights manually with correct units
        # For each ALP energy bin:
        # - decay_prob = 1 - exp(-gap / decay_length)
        # - decay_weight = flux * decay_prob * exposure
        axion_energies = np.array(flux_obj.axion_energy)
        axion_flux = np.array(flux_obj.axion_flux)
        
        # Use gap = 47 cm (middle of range) for nominal decay calculation
        # This is just for weighting; we'll calculate acceptance per-gap
        gap_for_weight = nominal_gap_cm  # cm
        
        # For each ALP energy, calculate decay probability in the gap
        manual_decay_weights = []
        for E in axion_energies:
            if E <= alp_mass:
                manual_decay_weights.append(0.0)
            else:
                gamma = E / alp_mass
                beta = np.sqrt(1 - (alp_mass/E)**2)
                decay_length = gamma * beta * C_LIGHT * tau_rest  # cm
                decay_prob = 1 - np.exp(-gap_for_weight / decay_length)
                # Weight per second = flux * decay_prob
                # For exposure period = flux * decay_prob * seconds
                manual_decay_weights.append(decay_prob)
        
        manual_decay_weights = np.array(manual_decay_weights, dtype=np.float32)
        
        # Assign manually calculated decay weights to flux object
        flux_obj.decay_axion_weight = manual_decay_weights
        
        print(f"  Manual decay weights (for gap={gap_for_weight}cm): {manual_decay_weights.sum():.4f}")
        
        # Create event generator
        detector_material = alplib_materials.Material("CsI")
        generator = alplib_generators.PhotonEventGenerator(flux_obj, detector_material)
        
        # Get total events (should now work)
        try:
            n_events = generator.decays(days_exposure=exposure_days, threshold=0.1)
            print(f"  Total ALP decay events (exposure): {n_events:.4f}")
        except Exception as e:
            print(f"  Warning: Could not calculate n_events: {e}")
            n_events = 0.0
        
        # Get decay 4-vectors - these include the exposure-weighted decay probability
        print(f"  Generating decay 4-vectors with {n_samples} samples...")
        p4_1, p4_2, weights = generator.simulate_decay_4vectors(
            days_exposure=exposure_days,
            n_samples=n_samples
        )
        
        n_samples_returned = len(p4_1)
        print(f"  Returned {n_samples_returned} decay samples")
        
        if n_samples_returned == 0:
            print(f"  WARNING: No decay samples generated")
            return {'events': [], 'n_events': 0.0, 'decay_width': decay_width,
                    'tau_rest': tau_rest, 'max_photon_energy': max_photon_energy}
        
        # Get total events for this exposure (scaled by decay probability from propagate)
        # First need to call propagate to get the decay weights
        flux_obj.propagate(decay_width)
        n_events = generator.decays(days_exposure=exposure_days, threshold=0.1)
        print(f"  Total ALP decay events (exposure): {n_events:.4f}")
        
        # Build event list with decay kinematics (no position yet - we'll sample per gap)
        print(f"  Building event list...")
        
        events = []
        axion_energies = np.array(flux_obj.axion_energy)
        n_flux_bins = len(axion_energies)
        
        for i in range(n_flux_bins):
            alp_E = axion_energies[i]
            if alp_E <= alp_mass:
                continue
            
            # Lorentz factor for this ALP energy
            gamma = alp_E / alp_mass
            beta = np.sqrt(1 - (alp_mass / alp_E)**2)
            
            # Lab-frame decay length
            lab_decay_length = gamma * beta * C_LIGHT * tau_rest  # cm
            lab_decay_length_m = lab_decay_length / 100  # meters
            
            # Number of samples per flux bin
            n_per_bin = n_samples  # from simulate_decay_4vectors
            
            for j in range(n_per_bin):
                idx = i * n_per_bin + j
                if idx >= n_samples_returned:
                    break
                if weights[idx] <= 0:
                    continue
                
                # Extract 4-vector info
                p1 = p4_1[idx]
                p2 = p4_2[idx]
                
                # Energies
                E1 = p1.p0
                E2 = p2.p0
                
                # Skip if photon energies are zero or negative
                if E1 <= 0 or E2 <= 0:
                    continue
                
                # Momenta for opening angle
                px1, py1, pz1 = p1.p1, p1.p2, p1.p3
                px2, py2, pz2 = p2.p1, p2.p2, p2.p3
                
                p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)
                p2_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
                
                if p1_mag < 1e-10 or p2_mag < 1e-10:
                    continue
                
                # Opening angle approximation: theta = m_a / E_alp (in radians)
                # This is the theoretical formula for highly relativistic ALP decay a -> gamma gamma
                if alp_E > 0:
                    theta_rad = alp_mass / alp_E
                    opening_angle_deg = np.degrees(theta_rad)
                else:
                    opening_angle_deg = 0.0
                
                # Store event without position - position will be sampled per gap
                events.append({
                    'weight': weights[idx],
                    'alp_energy_MeV': alp_E,
                    'photon1_energy_MeV': E1,
                    'photon2_energy_MeV': E2,
                    'photon1_px': px1, 'photon1_py': py1, 'photon1_pz': pz1,
                    'photon2_px': px2, 'photon2_py': py2, 'photon2_pz': pz2,
                    'photon1_p_mag': p1_mag,
                    'photon2_p_mag': p2_mag,
                    'opening_angle_deg': opening_angle_deg,
                    'lab_decay_length_cm': lab_decay_length,
                    'lab_decay_length_m': lab_decay_length_m,
                    'gamma': gamma,
                    'beta': beta
                })
        
        print(f"  Built {len(events)} decay events")
        
        # Summary statistics
        if len(events) > 0:
            theta_values = [e['opening_angle_deg'] for e in events]
            print(f"  Opening angles: mean={np.mean(theta_values):.1f}°, "
                  f"min={np.min(theta_values):.1f}°, max={np.max(theta_values):.1f}°")
            
            E1_values = [e['photon1_energy_MeV'] for e in events]
            E2_values = [e['photon2_energy_MeV'] for e in events]
            print(f"  Photon energies: E1 mean={np.mean(E1_values):.1f} MeV, "
                  f"E2 mean={np.mean(E2_values):.1f} MeV")
        
        return {
            'events': events,
            'n_events': n_events,
            'decay_width': decay_width,
            'tau_rest': tau_rest,
            'max_photon_energy': max_photon_energy,
            'alp_energy_min': alp_energy_min,
            'alp_energy_max': alp_energy_max,
            'total_alp_flux': total_alp_flux,
            'exposure_days': exposure_days
        }
        
    except ImportError as e:
        print(f"ERROR: alplib not available: {e}")
        return {'events': [], 'n_events': 0.0, 'error': str(e)}
    except Exception as e:
        print(f"ERROR: ALP generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'events': [], 'n_events': 0.0, 'error': str(e)}


def calculate_geometric_acceptance(
    events: List[Dict],
    gap_cm: float,
    target_exit_z: float = TARGET_EXIT_Z,
    calo_half_width: float = CALO_HALF_WIDTH,
    angle_cut_high: float = 10.0,
    energy_cut: float = 100.0
) -> Dict[str, Any]:
    """
    Calculate geometric acceptance for a given gap configuration.
    
    For each ALP decay event:
    1. Sample decay position from exponential distribution (based on lifetime)
    2. Check if decay happens within decay chamber (z < calo entrance)
    3. Extrapolate photon positions to calorimeter face
    4. Check if both photons hit calorimeter
    5. Apply separability criteria
    
    Returns dict with acceptance results.
    """
    calo_entrance_z = target_exit_z + gap_cm  # absolute z position
    
    separable_events = []
    accepted_events = []
    total_weight = 0.0
    
    for event in events:
        total_weight += event['weight']
        
        # Sample decay position from exponential distribution
        # P(z) ∝ exp(-z / decay_length)
        lab_decay_length = event['lab_decay_length_cm']  # cm
        
        # Sample z from exponential distribution
        u = np.random.random()
        z_decay_cm = -lab_decay_length * np.log(1 - u)  # cm from target exit
        
        # Skip if decay happens outside decay chamber (after calo entrance)
        if z_decay_cm > gap_cm:
            continue
        
        # Distance from decay to calorimeter entrance
        distance_to_calo = gap_cm - z_decay_cm  # cm
        
        # Get photon energies
        E1 = event['photon1_energy_MeV']
        E2 = event['photon2_energy_MeV']
        
        # Opening angle
        theta = event['opening_angle_deg']
        
        # Calculate photon positions at calorimeter entrance
        # Assume decay at beam axis (x=0, y=0)
        # Photons emitted with opening angle theta symmetrically around beam
        # For small angles: separation ≈ theta * distance (theta in radians)
        
        theta_rad = np.radians(theta)
        separation_at_calo = theta_rad * distance_to_calo  # cm
        
        # Check if both photons hit calorimeter
        # Assume beam-centered: each photon hits at ± separation/2
        half_separation = separation_at_calo / 2.0
        
        photon1_hits = half_separation <= calo_half_width
        photon2_hits = half_separation <= calo_half_width
        both_hit = photon1_hits and photon2_hits
        
        if both_hit:
            accepted_events.append(event)
            
            # Apply separability criteria
            if theta >= angle_cut_high:
                separable_events.append(event)
            elif E1 >= energy_cut and E2 >= energy_cut:
                separable_events.append(event)
    
    # Calculate metrics
    accepted_weight = sum(e['weight'] for e in accepted_events)
    separable_weight = sum(e['weight'] for e in separable_events)
    
    geometric_acceptance = accepted_weight / total_weight if total_weight > 0 else 0
    separability_of_accepted = separable_weight / accepted_weight if accepted_weight > 0 else 0
    separable_fraction = separable_weight / total_weight if total_weight > 0 else 0
    
    return {
        'gap': gap_cm,
        'calorimeter_z': calo_entrance_z,
        'total_events': len(events),
        'accepted_events': len(accepted_events),
        'separable_events': len(separable_events),
        'total_weight': total_weight,
        'accepted_weight': accepted_weight,
        'separable_weight': separable_weight,
        'geometric_acceptance': geometric_acceptance,
        'separability_of_accepted': separability_of_accepted,
        'separable_fraction': separable_fraction,
        'sep_efficiency': separability_of_accepted,
    }


def identify_pareto_front(separability: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Identify Pareto-optimal points."""
    n = len(separability)
    pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if (separability[j] >= separability[i] and 
                    background[j] <= background[i] and
                    (separability[j] > separability[i] or background[j] < background[i])):
                    pareto[i] = False
                    break
    
    return pareto


def run_gap_pareto_geometric(args):
    """Main function to run gap Pareto optimization with geometric acceptance."""
    
    output_dir = create_output_directory()
    
    # Gap values: 42 to 52 cm (1 cm steps)
    gap_values = list(range(42, 53))
    
    # Exposure in hours for display
    exposure_hours = args.exposure * 24
    
    print(f"\n{'#'*60}")
    print(f"# GAP PARETO OPTIMIZATION WITH GEOMETRIC ACCEPTANCE")
    print(f"#")
    print(f"# Target: {args.target_z} cm Z x {args.target_xy} cm XY")
    print(f"# Gap range: {gap_values[0]} to {gap_values[-1]} cm ({len(gap_values)} values)")
    print(f"# Events per config: {args.n_events}")
    print(f"# ALP mass: {args.alp_mass} MeV")
    print(f"# Coupling: {args.coupling:.2e} GeV^-1")
    print(f"# Exposure: {exposure_hours:.1f} hours ({args.exposure:.4f} days)")
    print(f"#")
    print(f"# MC samples: {args.n_samples}")
    print(f"#")
    print(f"# Geometry:")
    print(f"#   Target exit Z: {TARGET_EXIT_Z} cm")
    print(f"#   Calorimeter size: {CALO_SIZE_XY} cm x {CALO_SIZE_XY} cm")
    print(f"#")
    print(f"# Separability criteria:")
    print(f"#   - Opening angle >= {args.angle_cut_high} deg, OR")
    print(f"#   - Opening angle < {args.angle_cut_high} deg AND both photons >= {args.energy_cut} MeV")
    print(f"#")
    print(f"# Weighted background: photons + 10*neutrons @ calo entrance")
    print(f"#")
    print(f"# Output directory: {output_dir}")
    print(f"{'#'*60}")
    
    # Step 1: Generate ALP events once (gap-independent)
    config_prefix = f"Tz{args.target_z:.0f}_xy{args.target_xy:.0f}_G42_"
    photon_flux_file = Path('output') / f"{config_prefix}photon_flux_target_exit.csv"
    
    if not photon_flux_file.exists():
        # Try to find any flux file
        flux_files = list(Path('output').glob('*photon_flux_target_exit.csv'))
        if flux_files:
            photon_flux_file = flux_files[0]
            print(f"\nUsing flux file: {photon_flux_file}")
        else:
            print(f"ERROR: No photon flux file found in output/")
            return None
    
    alp_result = generate_alp_decay_events(
        photon_flux_file=photon_flux_file,
        n_primaries=args.n_events,
        beam_current=args.beam_current,
        exposure_days=args.exposure,
        coupling=args.coupling,
        alp_mass=args.alp_mass,
        n_samples=args.n_samples,
        nominal_gap_cm=47.0  # middle of range
    )
    
    events = alp_result.get('events', [])
    n_alp_events = alp_result.get('n_events', 0)
    
    print(f"\n{'='*60}")
    print(f"ALP Event Generation Summary")
    print(f"{'='*60}")
    print(f"  Total ALP decay events (exposure): {n_alp_events:.4f}")
    print(f"  Events with decay positions: {len(events)}")
    
    if len(events) == 0:
        print(f"\nERROR: No ALP events generated!")
        return None
    
    # Step 2: Loop over gaps with geometric acceptance
    results = []
    
    for gap in gap_values:
        print(f"\n{'='*60}")
        print(f"Gap configuration: {gap} cm")
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
        
        # Extract simulation results (background)
        sim_results = extract_simulation_results(
            gap, Path('output'), args.target_z, args.target_xy
        )
        
        # Calculate geometric acceptance
        geo_result = calculate_geometric_acceptance(
            events=events,
            gap_cm=gap,
            target_exit_z=TARGET_EXIT_Z,
            calo_half_width=CALO_HALF_WIDTH,
            angle_cut_high=args.angle_cut_high,
            energy_cut=args.energy_cut
        )
        
        # Combine results
        combined = {
            'gap': gap,
            'weighted_background': sim_results['weighted_background'],
            'separable_weight': geo_result['separable_weight'],
            'accepted_weight': geo_result['accepted_weight'],
            'total_weight': geo_result['total_weight'],
            'geometric_acceptance': geo_result['geometric_acceptance'],
            'separability_of_accepted': geo_result['separability_of_accepted'],
            'separable_fraction': geo_result['separable_fraction'],
            'sep_efficiency': geo_result['sep_efficiency'],
            'photons_calo_entrance': sim_results['photons_calo_entrance'],
            'neutrons_calo_entrance': sim_results['neutrons_calo_entrance']
        }
        
        results.append(combined)
        
        print(f"  Results:")
        print(f"    Weighted background: {sim_results['weighted_background']}")
        print(f"    ALP separable weight: {geo_result['separable_weight']:.4f}")
        print(f"    Geometric acceptance: {geo_result['geometric_acceptance']*100:.1f}%")
        print(f"    Separability (of accepted): {geo_result['separability_of_accepted']*100:.1f}%")
    
    if len(results) == 0:
        print("ERROR: No results collected")
        return None
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_file = output_dir / 'gap_pareto_geometric_results.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}")
    
    # Generate Pareto visualization
    generate_pareto_plot(results_df, output_dir, args, n_alp_events)
    
    print(f"\n{'#'*60}")
    print(f"# SCAN COMPLETE")
    print(f"#")
    print(f"# Results: {csv_file}")
    print(f"# Plots: {output_dir / 'gap_pareto_geometric.png'}")
    print(f"{'#'*60}")
    
    return results_df


def generate_pareto_plot(results_df: pd.DataFrame, output_dir: Path, args, n_alp_events: float):
    """Generate Pareto front visualization."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot generation")
        return
    
    exposure_hours = args.exposure * 24
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'DAMSA Gap Pareto (Geometric Acceptance)\n'
                 f'(Target: {args.target_z}×{args.target_xy} cm, '
                 f'm_a = {args.alp_mass:.1f} MeV, '
                 f'g = {args.coupling:.0e} GeV⁻¹, '
                 f'{exposure_hours:.1f} hr)', 
                 fontsize=12, fontweight='bold')
    
    # ========== Plot 1: Pareto Front ==========
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(
        results_df['weighted_background'],
        results_df['separable_weight'],
        c=results_df['gap'], cmap='viridis', s=150, edgecolors='black'
    )
    ax1.set_xlabel('Weighted Background (photons + 10×neutrons)', fontsize=11)
    ax1.set_ylabel('Separable ALP Signal Weight', fontsize=11)
    ax1.set_title('Pareto Front: Signal vs Background', fontsize=11)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Gap (cm)', fontsize=10)
    
    # Highlight Pareto-optimal points
    if len(results_df) > 1:
        pareto_mask = identify_pareto_front(
            results_df['separable_weight'].values,
            results_df['weighted_background'].values
        )
        pareto_df = results_df[pareto_mask]
        ax1.scatter(
            pareto_df['weighted_background'],
            pareto_df['separable_weight'],
            s=300, facecolors='none', edgecolors='red', linewidths=3,
            label=f'Pareto optimal ({pareto_mask.sum()} points)'
        )
        for _, row in pareto_df.iterrows():
            ax1.annotate(f"{row['gap']:.0f}", 
                        (row['weighted_background'], row['separable_weight']),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=9, color='red')
        ax1.legend(loc='lower right')
    
    # ========== Plot 2: Gap vs Separable Weight ==========
    ax2 = axes[0, 1]
    ax2.plot(results_df['gap'], results_df['separable_weight'], 'b-o', markersize=8, linewidth=2)
    ax2.set_xlabel('Gap Distance (cm)', fontsize=11)
    ax2.set_ylabel('Separable ALP Signal Weight', fontsize=11)
    ax2.set_title('Signal vs Gap Distance', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ========== Plot 3: Background Composition ==========
    ax3 = axes[1, 0]
    photons = results_df['photons_calo_entrance'].values
    neutrons_weighted = 10 * results_df['neutrons_calo_entrance'].values
    gaps = results_df['gap'].values
    
    ax3.bar(gaps, photons, label='Photons', color='steelblue', alpha=0.8)
    ax3.bar(gaps, neutrons_weighted, bottom=photons, label='10×Neutrons', color='coral', alpha=0.8)
    ax3.set_xlabel('Gap Distance (cm)', fontsize=11)
    ax3.set_ylabel('Weighted Background', fontsize=11)
    ax3.set_title('Background Composition', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 4: Acceptance Metrics ==========
    ax4 = axes[1, 1]
    ax4.plot(results_df['gap'], results_df['geometric_acceptance']*100, 'g-^', 
             markersize=8, linewidth=2, label='Geometric Acceptance')
    ax4.plot(results_df['gap'], results_df['separability_of_accepted']*100, 'r-s', 
             markersize=8, linewidth=2, label='Separability (of accepted)')
    ax4.set_xlabel('Gap Distance (cm)', fontsize=11)
    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_title('Acceptance Metrics vs Gap', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'gap_pareto_geometric.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Pareto plot saved to: {plot_file}")
    
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("GAP PARETO OPTIMIZATION SUMMARY (GEOMETRIC)")
    print("="*70)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    if len(results_df) > 0:
        best_signal = results_df.loc[results_df['separable_weight'].idxmax()]
        min_bg = results_df.loc[results_df['weighted_background'].idxmin()]
        
        pareto_mask = identify_pareto_front(
            results_df['separable_weight'].values,
            results_df['weighted_background'].values
        )
        pareto_gaps = results_df.loc[pareto_mask, 'gap'].values
        
        print(f"\n{'='*70}")
        print("OPTIMAL CONFIGURATIONS")
        print(f"{'='*70}")
        print(f"Max signal:          gap = {best_signal['gap']:.0f} cm, "
              f"signal = {best_signal['separable_weight']:.4f}, "
              f"bg = {best_signal['weighted_background']:.0f}")
        print(f"Min background:       gap = {min_bg['gap']:.0f} cm, "
              f"signal = {min_bg['separable_weight']:.4f}, "
              f"bg = {min_bg['weighted_background']:.0f}")
        print(f"Pareto-optimal gaps: {', '.join(f'{g:.0f}' for g in sorted(pareto_gaps))} cm")


def main():
    args = parse_args()
    results = run_gap_pareto_geometric(args)
    return 0 if results is not None else 1


if __name__ == "__main__":
    exit(main())
