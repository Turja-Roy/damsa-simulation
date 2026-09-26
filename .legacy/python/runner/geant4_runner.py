#!/usr/bin/env python3
"""
Geant4 Runner Module for DAMSA Optimization

This module provides a Python interface to run Geant4 simulations
with different geometry parameters and collect results.
"""

import subprocess
import tempfile
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import hashlib
import json
import time


@dataclass
class SimulationResult:
    """Container for simulation results."""
    # Photon data at target exit
    photon_energies: np.ndarray      # MeV
    photon_times: np.ndarray          # ns
    photon_angles: np.ndarray         # radians from beam axis
    photon_positions: np.ndarray      # (N, 3) in mm
    photon_momenta: np.ndarray        # (N, 3) unit vectors
    
    # Background data
    neutron_energies: np.ndarray      # MeV
    neutron_times: np.ndarray         # ns
    electron_energies: np.ndarray     # MeV (e+ and e-)
    electron_times: np.ndarray        # ns
    
    # Metadata
    n_primaries: int
    target_z_cm: float
    target_x_cm: float
    target_y_cm: float
    gap_cm: float
    run_time_seconds: float
    
    @property
    def n_photons(self) -> int:
        return len(self.photon_energies)
    
    @property
    def n_neutrons(self) -> int:
        return len(self.neutron_energies)
    
    @property
    def n_electrons(self) -> int:
        return len(self.electron_energies)
    
    def photons_in_timing_window(self, t_min: float, t_max: float) -> int:
        """Count photons within timing window [t_min, t_max] ns."""
        mask = (self.photon_times >= t_min) & (self.photon_times <= t_max)
        return np.sum(mask)
    
    def neutrons_in_timing_window(self, t_min: float, t_max: float) -> int:
        """Count neutrons within timing window [t_min, t_max] ns."""
        mask = (self.neutron_times >= t_min) & (self.neutron_times <= t_max)
        return np.sum(mask)
    
    def photons_in_angle_range(self, theta_max_deg: float) -> int:
        """Count photons within angle theta_max from beam axis."""
        theta_max_rad = np.radians(theta_max_deg)
        return np.sum(self.photon_angles <= theta_max_rad)
    
    def get_photon_flux_array(self, bin_width_MeV: float = 1.0) -> np.ndarray:
        """Get binned photon flux for alplib input."""
        if len(self.photon_energies) == 0:
            return np.array([[1.0, 0.0]])
        
        max_E = np.ceil(self.photon_energies.max())
        bins = np.arange(0, max_E + bin_width_MeV, bin_width_MeV)
        counts, edges = np.histogram(self.photon_energies, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        
        flux = np.column_stack([centers, counts])
        return flux[flux[:, 1] > 0]


class Geant4Runner:
    """
    Interface for running DAMSA Geant4 simulations with variable parameters.
    
    Parameters
    ----------
    executable : str
        Path to the damsa executable
    base_dir : str
        Base directory for temporary files
    n_events : int
        Number of primary events per simulation
    cache_dir : str, optional
        Directory to cache simulation results
    verbose : bool
        Print progress information
    """
    
    def __init__(self, 
                 executable: str = "./build/damsa_opt",
                 base_dir: str = ".",
                 n_events: int = 1000,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        self.executable = Path(executable).resolve()
        self.base_dir = Path(base_dir).resolve()
        self.n_events = n_events
        self.verbose = verbose
        
        # Setup cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / "optimization_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Verify executable exists
        if not self.executable.exists():
            raise FileNotFoundError(f"Executable not found: {self.executable}")
    
    def _get_cache_key(self, target_z: float, target_x: float, target_y: float, 
                       gap: float, n_events: int) -> str:
        """Generate unique cache key for parameter combination."""
        params = {
            'target_z': round(target_z, 2),
            'target_x': round(target_x, 2),
            'target_y': round(target_y, 2),
            'gap': round(gap, 2),
            'n_events': n_events
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    def _check_cache(self, cache_key: str) -> Optional[SimulationResult]:
        """Check if result exists in cache."""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                return SimulationResult(
                    photon_energies=data['photon_energies'],
                    photon_times=data['photon_times'],
                    photon_angles=data['photon_angles'],
                    photon_positions=data['photon_positions'],
                    photon_momenta=data['photon_momenta'],
                    neutron_energies=data['neutron_energies'],
                    neutron_times=data['neutron_times'],
                    electron_energies=data['electron_energies'],
                    electron_times=data['electron_times'],
                    n_primaries=int(data['n_primaries']),
                    target_z_cm=float(data['target_z_cm']),
                    target_x_cm=float(data['target_x_cm']),
                    target_y_cm=float(data['target_y_cm']),
                    gap_cm=float(data['gap_cm']),
                    run_time_seconds=float(data['run_time_seconds'])
                )
            except Exception as e:
                if self.verbose:
                    print(f"Cache read failed: {e}")
        return None
    
    def _save_cache(self, cache_key: str, result: SimulationResult):
        """Save result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        np.savez(cache_file,
                 photon_energies=result.photon_energies,
                 photon_times=result.photon_times,
                 photon_angles=result.photon_angles,
                 photon_positions=result.photon_positions,
                 photon_momenta=result.photon_momenta,
                 neutron_energies=result.neutron_energies,
                 neutron_times=result.neutron_times,
                 electron_energies=result.electron_energies,
                 electron_times=result.electron_times,
                 n_primaries=result.n_primaries,
                 target_z_cm=result.target_z_cm,
                 target_x_cm=result.target_x_cm,
                 target_y_cm=result.target_y_cm,
                 gap_cm=result.gap_cm,
                 run_time_seconds=result.run_time_seconds)
    
    def _create_macro(self, workdir: Path, n_events: int) -> Path:
        """Create Geant4 macro file for batch run."""
        macro_path = workdir / "run_opt.mac"
        macro_content = f"""# Auto-generated macro for optimization
/run/initialize
/run/beamOn {n_events}
"""
        macro_path.write_text(macro_content)
        return macro_path
    
    def _create_geometry_config(self, workdir: Path, target_z: float, 
                                 target_x: float, target_y: float, gap: float) -> Path:
        """
        Create geometry configuration file.
        
        Note: This requires the Geant4 code to read geometry from a config file
        or we pass parameters via macro commands.
        """
        config_path = workdir / "geometry.conf"
        config_content = f"""# Geometry configuration
TARGET_Z_CM {target_z}
TARGET_X_CM {target_x}
TARGET_Y_CM {target_y}
GAP_CM {gap}
"""
        config_path.write_text(config_content)
        return config_path
    
    def run(self, target_z: float, target_x: float, target_y: float,
            gap: float, n_events: Optional[int] = None,
            use_cache: bool = True) -> SimulationResult:
        """Run Geant4 simulation with specified geometry parameters."""
        if n_events is None:
            n_events = self.n_events
        
        # Check cache first
        cache_key = self._get_cache_key(target_z, target_x, target_y, gap, n_events)
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached is not None:
                if self.verbose:
                    print(f"Using cached result for key {cache_key}")
                return cached
        
        # Create working directory
        workdir = Path(tempfile.mkdtemp(prefix="damsa_opt_"))
        
        try:
            start_time = time.time()
            
            # Create output directory in working directory
            output_dir = workdir / "output"
            output_dir.mkdir()
            
            if self.verbose:
                print(f"Running simulation: target=({target_x}x{target_y}x{target_z}) cm, gap={gap} cm")
            
            # Execute damsa_opt with command-line arguments
            cmd = [
                str(self.executable),
                "--target-z", str(target_z),
                "--target-xy", str(target_x),  # Square target: x=y
                "--gap", str(gap),
                "--n-events", str(n_events),
                "--output-dir", str(output_dir),
            ]
            if not self.verbose:
                cmd.append("--quiet")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            run_time = time.time() - start_time
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"Simulation failed: {result.stderr[:500]}")
                # Return empty result
                return self._empty_result(target_z, target_x, target_y, gap, n_events, run_time)
            
            # Parse output files (output is in workdir/output, but FluxData.h writes to ./output)
            # Check both locations
            if (output_dir / "photon_flux_target_exit.csv").exists():
                actual_output = output_dir
            elif (self.base_dir / "output" / "photon_flux_target_exit.csv").exists():
                actual_output = self.base_dir / "output"
            else:
                actual_output = output_dir
            
            sim_result = self._parse_output(actual_output, target_z, target_x, target_y, 
                                           gap, n_events, run_time)
            
            # Cache result
            if use_cache:
                self._save_cache(cache_key, sim_result)
            
            return sim_result
            
        finally:
            # Cleanup
            shutil.rmtree(workdir, ignore_errors=True)
    
    def _parse_output(self, output_dir: Path, target_z: float, target_x: float,
                      target_y: float, gap: float, n_events: int,
                      run_time: float) -> SimulationResult:
        """Parse simulation output files."""
        
        # Read photon flux CSV
        photon_file = output_dir / "photon_flux_target_exit.csv"
        if photon_file.exists():
            df_photons = pd.read_csv(photon_file)
            photon_energies = df_photons['energy_MeV'].values
            photon_times = df_photons['time_ns'].values
            
            # Calculate angles from momentum
            px = df_photons['px'].values
            py = df_photons['py'].values
            pz = df_photons['pz'].values
            photon_angles = np.arccos(np.clip(pz, -1, 1))
            
            photon_positions = np.column_stack([
                df_photons['x_mm'].values,
                df_photons['y_mm'].values,
                df_photons['z_mm'].values
            ])
            photon_momenta = np.column_stack([px, py, pz])
        else:
            photon_energies = np.array([])
            photon_times = np.array([])
            photon_angles = np.array([])
            photon_positions = np.zeros((0, 3))
            photon_momenta = np.zeros((0, 3))
        
        # Read background CSV
        bkg_file = output_dir / "background_target_exit.csv"
        if bkg_file.exists():
            df_bkg = pd.read_csv(bkg_file)
            
            # Neutrons (PDG 2112)
            neutrons = df_bkg[df_bkg['pdg'] == 2112]
            neutron_energies = neutrons['energy_MeV'].values if len(neutrons) > 0 else np.array([])
            neutron_times = neutrons['time_ns'].values if len(neutrons) > 0 else np.array([])
            
            # Electrons and positrons (PDG 11, -11)
            electrons = df_bkg[df_bkg['pdg'].isin([11, -11])]
            electron_energies = electrons['energy_MeV'].values if len(electrons) > 0 else np.array([])
            electron_times = electrons['time_ns'].values if len(electrons) > 0 else np.array([])
        else:
            neutron_energies = np.array([])
            neutron_times = np.array([])
            electron_energies = np.array([])
            electron_times = np.array([])
        
        return SimulationResult(
            photon_energies=photon_energies,
            photon_times=photon_times,
            photon_angles=photon_angles,
            photon_positions=photon_positions,
            photon_momenta=photon_momenta,
            neutron_energies=neutron_energies,
            neutron_times=neutron_times,
            electron_energies=electron_energies,
            electron_times=electron_times,
            n_primaries=n_events,
            target_z_cm=target_z,
            target_x_cm=target_x,
            target_y_cm=target_y,
            gap_cm=gap,
            run_time_seconds=run_time
        )
    
    def _empty_result(self, target_z: float, target_x: float, target_y: float,
                      gap: float, n_events: int, run_time: float) -> SimulationResult:
        """Return empty result for failed simulation."""
        return SimulationResult(
            photon_energies=np.array([]),
            photon_times=np.array([]),
            photon_angles=np.array([]),
            photon_positions=np.zeros((0, 3)),
            photon_momenta=np.zeros((0, 3)),
            neutron_energies=np.array([]),
            neutron_times=np.array([]),
            electron_energies=np.array([]),
            electron_times=np.array([]),
            n_primaries=n_events,
            target_z_cm=target_z,
            target_x_cm=target_x,
            target_y_cm=target_y,
            gap_cm=gap,
            run_time_seconds=run_time
        )
    
    def run_batch(self, parameter_sets: np.ndarray, 
                  n_workers: int = 1) -> list:
        """Run multiple simulations (optionally in parallel)."""
        results = []
        
        if n_workers == 1:
            for i, params in enumerate(parameter_sets):
                if self.verbose:
                    print(f"Running simulation {i+1}/{len(parameter_sets)}")
                result = self.run(params[0], params[1], params[2], params[3])
                results.append(result)
        else:
            # Parallel execution
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.run, p[0], p[1], p[2], p[3]): i
                    for i, p in enumerate(parameter_sets)
                }
                
                results = [None] * len(parameter_sets)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
        
        return results


class MockGeant4Runner:
    """
    Mock runner for testing optimization without actual Geant4.
    
    Uses simple analytical models to approximate physics behavior.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.n_events = 1000
    
    def run(self, target_z: float, target_x: float, target_y: float,
            gap: float, n_events: Optional[int] = None,
            use_cache: bool = True) -> SimulationResult:
        """
        Generate mock simulation results based on simple physics models.
        """
        if n_events is None:
            n_events = self.n_events
        
        # Simple physics model
        # Photon production scales with target thickness (radiation lengths)
        X0_tungsten = 0.35  # cm
        n_X0 = target_z / X0_tungsten
        
        # Photon multiplicity increases then saturates
        photon_mult = 20 * n_X0 * (1 - np.exp(-n_X0/10))
        n_photons = int(n_events * photon_mult * np.random.uniform(0.9, 1.1))
        
        # Neutron production increases with target thickness
        neutron_mult = 0.5 * n_X0
        n_neutrons = int(n_events * neutron_mult * np.random.uniform(0.8, 1.2))
        
        # Electron/positron production
        n_electrons = int(n_events * 0.1 * n_X0 * np.random.uniform(0.8, 1.2))
        
        # Generate mock data
        # Photon energy spectrum (bremsstrahlung-like)
        beam_energy = 8000  # MeV
        photon_energies = np.random.exponential(beam_energy / 10, n_photons)
        photon_energies = photon_energies[photon_energies < beam_energy]
        n_photons = len(photon_energies)
        
        # Photon times (prompt, ~speed of light)
        c = 30.0  # cm/ns
        flight_time = (target_z / 2) / c  # From target center
        photon_times = flight_time + np.random.exponential(0.1, n_photons)
        
        # Photon angles (mostly forward, 1/gamma scaling)
        photon_angles = np.abs(np.random.exponential(0.05, n_photons))  # radians
        
        # Positions and momenta
        target_area_mm = target_x * target_y * 100  # mm^2
        photon_positions = np.column_stack([
            np.random.uniform(-target_x*5, target_x*5, n_photons),
            np.random.uniform(-target_y*5, target_y*5, n_photons),
            np.full(n_photons, -target_z * 5)  # At target exit (mm)
        ])
        
        # Forward-peaked momenta
        phi = np.random.uniform(0, 2*np.pi, n_photons)
        photon_momenta = np.column_stack([
            np.sin(photon_angles) * np.cos(phi),
            np.sin(photon_angles) * np.sin(phi),
            np.cos(photon_angles)
        ])
        
        # Neutrons (slower, broader)
        neutron_energies = np.random.exponential(100, n_neutrons)  # MeV
        # Neutron velocity: v = sqrt(2E/m), m_n ~ 940 MeV/c^2
        # For non-relativistic: v/c ~ sqrt(2E/m_n/c^2)
        beta_n = np.sqrt(2 * neutron_energies / 940) / 1  # crude approximation
        beta_n = np.clip(beta_n, 0.01, 0.5)
        neutron_times = (target_z / 2) / (beta_n * c) + np.random.exponential(5, n_neutrons)
        
        # Electrons (fast, like photons)
        electron_energies = np.random.exponential(500, n_electrons)
        electron_times = flight_time + np.random.exponential(0.2, n_electrons)
        
        return SimulationResult(
            photon_energies=photon_energies,
            photon_times=photon_times,
            photon_angles=photon_angles,
            photon_positions=photon_positions,
            photon_momenta=photon_momenta,
            neutron_energies=neutron_energies,
            neutron_times=neutron_times,
            electron_energies=electron_energies,
            electron_times=electron_times,
            n_primaries=n_events,
            target_z_cm=target_z,
            target_x_cm=target_x,
            target_y_cm=target_y,
            gap_cm=gap,
            run_time_seconds=0.1
        )


if __name__ == "__main__":
    # Test mock runner
    print("Testing MockGeant4Runner...")
    runner = MockGeant4Runner()
    
    result = runner.run(target_z=10, target_x=5, target_y=5, gap=100)
    
    print(f"\nSimulation Results:")
    print(f"  Photons: {result.n_photons}")
    print(f"  Neutrons: {result.n_neutrons}")
    print(f"  Electrons: {result.n_electrons}")
    print(f"  Photon energy range: {result.photon_energies.min():.1f} - {result.photon_energies.max():.1f} MeV")
    print(f"  Photon time range: {result.photon_times.min():.2f} - {result.photon_times.max():.2f} ns")
    print(f"  Neutron time range: {result.neutron_times.min():.2f} - {result.neutron_times.max():.2f} ns")
    
    # Test timing window
    t_min, t_max = 0, 5
    print(f"\n  Photons in [{t_min}, {t_max}] ns: {result.photons_in_timing_window(t_min, t_max)}")
    print(f"  Neutrons in [{t_min}, {t_max}] ns: {result.neutrons_in_timing_window(t_min, t_max)}")
