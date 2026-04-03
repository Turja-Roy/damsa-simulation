#!/usr/bin/env python3
"""
Objective Functions for DAMSA Multi-Objective Optimization

This module defines the objective functions for the optimization problem:
1. Signal rate (ALP production) - MAXIMIZE
2. Neutron background - MINIMIZE
3. Background photon (EM punch-through) - MINIMIZE or ensure separability
4. Signal-to-background ratio - MAXIMIZE

Key insight: Background photons should either be minimized or have angular/timing
distributions that allow separation from the ALP signal photons.

ALP signal photons: Come from target, decay in flight, isotropic in ALP frame
Background photons: Direct bremsstrahlung from target, forward-peaked

Author: DAMSA Collaboration
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Try to import alplib for signal calculation
# Add parent directory to path so alplib can be imported as a module
import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

ALPLIB_AVAILABLE = False
try:
    import alplib.fluxes as alplib_fluxes
    import alplib.materials as alplib_materials
    import alplib.generators as alplib_generators
    # Aliases for convenience
    FluxPrimakoffIsotropic = alplib_fluxes.FluxPrimakoffIsotropic
    PhotonEventGenerator = alplib_generators.PhotonEventGenerator
    Material = alplib_materials.Material
    ALPLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: alplib not installed. Using mock signal calculation. ({e})")


@dataclass
class ObjectiveValues:
    """Container for objective function values."""
    signal_rate: float              # Expected ALP decays per day
    neutron_background: float       # Neutrons in timing window
    photon_background: float        # Background photons in signal region
    em_background: float            # e+/e- background
    signal_to_background: float     # S/sqrt(B) or S/B
    photon_separability: float      # Angular separation metric
    
    def to_array(self, objectives: list = None) -> np.ndarray:
        """Convert to numpy array for optimization."""
        if objectives is None:
            objectives = ['signal', 'neutron', 'photon_bkg', 'separability']
        
        values = []
        for obj in objectives:
            if obj == 'signal':
                values.append(-self.signal_rate)  # Minimize negative = maximize
            elif obj == 'neutron':
                values.append(self.neutron_background)
            elif obj == 'photon_bkg':
                values.append(self.photon_background)
            elif obj == 'em':
                values.append(self.em_background)
            elif obj == 'snr':
                values.append(-self.signal_to_background)  # Maximize
            elif obj == 'separability':
                values.append(-self.photon_separability)  # Maximize separability
        
        return np.array(values)


class ObjectiveFunctions:
    """
    Calculate objective functions for DAMSA optimization.
    
    Objectives:
    1. Signal rate: Number of ALP decay photons detected
    2. Neutron background: Neutrons reaching detector in timing window
    3. Background photon: EM photons that could mimic signal
    4. Separability: How well signal and background are separated
    
    Parameters
    ----------
    beam_current_uA : float
        Beam current in microamperes (default: 62.5)
    axion_mass_MeV : float
        ALP mass for signal calculation
    axion_coupling : float
        ALP-photon coupling g_aγγ in MeV^-1
    exposure_days : float
        Exposure time for signal calculation
    detector_params : dict
        Detector geometry (area, length)
    """
    
    def __init__(self,
                 beam_current_uA: float = 62.5,
                 axion_mass_MeV: float = 100.0,
                 axion_coupling: float = 1e-4,
                 exposure_days: float = 30.0,
                 detector_params: Optional[Dict] = None):
        
        self.beam_current = beam_current_uA * 1e-6  # Convert to Amperes
        self.axion_mass = axion_mass_MeV
        self.axion_coupling = axion_coupling
        self.exposure_days = exposure_days
        
        # Default detector parameters
        self.detector = detector_params or {
            'area_m2': 0.25,      # 50x50 cm
            'length_m': 1.0,      # 1 m long
            'threshold_MeV': 10.0 # Detection threshold
        }
        
        # Electrons per second
        e_charge = 1.602176634e-19
        self.electrons_per_second = self.beam_current / e_charge
        
        # Signal region definition (based on ALP decay kinematics)
        # ALPs decay to 2 photons, each with E ~ m_a/2 in rest frame
        # Boosted, signal photons have energy ~ E_a * (1 + cos(theta*))/2
        self.signal_energy_min = 10.0  # MeV
        self.signal_energy_max = 8000.0  # MeV (beam energy)
        
        # Angular acceptance for signal
        # ALP decay photons: mostly forward but with spread from decay kinematics
        self.signal_angle_max = 20.0  # degrees
    
    def calculate(self, sim_result, 
                  t_min: float = 0.0, 
                  t_max: float = 10.0) -> ObjectiveValues:
        """
        Calculate all objective values from simulation result.
        
        Parameters
        ----------
        sim_result : SimulationResult
            Result from Geant4Runner
        t_min, t_max : float
            Timing window in ns
            
        Returns
        -------
        ObjectiveValues
            Container with all objective values
        """
        # 1. Calculate signal rate
        signal = self._calculate_signal(sim_result, t_min, t_max)
        
        # 2. Calculate neutron background
        neutrons = self._calculate_neutron_background(sim_result, t_min, t_max)
        
        # 3. Calculate background photon rate
        photon_bkg = self._calculate_photon_background(sim_result, t_min, t_max)
        
        # 4. Calculate EM (e+/e-) background
        em_bkg = self._calculate_em_background(sim_result, t_min, t_max)
        
        # 5. Calculate signal-to-background ratio
        total_bkg = neutrons + photon_bkg + em_bkg
        snr = signal / np.sqrt(total_bkg + 1) if total_bkg >= 0 else signal
        
        # 6. Calculate photon separability metric
        separability = self._calculate_photon_separability(sim_result, t_min, t_max)
        
        return ObjectiveValues(
            signal_rate=signal,
            neutron_background=neutrons,
            photon_background=photon_bkg,
            em_background=em_bkg,
            signal_to_background=snr,
            photon_separability=separability
        )
    
    def _calculate_signal(self, sim_result, t_min: float, t_max: float) -> float:
        """
        Calculate expected ALP signal rate.
        
        Uses alplib if available, otherwise uses a simple scaling model.
        """
        # Get photon flux
        flux = sim_result.get_photon_flux_array(bin_width_MeV=1.0)
        if len(flux) == 0:
            return 0.0
        
        # Scale to rate (photons per second)
        scale_factor = self.electrons_per_second / sim_result.n_primaries
        flux[:, 1] = flux[:, 1] * scale_factor
        
        if ALPLIB_AVAILABLE:
            return self._alplib_signal(flux, sim_result.gap_cm)
        else:
            return self._mock_signal(flux, sim_result.gap_cm)
    
    def _alplib_signal(self, flux: np.ndarray, gap_cm: float) -> float:
        """Calculate signal using alplib."""
        try:
            det_dist_m = gap_cm / 100.0
            
            flux_obj = FluxPrimakoffIsotropic(
                photon_flux=flux,
                target=Material("W"),
                det_dist=det_dist_m,
                det_length=self.detector['length_m'],
                det_area=self.detector['area_m2'],
                axion_mass=self.axion_mass,
                axion_coupling=self.axion_coupling,
                n_samples=10000
            )
            
            flux_obj.simulate()
            flux_obj.propagate()
            
            # PhotonEventGenerator needs (flux, detector_material)
            detector = Material("CsI")  # Use CsI for calorimeter
            generator = PhotonEventGenerator(flux_obj, detector)
            decays = generator.decays(
                days_exposure=self.exposure_days,
                threshold=self.detector['threshold_MeV']
            )
            
            return decays / self.exposure_days  # Rate per day
            
        except Exception as e:
            print(f"alplib calculation failed: {e}")
            return self._mock_signal(flux, gap_cm)
    
    def _mock_signal(self, flux: np.ndarray, gap_cm: float) -> float:
        """
        Simple mock signal calculation for testing.
        
        Signal scales as:
        - Photon flux (linear)
        - Coupling^4 (production * decay)
        - 1/gap^2 (solid angle)
        - exp(-gap/decay_length) for short-lived ALPs
        """
        total_flux = flux[:, 1].sum()
        
        # Rough scaling model
        # Production: g^2 * flux
        # Decay probability: g^2 * L / L_decay
        # Solid angle: area / gap^2
        
        gap_m = gap_cm / 100.0
        coupling = self.axion_coupling
        mass = self.axion_mass
        
        # Decay length estimate: L ~ E/m * 1/(g^2 * m)
        # For m=100 MeV, g=1e-4: L ~ few meters
        E_typical = 1000  # MeV
        decay_length = E_typical / mass * 1.0 / (coupling**2 * mass + 1e-10)
        decay_length = np.clip(decay_length, 0.1, 1e6)
        
        # Signal estimate
        signal = (total_flux * coupling**4 * 
                  self.detector['area_m2'] / (gap_m**2 + 0.1) *
                  (1 - np.exp(-gap_m / decay_length)) *
                  self.exposure_days)
        
        return max(signal, 0.0)
    
    def _calculate_neutron_background(self, sim_result, t_min: float, t_max: float) -> float:
        """
        Calculate neutron background in timing window.
        
        Scale to rate using beam current.
        """
        if len(sim_result.neutron_times) == 0:
            return 0.0
        
        # Count neutrons in timing window
        mask = (sim_result.neutron_times >= t_min) & (sim_result.neutron_times <= t_max)
        count = np.sum(mask)
        
        # Scale to rate
        scale_factor = self.electrons_per_second / sim_result.n_primaries
        rate_per_second = count * scale_factor
        
        # Convert to per day for exposure
        rate_per_day = rate_per_second * 86400
        
        return rate_per_day * self.exposure_days
    
    def _calculate_photon_background(self, sim_result, t_min: float, t_max: float) -> float:
        """
        Calculate background photon rate in signal region.
        
        Background photons are direct bremsstrahlung from the target that could
        mimic ALP decay photons. We need to count those that:
        1. Fall within the timing window
        2. Have energy in the signal energy range
        3. Have angle within the signal acceptance
        
        The key is to identify photons that would be indistinguishable from signal.
        """
        if len(sim_result.photon_energies) == 0:
            return 0.0
        
        # Apply cuts to identify "signal-like" background photons
        # Timing cut
        time_mask = (sim_result.photon_times >= t_min) & (sim_result.photon_times <= t_max)
        
        # Energy cut (signal region)
        energy_mask = (sim_result.photon_energies >= self.signal_energy_min) & \
                      (sim_result.photon_energies <= self.signal_energy_max)
        
        # Angular cut
        angle_deg = np.degrees(sim_result.photon_angles)
        angle_mask = angle_deg <= self.signal_angle_max
        
        # Combined: photons that could fake signal
        signal_like_mask = time_mask & energy_mask & angle_mask
        count = np.sum(signal_like_mask)
        
        # Scale to rate
        scale_factor = self.electrons_per_second / sim_result.n_primaries
        rate_per_second = count * scale_factor
        rate_per_day = rate_per_second * 86400
        
        return rate_per_day * self.exposure_days
    
    def _calculate_em_background(self, sim_result, t_min: float, t_max: float) -> float:
        """Calculate electron/positron background in timing window."""
        if len(sim_result.electron_times) == 0:
            return 0.0
        
        mask = (sim_result.electron_times >= t_min) & (sim_result.electron_times <= t_max)
        count = np.sum(mask)
        
        scale_factor = self.electrons_per_second / sim_result.n_primaries
        rate_per_second = count * scale_factor
        rate_per_day = rate_per_second * 86400
        
        return rate_per_day * self.exposure_days
    
    def _calculate_photon_separability(self, sim_result, t_min: float, t_max: float) -> float:
        """
        Calculate a metric for how separable background photons are from signal.
        
        The separability metric considers:
        1. Angular distribution: ALP decay photons have characteristic angle
        2. Energy spectrum: Different from bremsstrahlung
        3. Timing: Prompt vs delayed
        
        Higher value = better separability (easier to reject background)
        
        Key insight: 
        - ALP signal photons come from ALP decay, have specific kinematics
        - Background photons are bremsstrahlung, forward-peaked at low angles
        
        If background is concentrated at very forward angles (< 5 deg),
        we can use an angular cut to separate them from more isotropic ALP signal.
        """
        if len(sim_result.photon_angles) == 0:
            return 0.0
        
        # Apply timing cut
        time_mask = (sim_result.photon_times >= t_min) & (sim_result.photon_times <= t_max)
        
        angles_deg = np.degrees(sim_result.photon_angles[time_mask])
        energies = sim_result.photon_energies[time_mask]
        
        if len(angles_deg) == 0:
            return 0.0
        
        # Metric 1: Angular spread
        # Higher spread = less forward-peaked = harder to separate by angle cut
        # But we want INVERTED: if most background is at < 5 deg, that's GOOD
        # because we can cut on angle
        
        # Fraction of photons at very forward angles (< 5 degrees)
        # These are clearly background and easy to separate
        very_forward_frac = np.sum(angles_deg < 5.0) / len(angles_deg)
        
        # Fraction in "ambiguous" region (5-20 degrees)
        # ALP signal might also be here, so these are harder to separate
        ambiguous_frac = np.sum((angles_deg >= 5.0) & (angles_deg <= 20.0)) / len(angles_deg)
        
        # Fraction at large angles (> 20 degrees)
        # These are easy to reject (outside signal acceptance)
        large_angle_frac = np.sum(angles_deg > 20.0) / len(angles_deg)
        
        # Metric 2: Energy distribution
        # High energy photons (> 100 MeV) are more signal-like
        # Low energy photons (< 10 MeV) are easy to reject
        low_energy_frac = np.sum(energies < 10) / len(energies) if len(energies) > 0 else 0
        
        # Separability score:
        # - Bonus for photons at very forward angles (can use angular cut)
        # - Bonus for photons at very high angles (outside signal acceptance)
        # - Bonus for low energy photons (below threshold)
        # - Penalty for ambiguous region
        
        separability = (
            0.3 * very_forward_frac +    # Easy to cut with angle
            0.3 * large_angle_frac +     # Outside acceptance
            0.2 * low_energy_frac +      # Below threshold
            -0.2 * ambiguous_frac        # Hard to separate
        )
        
        # Normalize to [0, 1]
        separability = (separability + 0.2) / 0.8  # Shift and scale
        separability = np.clip(separability, 0, 1)
        
        return separability


class FigureOfMerit:
    """
    Combined figure of merit for single-objective optimization or ranking.
    
    Different FoM formulations for different physics priorities.
    """
    
    @staticmethod
    def signal_over_background(objectives: ObjectiveValues) -> float:
        """Simple S/B ratio."""
        total_bkg = (objectives.neutron_background + 
                     objectives.photon_background + 
                     objectives.em_background)
        if total_bkg <= 0:
            return objectives.signal_rate * 1000  # Very good
        return objectives.signal_rate / total_bkg
    
    @staticmethod
    def significance(objectives: ObjectiveValues) -> float:
        """Statistical significance S/sqrt(B)."""
        total_bkg = (objectives.neutron_background + 
                     objectives.photon_background + 
                     objectives.em_background)
        return objectives.signal_rate / np.sqrt(total_bkg + 1)
    
    @staticmethod
    def punzi_fom(objectives: ObjectiveValues, n_sigma: float = 5.0) -> float:
        """
        Punzi figure of merit: S / (n_sigma/2 + sqrt(B))
        
        Better for optimization when signal is small.
        """
        total_bkg = (objectives.neutron_background + 
                     objectives.photon_background + 
                     objectives.em_background)
        return objectives.signal_rate / (n_sigma/2 + np.sqrt(total_bkg + 1))
    
    @staticmethod
    def weighted_composite(objectives: ObjectiveValues,
                           w_signal: float = 0.4,
                           w_neutron: float = 0.2,
                           w_photon: float = 0.2,
                           w_separability: float = 0.2) -> float:
        """
        Weighted composite figure of merit.
        
        Combines signal maximization, background minimization,
        and separability into single metric.
        """
        # Normalize each component (higher = better for all)
        signal_norm = objectives.signal_rate / 1000  # Scale to ~1
        neutron_norm = 1.0 / (1.0 + objectives.neutron_background / 1000)
        photon_norm = 1.0 / (1.0 + objectives.photon_background / 1000)
        sep_norm = objectives.photon_separability
        
        fom = (w_signal * signal_norm + 
               w_neutron * neutron_norm +
               w_photon * photon_norm +
               w_separability * sep_norm)
        
        return fom


def apply_timing_optimization(sim_result, 
                               objective_func: ObjectiveFunctions,
                               t_range: Tuple[float, float] = (0, 100),
                               n_points: int = 50) -> Tuple[float, float, ObjectiveValues]:
    """
    Find optimal timing window for given geometry.
    
    Parameters
    ----------
    sim_result : SimulationResult
        Geant4 simulation result
    objective_func : ObjectiveFunctions
        Objective function calculator
    t_range : tuple
        Range of timing values to scan
    n_points : int
        Number of grid points
        
    Returns
    -------
    tuple
        (optimal_t_min, optimal_t_max, best_objectives)
    """
    best_fom = -np.inf
    best_t_min, best_t_max = 0, 10
    best_obj = None
    
    # Grid search over timing windows
    t_values = np.linspace(t_range[0], t_range[1], n_points)
    
    for i, t_min in enumerate(t_values[:-1]):
        for t_max in t_values[i+1:]:
            if t_max - t_min < 0.5:  # Minimum window width
                continue
            
            obj = objective_func.calculate(sim_result, t_min, t_max)
            fom = FigureOfMerit.significance(obj)
            
            if fom > best_fom:
                best_fom = fom
                best_t_min, best_t_max = t_min, t_max
                best_obj = obj
    
    return best_t_min, best_t_max, best_obj


if __name__ == "__main__":
    # Test with mock data
    from geant4_runner import MockGeant4Runner, SimulationResult
    
    print("Testing ObjectiveFunctions...")
    
    # Create mock simulation
    runner = MockGeant4Runner(verbose=False)
    result = runner.run(target_z=10, target_x=5, target_y=5, gap=100)
    
    # Calculate objectives
    obj_func = ObjectiveFunctions(
        axion_mass_MeV=100,
        axion_coupling=1e-4,
        exposure_days=30
    )
    
    objectives = obj_func.calculate(result, t_min=0, t_max=5)
    
    print("\nObjective Values:")
    print(f"  Signal rate: {objectives.signal_rate:.2e} per day")
    print(f"  Neutron background: {objectives.neutron_background:.2e}")
    print(f"  Photon background: {objectives.photon_background:.2e}")
    print(f"  EM background: {objectives.em_background:.2e}")
    print(f"  S/sqrt(B): {objectives.signal_to_background:.2f}")
    print(f"  Photon separability: {objectives.photon_separability:.3f}")
    
    # Test figure of merit calculations
    print("\nFigures of Merit:")
    print(f"  S/B: {FigureOfMerit.signal_over_background(objectives):.3f}")
    print(f"  Significance: {FigureOfMerit.significance(objectives):.3f}")
    print(f"  Punzi: {FigureOfMerit.punzi_fom(objectives):.3f}")
    print(f"  Weighted: {FigureOfMerit.weighted_composite(objectives):.3f}")
    
    # Test timing optimization
    print("\nOptimizing timing window...")
    best_t_min, best_t_max, best_obj = apply_timing_optimization(
        result, obj_func, t_range=(0, 50), n_points=20
    )
    print(f"  Optimal window: [{best_t_min:.1f}, {best_t_max:.1f}] ns")
    print(f"  Best significance: {best_obj.signal_to_background:.2f}")
