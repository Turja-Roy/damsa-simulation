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
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field

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


@dataclass
class ALPDecayPhotons:
    """
    Container for ALP decay photon kinematics from alplib.
    
    Each ALP decays to two photons (a -> gamma gamma).
    Arrays are indexed by event (one ALP per event).
    """
    # Photon 1 kinematics
    energy1: np.ndarray          # Energy in MeV
    theta1: np.ndarray           # Polar angle (rad) from beam axis
    phi1: np.ndarray             # Azimuthal angle (rad)
    px1: np.ndarray              # Momentum x-component
    py1: np.ndarray              # Momentum y-component  
    pz1: np.ndarray              # Momentum z-component
    
    # Photon 2 kinematics
    energy2: np.ndarray
    theta2: np.ndarray
    phi2: np.ndarray
    px2: np.ndarray
    py2: np.ndarray
    pz2: np.ndarray
    
    # Event weights (for proper normalization)
    weights: np.ndarray
    
    # Metadata
    axion_mass_MeV: float
    axion_coupling_MeV: float
    n_events: int
    
    @property
    def total_weight(self) -> float:
        """Total weighted number of decays."""
        return np.sum(self.weights)
    
    @property
    def all_energies(self) -> np.ndarray:
        """All photon energies (both photons from each decay)."""
        return np.concatenate([self.energy1, self.energy2])
    
    @property
    def all_thetas(self) -> np.ndarray:
        """All photon angles (both photons from each decay)."""
        return np.concatenate([self.theta1, self.theta2])
    
    @property
    def all_weights(self) -> np.ndarray:
        """Weights for all photons (duplicated for both photons per decay)."""
        return np.concatenate([self.weights, self.weights])
    
    def opening_angle(self) -> np.ndarray:
        """
        Calculate opening angle between the two decay photons.
        
        Returns angle in radians.
        """
        # Using dot product of momentum unit vectors
        p1_mag = np.sqrt(self.px1**2 + self.py1**2 + self.pz1**2)
        p2_mag = np.sqrt(self.px2**2 + self.py2**2 + self.pz2**2)
        
        dot = (self.px1*self.px2 + self.py1*self.py2 + self.pz1*self.pz2)
        cos_angle = np.clip(dot / (p1_mag * p2_mag + 1e-10), -1, 1)
        
        return np.arccos(cos_angle)
    
    def invariant_mass(self) -> np.ndarray:
        """
        Calculate invariant mass of the photon pair (should equal ALP mass).
        
        Returns mass in MeV.
        """
        E_tot = self.energy1 + self.energy2
        px_tot = self.px1 + self.px2
        py_tot = self.py1 + self.py2
        pz_tot = self.pz1 + self.pz2
        
        m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
        return np.sqrt(np.maximum(m2, 0))


class SeparabilityAnalysis:
    """
    Analyze separability between ALP signal photons and background photons.
    
    Uses kinematics from alplib (signal) and Geant4 (background) to compute
    metrics for how well signal can be distinguished from background.
    """
    
    def __init__(self, alp_photons: ALPDecayPhotons, bkg_energies: np.ndarray,
                 bkg_angles: np.ndarray, bkg_weights: np.ndarray = None):
        """
        Initialize separability analysis.
        
        Parameters
        ----------
        alp_photons : ALPDecayPhotons
            ALP decay photon kinematics from alplib
        bkg_energies : np.ndarray
            Background photon energies in MeV (from Geant4)
        bkg_angles : np.ndarray
            Background photon angles in radians (from Geant4)
        bkg_weights : np.ndarray, optional
            Background photon weights (default: unit weights)
        """
        self.alp = alp_photons
        self.bkg_energies = bkg_energies
        self.bkg_angles = bkg_angles
        self.bkg_weights = bkg_weights if bkg_weights is not None else np.ones(len(bkg_energies))
    
    def angular_separation_metric(self, angle_cut_deg: float = 5.0) -> Dict[str, float]:
        """
        Calculate angular separation between signal and background.
        
        ALP decay photons have specific angular distribution based on:
        - ALP production angle (forward-peaked, but with some spread)
        - Decay kinematics (isotropic in ALP rest frame, boosted to lab)
        
        Background bremsstrahlung is very forward-peaked.
        
        Parameters
        ----------
        angle_cut_deg : float
            Angular cut value in degrees
            
        Returns
        -------
        dict
            Separation metrics
        """
        angle_cut_rad = np.radians(angle_cut_deg)
        
        # Signal: ALP decay photons
        sig_angles = self.alp.all_thetas
        sig_weights = self.alp.all_weights
        
        # Background
        bkg_angles = self.bkg_angles
        bkg_weights = self.bkg_weights
        
        # Fraction of signal passing angular cut
        sig_passing = np.sum(sig_weights[sig_angles > angle_cut_rad])
        sig_total = np.sum(sig_weights)
        sig_efficiency = sig_passing / sig_total if sig_total > 0 else 0
        
        # Fraction of background passing angular cut (these survive as background)
        bkg_passing = np.sum(bkg_weights[bkg_angles > angle_cut_rad])
        bkg_total = np.sum(bkg_weights)
        bkg_efficiency = bkg_passing / bkg_total if bkg_total > 0 else 0
        
        # Background rejection = 1 - bkg_efficiency
        bkg_rejection = 1 - bkg_efficiency
        
        # Separation power: signal efficiency vs background rejection
        # ROC-like metric
        separation_power = sig_efficiency * bkg_rejection
        
        return {
            'angle_cut_deg': angle_cut_deg,
            'signal_efficiency': sig_efficiency,
            'background_efficiency': bkg_efficiency,
            'background_rejection': bkg_rejection,
            'separation_power': separation_power,
            'signal_mean_angle_deg': np.degrees(np.average(sig_angles, weights=sig_weights)) if sig_total > 0 else 0,
            'bkg_mean_angle_deg': np.degrees(np.average(bkg_angles, weights=bkg_weights)) if bkg_total > 0 else 0,
        }
    
    def energy_separation_metric(self, energy_cut_MeV: float = 100.0) -> Dict[str, float]:
        """
        Calculate energy-based separation between signal and background.
        
        Parameters
        ----------
        energy_cut_MeV : float
            Energy threshold cut in MeV
            
        Returns
        -------
        dict
            Separation metrics
        """
        # Signal
        sig_energies = self.alp.all_energies
        sig_weights = self.alp.all_weights
        
        # Background
        bkg_energies = self.bkg_energies
        bkg_weights = self.bkg_weights
        
        # Fraction passing energy cut
        sig_passing = np.sum(sig_weights[sig_energies > energy_cut_MeV])
        sig_total = np.sum(sig_weights)
        sig_efficiency = sig_passing / sig_total if sig_total > 0 else 0
        
        bkg_passing = np.sum(bkg_weights[bkg_energies > energy_cut_MeV])
        bkg_total = np.sum(bkg_weights)
        bkg_efficiency = bkg_passing / bkg_total if bkg_total > 0 else 0
        
        bkg_rejection = 1 - bkg_efficiency
        separation_power = sig_efficiency * bkg_rejection
        
        return {
            'energy_cut_MeV': energy_cut_MeV,
            'signal_efficiency': sig_efficiency,
            'background_efficiency': bkg_efficiency,
            'background_rejection': bkg_rejection,
            'separation_power': separation_power,
            'signal_mean_energy_MeV': np.average(sig_energies, weights=sig_weights) if sig_total > 0 else 0,
            'bkg_mean_energy_MeV': np.average(bkg_energies, weights=bkg_weights) if bkg_total > 0 else 0,
        }
    
    def optimal_cuts(self, n_angle_points: int = 20, n_energy_points: int = 20) -> Dict[str, Any]:
        """
        Find optimal cuts for maximum separation power.
        
        Returns
        -------
        dict
            Optimal cuts and their performance
        """
        # Scan angle cuts
        angle_cuts = np.linspace(0.5, 30, n_angle_points)
        best_angle_cut = 5.0
        best_angle_sep = 0
        
        for angle in angle_cuts:
            metrics = self.angular_separation_metric(angle)
            if metrics['separation_power'] > best_angle_sep:
                best_angle_sep = metrics['separation_power']
                best_angle_cut = angle
        
        # Scan energy cuts
        energy_cuts = np.logspace(0, 3, n_energy_points)  # 1 MeV to 1000 MeV
        best_energy_cut = 100.0
        best_energy_sep = 0
        
        for energy in energy_cuts:
            metrics = self.energy_separation_metric(energy)
            if metrics['separation_power'] > best_energy_sep:
                best_energy_sep = metrics['separation_power']
                best_energy_cut = energy
        
        # Combined metric using both cuts
        angle_metrics = self.angular_separation_metric(best_angle_cut)
        energy_metrics = self.energy_separation_metric(best_energy_cut)
        
        return {
            'optimal_angle_cut_deg': best_angle_cut,
            'optimal_angle_separation_power': best_angle_sep,
            'optimal_energy_cut_MeV': best_energy_cut,
            'optimal_energy_separation_power': best_energy_sep,
            'angle_metrics': angle_metrics,
            'energy_metrics': energy_metrics,
        }
    
    def compute_roc_curve(self, variable: str = 'angle', n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute ROC curve for signal vs background separation.
        
        Parameters
        ----------
        variable : str
            'angle' or 'energy'
        n_points : int
            Number of cut points to evaluate
            
        Returns
        -------
        dict
            ROC curve data: signal_efficiency, background_rejection, auc
        """
        if variable == 'angle':
            cuts = np.linspace(0.1, 45, n_points)
            sig_vals = self.alp.all_thetas
            bkg_vals = self.bkg_angles
        else:  # energy
            cuts = np.logspace(0, 4, n_points)  # 1 MeV to 10 GeV
            sig_vals = self.alp.all_energies
            bkg_vals = self.bkg_energies
        
        sig_weights = self.alp.all_weights
        bkg_weights = self.bkg_weights
        
        sig_total = np.sum(sig_weights)
        bkg_total = np.sum(bkg_weights)
        
        sig_eff = []
        bkg_rej = []
        
        for cut in cuts:
            if variable == 'angle':
                # For angle: signal at larger angles, background at smaller
                sig_pass = np.sum(sig_weights[np.degrees(sig_vals) > cut])
                bkg_pass = np.sum(bkg_weights[np.degrees(bkg_vals) > cut])
            else:
                # For energy: signal at higher energies
                sig_pass = np.sum(sig_weights[sig_vals > cut])
                bkg_pass = np.sum(bkg_weights[bkg_vals > cut])
            
            sig_eff.append(sig_pass / sig_total if sig_total > 0 else 0)
            bkg_rej.append(1 - bkg_pass / bkg_total if bkg_total > 0 else 1)
        
        sig_eff = np.array(sig_eff)
        bkg_rej = np.array(bkg_rej)
        
        # Compute AUC (area under ROC curve)
        # Sort by signal efficiency for proper integration
        sorted_idx = np.argsort(sig_eff)
        auc = np.trapezoid(bkg_rej[sorted_idx], sig_eff[sorted_idx])
        
        return {
            'signal_efficiency': sig_eff,
            'background_rejection': bkg_rej,
            'cuts': cuts,
            'auc': auc,
            'variable': variable
        }


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
                 axion_coupling: float = 1e-3,
                 exposure_days: float = 30.0,
                 detector_params: Optional[Dict] = None):
        """
        Initialize objective functions.
        
        Note on coupling units:
            axion_coupling is in GeV^-1 (standard physics convention).
            alplib internally uses MeV^-1, so we convert when calling alplib.
        """
        self.beam_current = beam_current_uA * 1e-6  # Convert to Amperes
        self.axion_mass = axion_mass_MeV
        self.axion_coupling = axion_coupling  # GeV^-1
        self.axion_coupling_mev = axion_coupling / 1000.0  # Convert GeV^-1 to MeV^-1: 1 GeV^-1 = 1e-3 MeV^-1
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
            
            # Note: alplib expects coupling in MeV^-1
            flux_obj = FluxPrimakoffIsotropic(
                photon_flux=flux,
                target=Material("W"),
                det_dist=det_dist_m,
                det_length=self.detector['length_m'],
                det_area=self.detector['area_m2'],
                axion_mass=self.axion_mass,
                axion_coupling=self.axion_coupling_mev,  # Use MeV^-1 units
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
    
    def get_alp_decay_photons(self, sim_result, n_samples: int = 10) -> Optional[ALPDecayPhotons]:
        """
        Extract ALP decay photon 4-vectors for separability analysis.
        
        Uses alplib's simulate_decay_4vectors() to get the kinematics
        of photons from ALP -> gamma gamma decay.
        
        Parameters
        ----------
        sim_result : SimulationResult
            Result from Geant4Runner (provides photon flux at target)
        n_samples : int
            Number of decay samples per ALP (for statistical coverage)
            
        Returns
        -------
        ALPDecayPhotons or None
            Container with decay photon kinematics, or None if alplib unavailable
        """
        if not ALPLIB_AVAILABLE:
            print("Warning: alplib not available for decay photon extraction")
            return None
        
        # Get photon flux from simulation
        flux = sim_result.get_photon_flux_array(bin_width_MeV=1.0)
        if len(flux) == 0:
            return None
        
        # Scale to rate (photons per second)
        scale_factor = self.electrons_per_second / sim_result.n_primaries
        flux[:, 1] = flux[:, 1] * scale_factor
        
        try:
            det_dist_m = sim_result.gap_cm / 100.0
            
            # Create Primakoff flux object
            flux_obj = FluxPrimakoffIsotropic(
                photon_flux=flux,
                target=Material("W"),
                det_dist=det_dist_m,
                det_length=self.detector['length_m'],
                det_area=self.detector['area_m2'],
                axion_mass=self.axion_mass,
                axion_coupling=self.axion_coupling_mev,
                n_samples=10000
            )
            
            # Simulate ALP production and propagation
            flux_obj.simulate()
            flux_obj.propagate()
            
            # Create event generator
            detector = Material("CsI")
            generator = PhotonEventGenerator(flux_obj, detector)
            
            # Get decay 4-vectors
            p4_photon1_list, p4_photon2_list, weight_list = generator.simulate_decay_4vectors(
                days_exposure=self.exposure_days,
                n_samples=n_samples
            )
            
            if len(p4_photon1_list) == 0:
                return None
            
            # Extract arrays from LorentzVector objects
            n_events = len(p4_photon1_list)
            
            # Photon 1
            energy1 = np.array([p4.p0 for p4 in p4_photon1_list])
            px1 = np.array([p4.p1 for p4 in p4_photon1_list])
            py1 = np.array([p4.p2 for p4 in p4_photon1_list])
            pz1 = np.array([p4.p3 for p4 in p4_photon1_list])
            p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)
            theta1 = np.arccos(np.clip(pz1 / (p1_mag + 1e-10), -1, 1))
            phi1 = np.arctan2(py1, px1)
            
            # Photon 2
            energy2 = np.array([p4.p0 for p4 in p4_photon2_list])
            px2 = np.array([p4.p1 for p4 in p4_photon2_list])
            py2 = np.array([p4.p2 for p4 in p4_photon2_list])
            pz2 = np.array([p4.p3 for p4 in p4_photon2_list])
            p2_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
            theta2 = np.arccos(np.clip(pz2 / (p2_mag + 1e-10), -1, 1))
            phi2 = np.arctan2(py2, px2)
            
            weights = np.array(weight_list)
            
            return ALPDecayPhotons(
                energy1=energy1, theta1=theta1, phi1=phi1,
                px1=px1, py1=py1, pz1=pz1,
                energy2=energy2, theta2=theta2, phi2=phi2,
                px2=px2, py2=py2, pz2=pz2,
                weights=weights,
                axion_mass_MeV=self.axion_mass,
                axion_coupling_MeV=self.axion_coupling_mev,
                n_events=n_events
            )
            
        except Exception as e:
            print(f"ALP decay photon extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_separability(self, sim_result, n_samples: int = 10) -> Optional[SeparabilityAnalysis]:
        """
        Compute separability analysis between ALP signal and background photons.
        
        Parameters
        ----------
        sim_result : SimulationResult
            Result from Geant4Runner
        n_samples : int
            Number of decay samples per ALP
            
        Returns
        -------
        SeparabilityAnalysis or None
            Separability analysis object, or None if failed
        """
        # Get ALP decay photons
        alp_photons = self.get_alp_decay_photons(sim_result, n_samples)
        if alp_photons is None:
            return None
        
        # Get background photons from simulation
        bkg_energies = sim_result.photon_energies
        bkg_angles = sim_result.photon_angles
        
        # Scale background to same exposure as signal
        scale_factor = self.electrons_per_second / sim_result.n_primaries
        bkg_weights = np.ones(len(bkg_energies)) * scale_factor * 86400 * self.exposure_days
        
        return SeparabilityAnalysis(alp_photons, bkg_energies, bkg_angles, bkg_weights)
    
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
        axion_coupling=1e-3,
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
    
    # Test ALP decay photon extraction and separability analysis
    print("\n--- Testing ALP Decay Photon Extraction ---")
    
    alp_photons = obj_func.get_alp_decay_photons(result, n_samples=5)
    if alp_photons is not None:
        print(f"  ALP decay events: {alp_photons.n_events}")
        print(f"  Total weighted decays: {alp_photons.total_weight:.2e}")
        print(f"  Mean photon 1 energy: {np.mean(alp_photons.energy1):.1f} MeV")
        print(f"  Mean photon 2 energy: {np.mean(alp_photons.energy2):.1f} MeV")
        print(f"  Mean opening angle: {np.degrees(np.mean(alp_photons.opening_angle())):.2f} deg")
        print(f"  Mean invariant mass: {np.mean(alp_photons.invariant_mass()):.1f} MeV")
        
        # Test separability analysis
        print("\n--- Testing Separability Analysis ---")
        sep_analysis = obj_func.compute_separability(result, n_samples=5)
        if sep_analysis is not None:
            # Angular separation
            ang_metrics = sep_analysis.angular_separation_metric(angle_cut_deg=5.0)
            print(f"\n  Angular separation (cut at 5 deg):")
            print(f"    Signal efficiency: {ang_metrics['signal_efficiency']:.3f}")
            print(f"    Background rejection: {ang_metrics['background_rejection']:.3f}")
            print(f"    Separation power: {ang_metrics['separation_power']:.3f}")
            print(f"    Signal mean angle: {ang_metrics['signal_mean_angle_deg']:.2f} deg")
            print(f"    Background mean angle: {ang_metrics['bkg_mean_angle_deg']:.2f} deg")
            
            # Energy separation
            eng_metrics = sep_analysis.energy_separation_metric(energy_cut_MeV=100.0)
            print(f"\n  Energy separation (cut at 100 MeV):")
            print(f"    Signal efficiency: {eng_metrics['signal_efficiency']:.3f}")
            print(f"    Background rejection: {eng_metrics['background_rejection']:.3f}")
            print(f"    Separation power: {eng_metrics['separation_power']:.3f}")
            
            # Optimal cuts
            print("\n  Finding optimal cuts...")
            opt_cuts = sep_analysis.optimal_cuts()
            print(f"    Optimal angle cut: {opt_cuts['optimal_angle_cut_deg']:.1f} deg")
            print(f"    Optimal energy cut: {opt_cuts['optimal_energy_cut_MeV']:.1f} MeV")
            print(f"    Best angle separation power: {opt_cuts['optimal_angle_separation_power']:.3f}")
            print(f"    Best energy separation power: {opt_cuts['optimal_energy_separation_power']:.3f}")
    else:
        print("  Note: alplib not available, skipping ALP decay extraction test")
