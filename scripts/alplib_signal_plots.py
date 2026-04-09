#!/usr/bin/env python3
"""
ALP Signal Visualization and Analysis Module for DAMSA Optimization

This module provides comprehensive tools for ALP (Axion-Like Particle) signal
calculations and visualization using alplib. It includes:

1. Flux loading and conversion from Geant4 output to alplib format
2. ALP signal calculation via Primakoff production
3. Energy and angular distribution plotting
4. Mass and coupling scans with sensitivity contours
5. Comparison between different detector configurations

REQUIRES alplib to be installed/available. No mock calculations.

Usage:
    # Single configuration analysis
    python alplib_signal_plots.py --flux-file photon_flux.csv --alp-mass 100
    
    # With scaling for beam current
    python alplib_signal_plots.py --flux-file photon_flux.csv --nprimaries 1000 --alp-mass 100
    
    # Full report with all plots
    python alplib_signal_plots.py --flux-file photon_flux.csv --full-report
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import argparse
import sys

# Add project root to path for alplib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try ROOT first for consistent output
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT_AVAILABLE = True
except ImportError:
    ROOT_AVAILABLE = False

# Matplotlib fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LogNorm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# alplib imports - REQUIRED
try:
    import alplib.fluxes as alplib_fluxes
    import alplib.materials as alplib_materials
    import alplib.generators as alplib_generators
    ALPLIB_AVAILABLE = True
except ImportError:
    ALPLIB_AVAILABLE = False

# =============================================================================
# ALP PROXY MASS CALCULATION
# =============================================================================
# 
# The opening angle for a → γγ decay is: θ_open ≈ 2*m_a/E_a (ultra-relativistic)
# This depends only on the boost γ = E_a/m_a, not the absolute values.
#
# To preserve opening angle with a proxy ALP mass (when using lower-energy photons
# from a beam dump), use:
#   m_proxy = E_proxy / gamma_target
#   where gamma_target = E_target / m_target (the real ALP scenario you care about)
#
# Example: Preserve opening angle of 100 MeV ALP at E_a = 1 GeV (γ = 10)
#   - With 80 MeV photons from dump: m_proxy = 80/10 = 8 MeV
#   - With 60 MeV photons from dump: m_proxy = 60/10 = 6 MeV
#   - With 100 MeV photons from dump: m_proxy = 100/10 = 10 MeV
#
# IMPORTANT CAVEAT: This preserves opening angle but NOT decay length.
# Decay length scales as L ∝ γ/m_a³, so the proxy will have a much longer
# decay length than the real ALP. This is fine for geometry optimization
# (detector placement, angular coverage) but NOT for absolute signal rates.
# =============================================================================

# Target ALP scenario for proxy calculations
TARGET_ALP_MASS_MEV = 100.0     # Target ALP mass you want to preserve opening angle for (MeV)
TARGET_ALP_ENERGY_MEV = 1000.0  # Assumed typical ALP energy from 8 GeV electron beam (MeV)
# gamma = 10 for the target scenario


def calculate_proxy_mass(photon_energy_MeV: float,
                         target_alp_mass_MeV: float = TARGET_ALP_MASS_MEV,
                         target_alp_energy_MeV: float = TARGET_ALP_ENERGY_MEV) -> float:
    """
    Calculate proxy ALP mass that preserves opening angle.
    
    Parameters
    ----------
    photon_energy_MeV : float
        Available photon energy from beam dump (MeV)
    target_alp_mass_MeV : float
        Target ALP mass to preserve opening angle for (MeV)
    target_alp_energy_MeV : float
        Assumed typical ALP energy in real scenario (MeV)
    
    Returns
    -------
    float
        Proxy ALP mass in MeV that gives same opening angle
    """
    gamma_target = target_alp_energy_MeV / target_alp_mass_MeV
    proxy_mass = photon_energy_MeV / gamma_target
    return proxy_mass


def get_default_proxy_mass(photon_energy_MeV: float = 80.0) -> float:
    """
    Get default proxy mass for typical photon energies from tungsten dump.
    
    Default assumes target ALP scenario: m_a = 100 MeV at E_a = 1 GeV (γ = 10)
    
    Parameters
    ----------
    photon_energy_MeV : float
        Typical photon energy from dump (default: 80 MeV)
    
    Returns
    -------
    float
        Proxy ALP mass in MeV
    """
    return calculate_proxy_mass(photon_energy_MeV, TARGET_ALP_MASS_MEV, TARGET_ALP_ENERGY_MEV)


def check_alplib_available():
    """Check if alplib is available and raise error if not."""
    if not ALPLIB_AVAILABLE:
        raise ImportError(
            "alplib is required for ALP signal calculations.\n"
            "Clone from: https://github.com/athompson-git/alplib\n"
            "Then add to PYTHONPATH or place in project root."
        )


def load_flux_for_alplib(csv_path: str, n_primaries: int, 
                         beam_current_uA: float = 62.5) -> np.ndarray:
    """
    Load Geant4 photon flux and convert to alplib format with proper scaling.
    
    This function reads the raw photon flux CSV from Geant4 simulation,
    bins the energies, and scales by beam current to get photons/second.
    
    Parameters
    ----------
    csv_path : str
        Path to photon flux CSV (must have 'energy_MeV' or 'energy' column)
    n_primaries : int
        Number of primary electrons simulated
    beam_current_uA : float
        Beam current in microamperes (default: 62.5 for LCLS-II)
        
    Returns
    -------
    np.ndarray
        2D array [[energy_MeV, rate_per_second], ...] suitable for alplib
    """
    df = pd.read_csv(csv_path)
    
    # Handle different column naming conventions
    if 'energy_MeV' in df.columns:
        energies = df['energy_MeV'].values
    elif 'energy' in df.columns:
        energies = df['energy'].values
    else:
        raise ValueError("CSV must have 'energy_MeV' or 'energy' column")
    
    weights = df['weight'].values if 'weight' in df.columns else np.ones(len(energies))
    
    # Calculate scaling factor: photons per electron -> photons per second
    e_charge = 1.602176634e-19
    beam_current = beam_current_uA * 1e-6
    electrons_per_second = beam_current / e_charge
    scale_factor = electrons_per_second / n_primaries
    
    # Bin the photon energies (1 MeV bins)
    max_energy = np.ceil(energies.max())
    bins = np.arange(0, max_energy + 1, 1.0)
    counts, bin_edges = np.histogram(energies, bins=bins, weights=weights)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    rates = counts * scale_factor
    
    # Create flux array (only non-zero bins)
    flux = np.column_stack([bin_centers, rates])
    flux = flux[flux[:, 1] > 0]
    
    return flux


class ALPSignalVisualizer:
    """
    Visualize ALP signal predictions from alplib.
    
    Creates comprehensive plots showing expected ALP decay signals
    for different detector configurations and physics parameters.
    
    REQUIRES alplib - will raise ImportError if not available.
    """
    
    def __init__(self, output_dir: str = "alp_signal_plots"):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory for output plots
            
        Raises
        ------
        ImportError
            If alplib is not available
        """
        check_alplib_available()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default physics parameters
        self.beam_energy_GeV = 8.0
        self.beam_current_uA = 62.5
        self.exposure_days = 30.0
        
        # Store calculated results
        self.results = {}
        
    def load_photon_flux(self, flux_file: str) -> pd.DataFrame:
        """
        Load photon flux from Geant4 output.
        
        Parameters
        ----------
        flux_file : str
            Path to flux CSV file
            
        Returns
        -------
        pd.DataFrame
            Flux data
        """
        df = pd.read_csv(flux_file)
        
        # Expected columns: energy, theta, phi, x, y, z, weight
        # Handle both 'energy' and 'energy_MeV' column names
        if 'energy_MeV' in df.columns:
            df = df.rename(columns={'energy_MeV': 'energy'})
        elif 'energy_GeV' in df.columns:
            df['energy'] = df['energy_GeV'] * 1000  # Convert to MeV
            df = df.drop(columns=['energy_GeV'])
        
        required_cols = ['energy']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def calculate_alp_signal(
        self,
        flux_df: pd.DataFrame,
        alp_mass_MeV: float = 100.0,
        coupling: float = 1e-3,
        detector_distance_m: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate ALP signal using alplib.
        
        Parameters
        ----------
        flux_df : pd.DataFrame
            Photon flux data
        alp_mass_MeV : float
            ALP mass in MeV
        coupling : float
            ALP-photon coupling g_aγγ in GeV^-1 (standard convention)
            Note: alplib internally uses MeV^-1, so we convert.
        detector_distance_m : float
            Distance to detector in meters
            
        Returns
        -------
        dict
            Signal calculation results
            
        Raises
        ------
        ImportError
            If alplib is not available
        RuntimeError
            If alplib calculation fails
        """
        check_alplib_available()
        
        # Convert flux to alplib format
        energies = flux_df['energy'].values  # MeV
        
        # Extract angular information if available
        if 'theta' in flux_df.columns:
            thetas = flux_df['theta'].values
        elif 'px' in flux_df.columns and 'pz' in flux_df.columns:
            px = flux_df['px'].values
            py = flux_df.get('py', np.zeros_like(px)).values
            pz = flux_df['pz'].values
            p_total = np.sqrt(px**2 + py**2 + pz**2)
            thetas = np.arccos(np.clip(pz / (p_total + 1e-10), -1, 1))
        else:
            thetas = None
        
        # Create alplib flux object using Primakoff production
        try:
            # Convert to alplib format: [[energy, weight], ...]
            # Each photon gets weight=1 (will be scaled by beam current externally)
            photon_flux_array = np.column_stack([energies, np.ones_like(energies)])
            
            # Create target material
            target_material = alplib_materials.Material("W")
            
            # Create detector material for event generation
            detector_material = alplib_materials.Material("CsI")
            
            # Convert coupling from GeV^-1 to MeV^-1 (alplib uses MeV^-1)
            coupling_mev = coupling / 1000.0
            
            # Create Primakoff flux generator
            # Note: alplib uses meters for distances, MeV^-1 for coupling
            flux_obj = alplib_fluxes.FluxPrimakoffIsotropic(
                photon_flux=photon_flux_array,
                target=target_material,
                det_dist=detector_distance_m,
                det_length=1.0,  # 1 meter detector length
                det_area=0.25,   # 50x50 cm detector area
                axion_mass=alp_mass_MeV,
                axion_coupling=coupling_mev,  # MeV^-1
                n_samples=1000
            )
            
            # Simulate ALP production from photon flux
            flux_obj.simulate()
            
            # Propagate ALPs and calculate decay probabilities
            flux_obj.propagate()
            
            # Create event generator for photon channel (ALP -> gamma gamma)
            generator = alplib_generators.PhotonEventGenerator(
                flux_obj,
                detector_material
            )
            
            # Get total decay events for exposure period
            # decays() returns a scalar (total number of events)
            n_events = generator.decays(
                days_exposure=self.exposure_days,
                threshold=0.1  # MeV threshold
            )
            
            # Get decay energies from flux object (axion energies ~ photon energies)
            decay_energies = np.array(flux_obj.axion_energy) if hasattr(flux_obj, 'axion_energy') else np.array([])
            decay_angles = np.array(flux_obj.axion_angle) if hasattr(flux_obj, 'axion_angle') and len(flux_obj.axion_angle) > 0 else None
            
            result = {
                'alp_mass_MeV': alp_mass_MeV,
                'coupling': coupling,
                'n_events': n_events,
                'flux_integral': len(energies),
                'energies': energies,
                'thetas': thetas,
                'decay_energies': decay_energies,
                'decay_angles': decay_angles,
                'exposure_days': self.exposure_days,
                'detector_distance_m': detector_distance_m
            }
            
        except Exception as e:
            raise RuntimeError(f"alplib calculation failed: {e}")
        
        return result
    
    def plot_energy_distribution(
        self,
        result: Dict[str, Any],
        save: bool = True
    ) -> Any:
        """
        Plot energy distribution of ALP decay photons.
        
        Parameters
        ----------
        result : dict
            ALP calculation result
        save : bool
            Save plot to file
            
        Returns
        -------
        Figure
            Plot figure
        """
        if ROOT_AVAILABLE:
            return self._plot_energy_root(result, save)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_energy_matplotlib(result, save)
        else:
            raise ImportError("No plotting library available")
    
    def _plot_energy_root(self, result: Dict[str, Any], save: bool) -> ROOT.TCanvas:
        """Create energy distribution plot with ROOT."""
        mass = result['alp_mass_MeV']
        coupling = result['coupling']
        
        canvas = ROOT.TCanvas("c_energy", "ALP Decay Energy", 800, 600)
        
        # Input flux energy distribution
        h_flux = ROOT.TH1D("h_flux", "Photon Flux Energy;Energy [MeV];Counts",
                          100, 0, 1000)
        for e in result['energies']:
            h_flux.Fill(e)
        
        h_flux.SetLineColor(ROOT.kBlue)
        h_flux.SetLineWidth(2)
        h_flux.Draw("HIST")
        
        # ALP decay energy distribution (if available)
        if 'decay_energies' in result and len(result['decay_energies']) > 0:
            h_decay = ROOT.TH1D("h_decay", "ALP Decay Photons",
                               100, 0, 500)
            for e in result['decay_energies']:
                h_decay.Fill(e)
            
            h_decay.SetLineColor(ROOT.kRed)
            h_decay.SetLineWidth(2)
            h_decay.Scale(h_flux.GetMaximum() / h_decay.GetMaximum() * 0.5)
            h_decay.Draw("HIST SAME")
        
        # Legend
        leg = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
        leg.AddEntry(h_flux, "Input Photon Flux", "l")
        if 'decay_energies' in result:
            leg.AddEntry(h_decay, f"ALP Decay (m={mass:.0f} MeV)", "l")
        leg.Draw()
        
        # Add info text
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextSize(0.035)
        text.DrawLatex(0.15, 0.85, f"m_a = {mass:.0f} MeV")
        text.DrawLatex(0.15, 0.80, f"g_{{a#gamma#gamma}} = {coupling:.1e}")
        text.DrawLatex(0.15, 0.75, f"N_{{events}} = {result['n_events']:.0f}")
        
        canvas.Update()
        
        if save:
            png_path = self.output_dir / f"energy_dist_m{mass:.0f}.png"
            canvas.SaveAs(str(png_path))
            root_path = self.output_dir / f"energy_dist_m{mass:.0f}.root"
            canvas.SaveAs(str(root_path))
        
        return canvas
    
    def _plot_energy_matplotlib(self, result: Dict[str, Any], save: bool) -> plt.Figure:
        """Create energy distribution plot with matplotlib."""
        mass = result['alp_mass_MeV']
        coupling = result['coupling']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Input flux
        ax.hist(result['energies'], bins=100, range=(0, 1000),
                alpha=0.7, label='Input Photon Flux', color='blue')
        
        # ALP decay
        if 'decay_energies' in result and len(result['decay_energies']) > 0:
            ax.hist(result['decay_energies'], bins=100, range=(0, 500),
                   alpha=0.7, label=f'ALP Decay (m={mass:.0f} MeV)', color='red')
        
        ax.set_xlabel('Energy [MeV]', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('ALP Signal Energy Distribution', fontsize=14)
        ax.legend()
        
        # Info text
        info_text = f"$m_a$ = {mass:.0f} MeV\n$g_{{a\\gamma\\gamma}}$ = {coupling:.1e}\n$N_{{events}}$ = {result['n_events']:.0f}"
        ax.text(0.95, 0.95, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            png_path = self.output_dir / f"energy_dist_m{mass:.0f}.png"
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")
        
        return fig
    
    def plot_angular_distribution(
        self,
        result: Dict[str, Any],
        save: bool = True
    ) -> Any:
        """
        Plot angular distribution of ALP decays.
        """
        if 'decay_angles' not in result or len(result['decay_angles']) == 0:
            print("No angular distribution data available")
            return None
        
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            angles_deg = np.degrees(result['decay_angles'])
            ax.hist(angles_deg, bins=50, alpha=0.7, color='green')
            
            ax.set_xlabel('Angle [degrees]', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            ax.set_title('ALP Decay Angular Distribution', fontsize=14)
            
            mass = result['alp_mass_MeV']
            
            plt.tight_layout()
            
            if save:
                png_path = self.output_dir / f"angular_dist_m{mass:.0f}.png"
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
            
            return fig
        
        return None
    
    def plot_mass_scan(
        self,
        flux_df: pd.DataFrame,
        mass_range: Tuple[float, float] = (10, 500),
        n_points: int = 20,
        coupling: float = 1e-3,
        save: bool = True
    ) -> Any:
        """
        Plot signal rate vs ALP mass.
        
        Parameters
        ----------
        flux_df : pd.DataFrame
            Photon flux data
        mass_range : tuple
            Min and max ALP mass in MeV
        n_points : int
            Number of mass points
        coupling : float
            ALP-photon coupling
        save : bool
            Save plot
            
        Returns
        -------
        Figure
            Plot figure
        """
        masses = np.linspace(mass_range[0], mass_range[1], n_points)
        n_events = []
        
        for mass in masses:
            result = self.calculate_alp_signal(flux_df, mass, coupling)
            n_events.append(result['n_events'])
        
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.semilogy(masses, n_events, 'o-', linewidth=2, markersize=6)
            
            ax.set_xlabel('ALP Mass [MeV]', fontsize=12)
            ax.set_ylabel('Expected Events', fontsize=12)
            ax.set_title(f'ALP Signal Rate vs Mass (g = {coupling:.1e})', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add threshold line
            ax.axhline(y=3, color='red', linestyle='--', label='3 events threshold')
            ax.legend()
            
            plt.tight_layout()
            
            if save:
                png_path = self.output_dir / "mass_scan.png"
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
            
            return fig
        
        return None
    
    def plot_coupling_scan(
        self,
        flux_df: pd.DataFrame,
        alp_mass_MeV: float = 100.0,
        coupling_range: Tuple[float, float] = (1e-6, 1e-2),
        n_points: int = 20,
        save: bool = True
    ) -> Any:
        """
        Plot signal rate vs coupling strength.
        """
        couplings = np.logspace(np.log10(coupling_range[0]), 
                                np.log10(coupling_range[1]), n_points)
        n_events = []
        
        for coupling in couplings:
            result = self.calculate_alp_signal(flux_df, alp_mass_MeV, coupling)
            n_events.append(result['n_events'])
        
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.loglog(couplings, n_events, 'o-', linewidth=2, markersize=6)
            
            ax.set_xlabel('$g_{a\\gamma\\gamma}$ [GeV$^{-1}$]', fontsize=12)
            ax.set_ylabel('Expected Events', fontsize=12)
            ax.set_title(f'ALP Signal Rate vs Coupling (m = {alp_mass_MeV:.0f} MeV)', fontsize=14)
            ax.grid(True, alpha=0.3, which='both')
            
            # Add threshold line
            ax.axhline(y=3, color='red', linestyle='--', label='3 events threshold')
            ax.legend()
            
            plt.tight_layout()
            
            if save:
                png_path = self.output_dir / "coupling_scan.png"
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
            
            return fig
        
        return None
    
    def plot_sensitivity_contour(
        self,
        flux_df: pd.DataFrame,
        mass_range: Tuple[float, float] = (10, 500),
        coupling_range: Tuple[float, float] = (1e-6, 1e-2),
        n_mass: int = 20,
        n_coupling: int = 20,
        save: bool = True
    ) -> Any:
        """
        Plot 2D sensitivity contour (mass vs coupling).
        """
        masses = np.linspace(mass_range[0], mass_range[1], n_mass)
        couplings = np.logspace(np.log10(coupling_range[0]),
                                np.log10(coupling_range[1]), n_coupling)
        
        events_grid = np.zeros((n_coupling, n_mass))
        
        for i, coupling in enumerate(couplings):
            for j, mass in enumerate(masses):
                result = self.calculate_alp_signal(flux_df, mass, coupling)
                events_grid[i, j] = max(0.1, result['n_events'])  # Avoid log(0)
        
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            M, C = np.meshgrid(masses, couplings)
            
            pcm = ax.pcolormesh(M, C, events_grid, 
                               norm=LogNorm(vmin=0.1, vmax=events_grid.max()),
                               cmap='viridis')
            
            # Add contour lines for 3 and 10 events
            cs = ax.contour(M, C, events_grid, levels=[3, 10, 100],
                           colors=['red', 'orange', 'yellow'], linewidths=2)
            ax.clabel(cs, inline=True, fontsize=10, fmt='%d events')
            
            ax.set_xlabel('ALP Mass [MeV]', fontsize=12)
            ax.set_ylabel('$g_{a\\gamma\\gamma}$ [GeV$^{-1}$]', fontsize=12)
            ax.set_yscale('log')
            ax.set_title('DAMSA Sensitivity to ALPs', fontsize=14)
            
            plt.colorbar(pcm, ax=ax, label='Expected Events')
            
            plt.tight_layout()
            
            if save:
                png_path = self.output_dir / "sensitivity_contour.png"
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
            
            return fig
        
        return None
    
    def generate_full_report(
        self,
        flux_file: str,
        alp_mass_MeV: float = 100.0,
        coupling: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Generate full signal visualization report.
        
        Parameters
        ----------
        flux_file : str
            Path to photon flux file
        alp_mass_MeV : float
            ALP mass
        coupling : float
            ALP coupling
            
        Returns
        -------
        dict
            Report summary
        """
        print(f"Generating ALP signal report...")
        print(f"  Mass: {alp_mass_MeV} MeV")
        print(f"  Coupling: {coupling}")
        print(f"  Output: {self.output_dir}")
        
        # Load flux
        flux_df = self.load_photon_flux(flux_file)
        print(f"  Loaded {len(flux_df)} photon flux entries")
        
        # Calculate signal
        result = self.calculate_alp_signal(flux_df, alp_mass_MeV, coupling)
        print(f"  Expected events: {result['n_events']}")
        
        # Generate plots
        plots_generated = []
        
        # Energy distribution
        try:
            self.plot_energy_distribution(result)
            plots_generated.append('energy_distribution')
        except Exception as e:
            print(f"  Failed to generate energy distribution: {e}")
        
        # Angular distribution
        try:
            self.plot_angular_distribution(result)
            plots_generated.append('angular_distribution')
        except Exception as e:
            print(f"  Failed to generate angular distribution: {e}")
        
        # Mass scan
        try:
            self.plot_mass_scan(flux_df, coupling=coupling)
            plots_generated.append('mass_scan')
        except Exception as e:
            print(f"  Failed to generate mass scan: {e}")
        
        # Coupling scan
        try:
            self.plot_coupling_scan(flux_df, alp_mass_MeV=alp_mass_MeV)
            plots_generated.append('coupling_scan')
        except Exception as e:
            print(f"  Failed to generate coupling scan: {e}")
        
        # Sensitivity contour
        try:
            self.plot_sensitivity_contour(flux_df)
            plots_generated.append('sensitivity_contour')
        except Exception as e:
            print(f"  Failed to generate sensitivity contour: {e}")
        
        # Save summary
        summary = {
            'flux_file': str(flux_file),
            'alp_mass_MeV': alp_mass_MeV,
            'coupling': coupling,
            'n_events': result['n_events'],
            'flux_entries': len(flux_df),
            'plots_generated': plots_generated,
            'output_dir': str(self.output_dir)
        }
        
        summary_path = self.output_dir / "signal_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nReport complete. Generated {len(plots_generated)} plots.")
        print(f"Summary saved: {summary_path}")
        
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ALP signal visualization plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with raw flux DataFrame
    python alplib_signal_plots.py -f output/photon_flux.csv -m 100
    
    # With proper beam current scaling (recommended for absolute rates)
    python alplib_signal_plots.py -f output/photon_flux.csv --nprimaries 1000 -m 100
    
    # Full report with all scans
    python alplib_signal_plots.py -f output/photon_flux.csv --nprimaries 1000 --full-report
        """
    )
    
    parser.add_argument('--flux-file', '-f', required=True,
                       help='Path to photon flux CSV file')
    parser.add_argument('--alp-mass', '-m', type=float, default=None,
                       help='ALP mass in MeV (default: auto-calculated proxy mass if --use-proxy, else 100)')
    parser.add_argument('--use-proxy', action='store_true',
                       help='Use proxy ALP mass to preserve opening angle (requires --photon-energy)')
    parser.add_argument('--photon-energy', type=float, default=80.0,
                       help='Typical photon energy from dump in MeV (default: 80, used with --use-proxy)')
    parser.add_argument('--coupling', '-g', type=float, default=1e-3,
                       help='ALP-photon coupling in GeV^-1 (default: 1e-3)')
    parser.add_argument('--output', '-o', default='alp_signal_plots',
                       help='Output directory (default: alp_signal_plots)')
    
    # Beam scaling parameters
    parser.add_argument('--nprimaries', '-n', type=int, default=None,
                       help='Number of primary electrons simulated (enables beam scaling)')
    parser.add_argument('--beam-current', type=float, default=62.5,
                       help='Beam current in μA (default: 62.5 for LCLS-II)')
    
    # Detector parameters
    parser.add_argument('--det-dist', type=float, default=1.0,
                       help='Target-detector distance in m (default: 1.0)')
    parser.add_argument('--exposure', type=float, default=30.0,
                       help='Exposure time in days (default: 30)')
    
    # Scan options
    parser.add_argument('--mass-scan', action='store_true',
                       help='Generate mass scan plot')
    parser.add_argument('--coupling-scan', action='store_true',
                       help='Generate coupling scan plot')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Generate 2D sensitivity contour')
    parser.add_argument('--full-report', action='store_true',
                       help='Generate full report with all plots')
    
    args = parser.parse_args()
    
    visualizer = ALPSignalVisualizer(args.output)
    visualizer.exposure_days = args.exposure
    
    # If nprimaries provided, use scaled flux loading
    if args.nprimaries is not None:
        print(f"Loading flux with beam scaling: {args.nprimaries} primaries, {args.beam_current} μA")
        flux_array = load_flux_for_alplib(args.flux_file, args.nprimaries, args.beam_current)
        print(f"  Energy range: {flux_array[:, 0].min():.1f} - {flux_array[:, 0].max():.1f} MeV")
        print(f"  Total rate: {flux_array[:, 1].sum():.3e} photons/second")
        
        # Convert to DataFrame format for visualizer compatibility
        flux_df = pd.DataFrame({
            'energy': flux_array[:, 0],
            'weight': flux_array[:, 1]
        })
    else:
        flux_df = visualizer.load_photon_flux(args.flux_file)
        print(f"Loaded {len(flux_df)} flux entries (no beam scaling)")
    
    # Handle proxy mass calculation
    # Default: Use proxy mass that preserves opening angle of 100 MeV ALP at 1 GeV
    if args.use_proxy:
        proxy_mass = calculate_proxy_mass(args.photon_energy, TARGET_ALP_MASS_MEV, TARGET_ALP_ENERGY_MEV)
        print(f"\n=== Proxy ALP Mass Calculation ===")
        print(f"Target scenario: m_a = {TARGET_ALP_MASS_MEV} MeV at E_a = {TARGET_ALP_ENERGY_MEV} MeV (γ = {TARGET_ALP_ENERGY_MEV/TARGET_ALP_MASS_MEV})")
        print(f"Photon energy from dump: {args.photon_energy} MeV")
        print(f"Proxy ALP mass: {proxy_mass:.2f} MeV")
        print(f"Opening angle: ~{2*proxy_mass/args.photon_energy*1000:.1f} mrad (same as target)")
        print(f"================================\n")
        alp_mass = proxy_mass
    elif args.alp_mass is None:
        # Default to 100 MeV if neither --use-proxy nor --alp-mass specified
        alp_mass = 100.0
        print(f"\nUsing default ALP mass: {alp_mass} MeV\n")
    else:
        alp_mass = args.alp_mass
    
    if args.full_report:
        visualizer.generate_full_report(
            args.flux_file,
            alp_mass,
            args.coupling
        )
    else:
        # Calculate signal
        result = visualizer.calculate_alp_signal(
            flux_df, alp_mass, args.coupling, args.det_dist
        )
        print(f"Expected events: {result['n_events']}")
        
        # Generate requested plots
        visualizer.plot_energy_distribution(result)
        
        if args.mass_scan:
            visualizer.plot_mass_scan(flux_df, coupling=args.coupling)
        
        if args.coupling_scan:
            visualizer.plot_coupling_scan(flux_df, alp_mass_MeV=alp_mass)
        
        if args.sensitivity:
            visualizer.plot_sensitivity_contour(flux_df)
    
    return 0


if __name__ == "__main__":
    exit(main())
