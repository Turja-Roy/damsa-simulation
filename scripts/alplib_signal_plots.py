#!/usr/bin/env python3
"""
ALP Signal Visualization Module for DAMSA Optimization

This module generates visualization plots for ALP (Axion-Like Particle) signal
calculations using alplib. It creates:
1. Energy distribution of ALP decay photons
2. Angular distribution of ALP decays
3. Signal rate vs ALP mass
4. Expected event counts vs coupling strength
5. Comparison between different detector configurations

REQUIRES alplib to be installed/available. No mock calculations.

Usage:
    python alplib_signal_plots.py --flux-file photon_flux.csv --alp-mass 100
    python alplib_signal_plots.py --results-dir optimization_results/

Author: DAMSA Collaboration
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


def check_alplib_available():
    """Check if alplib is available and raise error if not."""
    if not ALPLIB_AVAILABLE:
        raise ImportError(
            "alplib is required for ALP signal calculations.\n"
            "Clone from: https://github.com/athompson-git/alplib\n"
            "Then add to PYTHONPATH or place in project root."
        )


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
        required_cols = ['energy']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def calculate_alp_signal(
        self,
        flux_df: pd.DataFrame,
        alp_mass_MeV: float = 100.0,
        coupling: float = 1e-4,
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
            ALP-photon coupling g_aγγ
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
            flux = alplib_fluxes.FluxPrimakoffIsotropic(
                photon_energies_MeV=energies,
                target_z=74,  # Tungsten
                target_a=183.84
            )
            
            # Set ALP parameters
            flux.set_alp_mass(alp_mass_MeV)
            flux.set_coupling(coupling)
            
            # Calculate expected events using PhotonEventGenerator
            detector_material = alplib_materials.Material('CsI')
            generator = alplib_generators.PhotonEventGenerator(
                flux,
                detector_material
            )
            
            # Get decay events
            events = generator.decays(
                days_exposure=self.exposure_days,
                threshold=0.1  # MeV threshold
            )
            
            # Extract event properties
            if hasattr(events, '__len__'):
                n_events = len(events)
                if n_events > 0 and hasattr(events[0], 'energy'):
                    decay_energies = np.array([e.energy for e in events])
                    decay_angles = np.array([e.theta for e in events]) if hasattr(events[0], 'theta') else None
                else:
                    decay_energies = np.array([])
                    decay_angles = None
            else:
                n_events = int(events) if events else 0
                decay_energies = np.array([])
                decay_angles = None
            
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
        coupling: float = 1e-4,
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
        coupling: float = 1e-4
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
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--flux-file', '-f', required=True,
                       help='Path to photon flux CSV file')
    parser.add_argument('--alp-mass', '-m', type=float, default=100.0,
                       help='ALP mass in MeV (default: 100)')
    parser.add_argument('--coupling', '-g', type=float, default=1e-4,
                       help='ALP-photon coupling (default: 1e-4)')
    parser.add_argument('--output', '-o', default='alp_signal_plots',
                       help='Output directory (default: alp_signal_plots)')
    
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
    
    if args.full_report:
        visualizer.generate_full_report(
            args.flux_file,
            args.alp_mass,
            args.coupling
        )
    else:
        # Load flux
        flux_df = visualizer.load_photon_flux(args.flux_file)
        print(f"Loaded {len(flux_df)} flux entries")
        
        # Calculate signal
        result = visualizer.calculate_alp_signal(
            flux_df, args.alp_mass, args.coupling
        )
        print(f"Expected events: {result['n_events']}")
        
        # Generate requested plots
        visualizer.plot_energy_distribution(result)
        
        if args.mass_scan:
            visualizer.plot_mass_scan(flux_df, coupling=args.coupling)
        
        if args.coupling_scan:
            visualizer.plot_coupling_scan(flux_df, alp_mass_MeV=args.alp_mass)
        
        if args.sensitivity:
            visualizer.plot_sensitivity_contour(flux_df)
    
    return 0


if __name__ == "__main__":
    exit(main())
