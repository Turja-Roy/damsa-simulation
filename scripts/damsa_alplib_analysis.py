#!/usr/bin/env python3
"""
DAMSA-alplib Integration Script

This script demonstrates how to use DAMSA Geant4 photon flux output
with alplib for ALP signal calculation.

Requirements:
    - alplib (pip install alplib)
    - numpy
    - matplotlib (for plotting)

Usage:
    python damsa_alplib_analysis.py output/photon_flux_target_exit.csv \
        --nprimaries 1000 --axion-mass 100 --axion-coupling 1e-4

Author: DAMSA Collaboration
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path so alplib can be imported as a module
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Check if alplib is available
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
    print(f"Warning: alplib not installed. Signal calculation will be skipped. ({e})")


def load_flux_for_alplib(csv_path: str, n_primaries: int, 
                         beam_current_uA: float = 62.5) -> np.ndarray:
    """
    Load Geant4 photon flux and convert to alplib format.
    
    Parameters
    ----------
    csv_path : str
        Path to photon flux CSV
    n_primaries : int
        Number of primary electrons simulated
    beam_current_uA : float
        Beam current in microamperes
        
    Returns
    -------
    np.ndarray
        2D array [[energy_MeV, rate_per_second], ...]
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Calculate scaling factor
    e_charge = 1.602176634e-19
    beam_current = beam_current_uA * 1e-6
    electrons_per_second = beam_current / e_charge
    scale_factor = electrons_per_second / n_primaries
    
    # Bin the photon energies (1 MeV bins)
    energies = df['energy_MeV'].values
    weights = df['weight'].values if 'weight' in df.columns else np.ones(len(energies))
    
    max_energy = np.ceil(energies.max())
    bins = np.arange(0, max_energy + 1, 1.0)
    counts, bin_edges = np.histogram(energies, bins=bins, weights=weights)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    rates = counts * scale_factor
    
    # Create flux array (only non-zero bins)
    flux = np.column_stack([bin_centers, rates])
    flux = flux[flux[:, 1] > 0]
    
    return flux


def calculate_alp_signal(photon_flux: np.ndarray,
                         axion_mass_MeV: float,
                         axion_coupling_MeV: float,
                         det_dist_m: float = 1.0,
                         det_length_m: float = 1.0,
                         det_area_m2: float = 0.25,
                         n_samples: int = 100000,
                         exposure_days: float = 30.0,
                         threshold_MeV: float = 10.0) -> dict:
    """
    Calculate expected ALP signal using alplib.
    
    Parameters
    ----------
    photon_flux : np.ndarray
        Photon flux array [[energy_MeV, rate_per_second], ...]
    axion_mass_MeV : float
        ALP mass in MeV
    axion_coupling_MeV : float
        ALP-photon coupling g_aγγ in MeV^-1
    det_dist_m : float
        Distance from target to detector in meters
    det_length_m : float
        Detector length in meters
    det_area_m2 : float
        Detector area in m^2
    n_samples : int
        Number of Monte Carlo samples
    exposure_days : float
        Exposure time in days
    threshold_MeV : float
        Energy threshold for detection in MeV
        
    Returns
    -------
    dict
        Dictionary with signal calculation results
    """
    if not ALPLIB_AVAILABLE:
        return {'error': 'alplib not available'}
    
    # Create Primakoff flux object
    flux = FluxPrimakoffIsotropic(
        photon_flux=photon_flux,
        target=Material("W"),  # Tungsten
        det_dist=det_dist_m,
        det_length=det_length_m,
        det_area=det_area_m2,
        axion_mass=axion_mass_MeV,
        axion_coupling=axion_coupling_MeV,
        n_samples=n_samples
    )
    
    # Simulate ALP production and propagation
    flux.simulate()
    flux.propagate()
    
    # Create event generator
    generator = PhotonEventGenerator(flux)
    
    # Calculate expected decays
    decays = generator.decays(
        exposure_days=exposure_days,
        threshold_MeV=threshold_MeV
    )
    
    # Gather results
    results = {
        'axion_mass_MeV': axion_mass_MeV,
        'axion_coupling_MeV': axion_coupling_MeV,
        'det_dist_m': det_dist_m,
        'det_length_m': det_length_m,
        'det_area_m2': det_area_m2,
        'exposure_days': exposure_days,
        'threshold_MeV': threshold_MeV,
        'n_samples': n_samples,
        'expected_decays': decays,
        'decay_rate_per_day': decays / exposure_days if exposure_days > 0 else 0,
    }
    
    return results


def scan_coupling(photon_flux: np.ndarray,
                  axion_mass_MeV: float,
                  coupling_range: tuple = (1e-6, 1e-2),
                  n_points: int = 20,
                  **kwargs) -> dict:
    """
    Scan over ALP-photon coupling values.
    
    Parameters
    ----------
    photon_flux : np.ndarray
        Photon flux array
    axion_mass_MeV : float
        ALP mass in MeV
    coupling_range : tuple
        (min, max) coupling values in MeV^-1
    n_points : int
        Number of coupling values to scan
    **kwargs : dict
        Additional arguments passed to calculate_alp_signal
        
    Returns
    -------
    dict
        Dictionary with scan results
    """
    if not ALPLIB_AVAILABLE:
        return {'error': 'alplib not available'}
    
    couplings = np.logspace(np.log10(coupling_range[0]), 
                           np.log10(coupling_range[1]), 
                           n_points)
    
    decays = []
    for g in couplings:
        result = calculate_alp_signal(photon_flux, axion_mass_MeV, g, **kwargs)
        decays.append(result['expected_decays'])
        print(f"  g = {g:.2e} MeV^-1: {result['expected_decays']:.2e} decays")
    
    return {
        'couplings': couplings,
        'decays': np.array(decays),
        'axion_mass_MeV': axion_mass_MeV,
    }


def scan_mass(photon_flux: np.ndarray,
              axion_coupling_MeV: float,
              mass_range: tuple = (1, 1000),
              n_points: int = 20,
              **kwargs) -> dict:
    """
    Scan over ALP mass values.
    
    Parameters
    ----------
    photon_flux : np.ndarray
        Photon flux array
    axion_coupling_MeV : float
        ALP-photon coupling in MeV^-1
    mass_range : tuple
        (min, max) mass values in MeV
    n_points : int
        Number of mass values to scan
    **kwargs : dict
        Additional arguments passed to calculate_alp_signal
        
    Returns
    -------
    dict
        Dictionary with scan results
    """
    if not ALPLIB_AVAILABLE:
        return {'error': 'alplib not available'}
    
    masses = np.logspace(np.log10(mass_range[0]), 
                        np.log10(mass_range[1]), 
                        n_points)
    
    decays = []
    for m in masses:
        result = calculate_alp_signal(photon_flux, m, axion_coupling_MeV, **kwargs)
        decays.append(result['expected_decays'])
        print(f"  m = {m:.1f} MeV: {result['expected_decays']:.2e} decays")
    
    return {
        'masses': masses,
        'decays': np.array(decays),
        'axion_coupling_MeV': axion_coupling_MeV,
    }


def plot_flux(flux: np.ndarray, output_path: str = None):
    """Plot the photon flux spectrum."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(flux[:, 0], flux[:, 1], width=0.8, alpha=0.7, label='Photon flux')
    ax.set_xlabel('Energy (MeV)', fontsize=12)
    ax.set_ylabel('Rate (photons/second)', fontsize=12)
    ax.set_title('DAMSA Photon Flux at Target Exit', fontsize=14)
    ax.set_yscale('log')
    ax.set_xlim(0, flux[:, 0].max() * 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved flux plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_coupling_scan(scan_result: dict, output_path: str = None):
    """Plot coupling scan results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(scan_result['couplings'], scan_result['decays'], 'o-', lw=2)
    ax.axhline(y=1, color='r', linestyle='--', label='1 event')
    ax.axhline(y=3, color='orange', linestyle='--', label='3 events')
    
    ax.set_xlabel(r'$g_{a\gamma\gamma}$ (MeV$^{-1}$)', fontsize=12)
    ax.set_ylabel('Expected Decays', fontsize=12)
    ax.set_title(f'ALP Signal vs Coupling (m_a = {scan_result["axion_mass_MeV"]:.1f} MeV)', 
                fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved coupling scan plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate ALP signal from DAMSA simulation using alplib',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', type=str, help='Input photon flux CSV')
    parser.add_argument('--nprimaries', '-n', type=int, required=True,
                        help='Number of primary electrons simulated')
    parser.add_argument('--beam-current', type=float, default=62.5,
                        help='Beam current in μA (default: 62.5)')
    parser.add_argument('--axion-mass', type=float, default=100.0,
                        help='ALP mass in MeV (default: 100)')
    parser.add_argument('--axion-coupling', type=float, default=1e-4,
                        help='ALP coupling in MeV^-1 (default: 1e-4)')
    parser.add_argument('--det-dist', type=float, default=1.0,
                        help='Target-detector distance in m (default: 1.0)')
    parser.add_argument('--det-length', type=float, default=1.0,
                        help='Detector length in m (default: 1.0)')
    parser.add_argument('--det-area', type=float, default=0.25,
                        help='Detector area in m^2 (default: 0.25 = 50x50 cm)')
    parser.add_argument('--exposure', type=float, default=30.0,
                        help='Exposure time in days (default: 30)')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Energy threshold in MeV (default: 10)')
    parser.add_argument('--scan-coupling', action='store_true',
                        help='Perform coupling scan')
    parser.add_argument('--scan-mass', action='store_true',
                        help='Perform mass scan')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load photon flux
    print(f"\nLoading photon flux from: {args.input}")
    flux = load_flux_for_alplib(args.input, args.nprimaries, args.beam_current)
    print(f"Loaded flux with {len(flux)} energy bins")
    print(f"Energy range: {flux[:, 0].min():.1f} - {flux[:, 0].max():.1f} MeV")
    print(f"Total rate: {flux[:, 1].sum():.3e} photons/second")
    
    # Plot flux
    plot_flux(flux, str(output_dir / 'photon_flux.png'))
    
    if not ALPLIB_AVAILABLE:
        print("\nalplib not available - skipping signal calculation")
        print("Install with: pip install alplib")
        return
    
    # Calculate single point signal
    print(f"\n=== ALP Signal Calculation ===")
    print(f"ALP mass: {args.axion_mass} MeV")
    print(f"ALP coupling: {args.axion_coupling} MeV^-1")
    print(f"Detector distance: {args.det_dist} m")
    print(f"Detector length: {args.det_length} m")
    print(f"Detector area: {args.det_area} m^2")
    print(f"Exposure: {args.exposure} days")
    print(f"Threshold: {args.threshold} MeV")
    
    result = calculate_alp_signal(
        flux,
        axion_mass_MeV=args.axion_mass,
        axion_coupling_MeV=args.axion_coupling,
        det_dist_m=args.det_dist,
        det_length_m=args.det_length,
        det_area_m2=args.det_area,
        exposure_days=args.exposure,
        threshold_MeV=args.threshold
    )
    
    print(f"\nExpected decays: {result['expected_decays']:.3e}")
    print(f"Rate: {result['decay_rate_per_day']:.3e} per day")
    
    # Coupling scan
    if args.scan_coupling:
        print("\n=== Coupling Scan ===")
        scan = scan_coupling(
            flux,
            axion_mass_MeV=args.axion_mass,
            det_dist_m=args.det_dist,
            det_length_m=args.det_length,
            det_area_m2=args.det_area,
            exposure_days=args.exposure,
            threshold_MeV=args.threshold,
            n_samples=10000  # Fewer samples for speed
        )
        plot_coupling_scan(scan, str(output_dir / 'coupling_scan.png'))
    
    # Mass scan
    if args.scan_mass:
        print("\n=== Mass Scan ===")
        scan = scan_mass(
            flux,
            axion_coupling_MeV=args.axion_coupling,
            det_dist_m=args.det_dist,
            det_length_m=args.det_length,
            det_area_m2=args.det_area,
            exposure_days=args.exposure,
            threshold_MeV=args.threshold,
            n_samples=10000
        )
        # Save mass scan results
        np.savez(str(output_dir / 'mass_scan.npz'), **scan)
        print(f"Saved mass scan to: {output_dir / 'mass_scan.npz'}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
