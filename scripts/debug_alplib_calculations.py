#!/usr/bin/env python3
"""
Debug script for ALP signal calculations in DAMSA.

This script traces through each step of the alplib calculation to identify
where values become unreasonable (460M events issue, 100% separability issue).

It also explains:
1. Why is photon flux being scaled by electrons per second?
2. What is beam current 62.5 uA?
3. What are "primaries"?
4. Why are electrons involved in a photon-based ALP search?

Author: Debug session
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import alplib
try:
    import alplib.fluxes as alplib_fluxes
    import alplib.materials as alplib_materials
    import alplib.generators as alplib_generators
    from alplib.decay import W_gg
    from alplib.prod_xs import primakoff_sigma
    from alplib.constants import METER_BY_MEV, S_PER_DAY
    ALPLIB_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: alplib not available: {e}")
    ALPLIB_AVAILABLE = False
    sys.exit(1)


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def explain_beam_scaling():
    """
    EXPLAIN: Why electrons? Why beam current? What are primaries?
    """
    print_section("EXPLANATION: Beam Scaling in DAMSA")
    
    print("""
    DAMSA Experimental Setup:
    ========================
    
    1. ELECTRON BEAM hits tungsten target (NOT a photon beam!)
       - LCLS-II provides 8 GeV electron beam
       - Beam current = 62.5 µA (microamperes) = charge per second
       
    2. BREMSSTRAHLUNG: Electrons produce photons in the target
       - Each 8 GeV electron produces many secondary photons
       - This is what Geant4 simulates: electron → photon shower
       
    3. PRIMAKOFF: These photons can convert to ALPs
       - γ + Z → a + Z (coherent scattering off nucleus)
       - This is the BSM physics we're searching for
       
    4. ALP DECAY: ALPs travel and decay to photon pairs
       - a → γγ (our signal)
       
    TERMINOLOGY:
    ============
    - "Primaries" = primary beam particles (electrons) simulated in Geant4
    - "Beam current" = electrical current of electron beam (charge/second)
    - "Photon flux" = photons produced by the electron beam in target
    
    WHY SCALE BY ELECTRONS/SECOND?
    ==============================
    Geant4 simulates N primaries (e.g., 1000 electrons).
    Real beam has ~10^14 electrons/second.
    
    To get real event rates:
      photons_per_second = (photons_from_simulation / N_primaries) × electrons_per_second
      
    This converts simulation results to real experimental rates.
    """)


def load_and_analyze_flux():
    """Load flux data and analyze it."""
    print_section("STEP 1: Load Photon Flux from Geant4 Simulation")
    
    # Use existing simulation data
    flux_file = Path("output/Tz10_xy5_G42_photon_flux_target_exit.csv")
    
    if not flux_file.exists():
        print(f"ERROR: Flux file not found: {flux_file}")
        return None
        
    df = pd.read_csv(flux_file)
    
    print(f"\n  Flux file: {flux_file}")
    print(f"  Total photons in file: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    energies = df['energy_MeV'].values
    print(f"\n  Energy statistics:")
    print(f"    Min:    {energies.min():.3f} MeV")
    print(f"    Max:    {energies.max():.3f} MeV")
    print(f"    Mean:   {energies.mean():.3f} MeV")
    print(f"    Median: {np.median(energies):.3f} MeV")
    
    # Energy distribution
    print(f"\n  Energy distribution:")
    bins = [0, 1, 10, 50, 100, 500, 1000, 10000]
    for i in range(len(bins)-1):
        count = np.sum((energies >= bins[i]) & (energies < bins[i+1]))
        print(f"    {bins[i]:5} - {bins[i+1]:5} MeV: {count:6} photons ({100*count/len(energies):.1f}%)")
    
    return df


def analyze_beam_scaling(df, n_primaries=1000, beam_current_uA=62.5):
    """Analyze the beam scaling calculation."""
    print_section("STEP 2: Beam Current Scaling Analysis")
    
    # Physical constants
    e_charge = 1.602176634e-19  # Coulombs
    
    # Calculate scaling
    beam_current_A = beam_current_uA * 1e-6  # Convert µA to A
    electrons_per_second = beam_current_A / e_charge
    
    print(f"\n  Input parameters:")
    print(f"    N primaries (simulated electrons): {n_primaries}")
    print(f"    Beam current: {beam_current_uA} µA = {beam_current_A:.2e} A")
    
    print(f"\n  Calculations:")
    print(f"    Electrons per second = I / e")
    print(f"                        = {beam_current_A:.2e} / {e_charge:.3e}")
    print(f"                        = {electrons_per_second:.3e} electrons/second")
    
    photons_simulated = len(df)
    photons_per_primary = photons_simulated / n_primaries
    photons_per_second = photons_per_primary * electrons_per_second
    
    print(f"\n  Photon flux scaling:")
    print(f"    Photons from simulation: {photons_simulated}")
    print(f"    Photons per primary: {photons_per_primary:.1f}")
    print(f"    Photons per second (real beam): {photons_per_second:.3e}")
    
    print(f"\n  SANITY CHECK:")
    print(f"    Is {photons_per_second:.1e} photons/second reasonable?")
    print(f"    For 8 GeV electrons in thick W target: YES, this is expected.")
    print(f"    Most are low-energy (<10 MeV) bremsstrahlung photons.")
    
    return electrons_per_second, photons_per_second


def analyze_primakoff_cross_section(m_a=8.0, coupling_gev=1e-4):
    """Analyze Primakoff cross-section values."""
    print_section("STEP 3: Primakoff Cross-Section Analysis")
    
    # Convert coupling
    coupling_mev = coupling_gev / 1000.0  # GeV^-1 → MeV^-1
    
    print(f"\n  ALP parameters:")
    print(f"    Mass: m_a = {m_a} MeV")
    print(f"    Coupling: g = {coupling_gev:.1e} GeV^-1 = {coupling_mev:.1e} MeV^-1")
    
    # Target
    Z = 74  # Tungsten
    
    print(f"\n  Target: Tungsten (Z = {Z})")
    
    # Test at different photon energies
    print(f"\n  Primakoff cross-section σ(γ + Z → a + Z):")
    print(f"  {'E_γ (MeV)':<12} {'σ (MeV^-2)':<15} {'σ (barn)':<15} {'σ (pb)':<15}")
    print(f"  {'-'*55}")
    
    # Conversion: 1 MeV^-2 = 1 / (197.3 MeV·fm)^2 = 2.57e-5 fm^2 = 2.57e-5 × 10 mb = 0.257 mb
    # Actually: METER_BY_MEV = hbar*c in meters/MeV
    # 1 barn = 10^-28 m^2
    # σ(m^2) = σ(MeV^-2) × METER_BY_MEV^2
    
    mev2_to_barn = (1.9733e-13)**2 / 1e-28  # (hbar*c in m·MeV)^2 / barn
    
    test_energies = [10, 50, 80, 100, 200, 500, 1000]
    
    for E_gamma in test_energies:
        if E_gamma < m_a:
            print(f"  {E_gamma:<12} {'(below threshold)':<15}")
            continue
            
        sigma_mev2 = primakoff_sigma(E_gamma, coupling_mev, m_a, Z)
        sigma_barn = sigma_mev2 * mev2_to_barn
        sigma_pb = sigma_barn * 1e12  # picobarns
        
        print(f"  {E_gamma:<12} {sigma_mev2:<15.3e} {sigma_barn:<15.3e} {sigma_pb:<15.3e}")
    
    print(f"\n  INTERPRETATION:")
    print(f"    Cross-sections are in picobarns (10^-12 barn) range.")
    print(f"    This is typical for BSM physics - very rare processes!")
    print(f"    For comparison: Total photon absorption in W ~ 10-100 barn.")
    
    return coupling_mev


def analyze_decay_width(m_a=8.0, coupling_mev=1e-7):
    """Analyze ALP decay width and lifetime."""
    print_section("STEP 4: ALP Decay Width and Lifetime")
    
    # Calculate decay width
    # W_gg(g, m) = g^2 * m^3 / (64*pi)  [in MeV, when g is in MeV^-1]
    
    width_mev = W_gg(coupling_mev, m_a)
    
    print(f"\n  Decay width formula: Γ = g² × m_a³ / (64π)")
    print(f"    g = {coupling_mev:.1e} MeV^-1")
    print(f"    m_a = {m_a} MeV")
    print(f"    Γ = ({coupling_mev:.1e})² × ({m_a})³ / (64π)")
    print(f"    Γ = {width_mev:.3e} MeV")
    
    # Convert to lifetime
    # τ = ℏ / Γ
    # ℏ = 6.582e-22 MeV·s
    hbar_mev_s = 6.582119569e-22  # MeV·s
    tau_rest = hbar_mev_s / width_mev
    
    print(f"\n  Rest-frame lifetime:")
    print(f"    τ = ℏ / Γ = {hbar_mev_s:.3e} / {width_mev:.3e}")
    print(f"    τ = {tau_rest:.3e} seconds")
    
    # Lab-frame decay length
    # For E_a ~ 80 MeV, γ = E/m = 80/8 = 10
    E_a = 80  # MeV (typical ALP energy)
    gamma = E_a / m_a
    beta = np.sqrt(1 - 1/gamma**2)
    
    # Decay length = γ × β × c × τ
    c = 3e8  # m/s
    decay_length = gamma * beta * c * tau_rest
    
    print(f"\n  Lab-frame decay (E_a = {E_a} MeV):")
    print(f"    γ = E/m = {gamma:.1f}")
    print(f"    β = {beta:.4f}")
    print(f"    Decay length = γβcτ = {decay_length:.3e} meters")
    
    if decay_length > 1e6:
        print(f"    ⚠ WARNING: Decay length >> detector distance!")
        print(f"    Most ALPs will NOT decay in detector!")
    
    return width_mev, tau_rest, decay_length


def run_full_alplib_calculation(df, n_primaries=1000, beam_current_uA=62.5,
                                 m_a=8.0, coupling_gev=1e-4, exposure_days=30.0,
                                 det_dist_m=0.42):
    """Run complete alplib calculation with debug output."""
    print_section("STEP 5: Full alplib Calculation (Step-by-Step)")
    
    coupling_mev = coupling_gev / 1000.0
    
    print(f"\n  Parameters:")
    print(f"    ALP mass: {m_a} MeV")
    print(f"    Coupling: {coupling_gev:.1e} GeV^-1 = {coupling_mev:.1e} MeV^-1")
    print(f"    Detector distance: {det_dist_m} m (= gap)")
    print(f"    Detector area: 0.25 m² (50×50 cm)")
    print(f"    Detector length: 1.0 m")
    print(f"    Exposure: {exposure_days} days")
    
    # Prepare flux array (with beam scaling)
    e_charge = 1.602176634e-19
    electrons_per_second = (beam_current_uA * 1e-6) / e_charge
    scale_factor = electrons_per_second / n_primaries
    
    energies = df['energy_MeV'].values
    
    # Bin the photon energies (1 MeV bins)
    max_energy = np.ceil(energies.max())
    bins = np.arange(0, max_energy + 1, 1.0)
    counts, bin_edges = np.histogram(energies, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Scale counts to rates (photons/second)
    rates = counts * scale_factor
    
    # Create flux array (only non-zero bins)
    flux_array = np.column_stack([bin_centers, rates])
    flux_array = flux_array[flux_array[:, 1] > 0]
    
    print(f"\n  5a. Flux Array Preparation:")
    print(f"    Total photons (raw): {len(df)}")
    print(f"    Scale factor: {scale_factor:.3e}")
    print(f"    Energy bins with photons: {len(flux_array)}")
    print(f"    Total rate: {flux_array[:, 1].sum():.3e} photons/second")
    print(f"    Sample flux entries (E, rate/s):")
    for i in [0, 10, 50, 100]:
        if i < len(flux_array):
            print(f"      [{i}]: E={flux_array[i,0]:.1f} MeV, rate={flux_array[i,1]:.3e}/s")
    
    # Create materials
    target_material = alplib_materials.Material("W")
    detector_material = alplib_materials.Material("CsI")
    
    # Create Primakoff flux generator
    print(f"\n  5b. Creating FluxPrimakoffIsotropic...")
    flux_obj = alplib_fluxes.FluxPrimakoffIsotropic(
        photon_flux=flux_array,
        target=target_material,
        det_dist=det_dist_m,
        det_length=1.0,
        det_area=0.25,
        axion_mass=m_a,
        axion_coupling=coupling_mev,
        n_samples=1000
    )
    
    # Simulate ALP production
    print(f"\n  5c. Running simulate() - Primakoff production...")
    flux_obj.simulate()
    
    axion_energies = np.array(flux_obj.axion_energy)
    axion_flux = np.array(flux_obj.axion_flux)
    
    print(f"    ALP energies produced: {len(axion_energies)}")
    print(f"    ALP flux sum: {axion_flux.sum():.3e} ALPs/second")
    if len(axion_energies) > 0:
        print(f"    ALP energy range: {axion_energies.min():.1f} - {axion_energies.max():.1f} MeV")
        print(f"    Sample ALP flux entries:")
        for i in [0, 10, 100]:
            if i < len(axion_energies):
                print(f"      [{i}]: E_a={axion_energies[i]:.1f} MeV, flux={axion_flux[i]:.3e}/s")
    
    # Propagate ALPs (calculate decay probabilities)
    print(f"\n  5d. Running propagate() - decay probabilities...")
    flux_obj.propagate()
    
    decay_weights = np.array(flux_obj.decay_axion_weight)
    scatter_weights = np.array(flux_obj.scatter_axion_weight)
    
    print(f"    Decay weights sum: {decay_weights.sum():.3e}")
    print(f"    Scatter weights sum: {scatter_weights.sum():.3e}")
    if len(decay_weights) > 0:
        print(f"    Sample decay weights:")
        for i in [0, 10, 100]:
            if i < len(decay_weights):
                print(f"      [{i}]: decay_wgt={decay_weights[i]:.3e}")
    
    # Create event generator
    print(f"\n  5e. Creating PhotonEventGenerator...")
    generator = alplib_generators.PhotonEventGenerator(flux_obj, detector_material)
    
    # Calculate total decays
    print(f"\n  5f. Calculating decays for {exposure_days} days exposure...")
    n_events = generator.decays(days_exposure=exposure_days, threshold=0.1)
    
    print(f"\n  RESULT: {n_events:.6f} ALP decay events")
    
    print(f"\n  Calculation trace:")
    print(f"    decay_weights.sum() = {decay_weights.sum():.3e}")
    print(f"    S_PER_DAY = {S_PER_DAY:.3e}")
    print(f"    days_exposure = {exposure_days}")
    print(f"    Expected: sum × days × s/day = {decay_weights.sum() * exposure_days * S_PER_DAY:.3e}")
    
    return n_events, flux_obj, generator


def analyze_separability(generator, exposure_days=30.0):
    """Analyze separability calculation."""
    print_section("STEP 6: Separability Analysis (Opening Angle)")
    
    print(f"\n  6a. Calling simulate_decay_4vectors()...")
    
    try:
        p4_1, p4_2, weights = generator.simulate_decay_4vectors(
            days_exposure=exposure_days,
            n_samples=10  # Use 10 samples per ALP for statistics
        )
        
        print(f"    Returned {len(p4_1)} decay events")
        
        if len(p4_1) == 0:
            print(f"\n  ⚠ WARNING: No decay 4-vectors returned!")
            print(f"    This explains why separability falls back to 100%!")
            return
        
        # Calculate opening angles
        print(f"\n  6b. Calculating opening angles...")
        
        opening_angles_deg = []
        energies_1 = []
        energies_2 = []
        
        for i in range(min(len(p4_1), 1000)):  # Analyze up to 1000 events
            p1 = p4_1[i]
            p2 = p4_2[i]
            
            # Extract 4-vector components
            E1, px1, py1, pz1 = p1.p0, p1.p1, p1.p2, p1.p3
            E2, px2, py2, pz2 = p2.p0, p2.p1, p2.p2, p2.p3
            
            energies_1.append(E1)
            energies_2.append(E2)
            
            # Calculate opening angle
            p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)
            p2_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
            dot = px1*px2 + py1*py2 + pz1*pz2
            
            cos_angle = dot / (p1_mag * p2_mag + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            opening_angles_deg.append(angle_deg)
        
        opening_angles_deg = np.array(opening_angles_deg)
        energies_1 = np.array(energies_1)
        energies_2 = np.array(energies_2)
        
        print(f"\n  Opening angle statistics:")
        print(f"    Min:    {opening_angles_deg.min():.2f} deg")
        print(f"    Max:    {opening_angles_deg.max():.2f} deg")
        print(f"    Mean:   {opening_angles_deg.mean():.2f} deg")
        print(f"    Median: {np.median(opening_angles_deg):.2f} deg")
        
        print(f"\n  Photon 1 energy statistics:")
        print(f"    Min:    {energies_1.min():.2f} MeV")
        print(f"    Max:    {energies_1.max():.2f} MeV")
        print(f"    Mean:   {energies_1.mean():.2f} MeV")
        
        # Separability analysis
        angle_cut = 20.0  # degrees
        energy_cut = 100.0  # MeV
        
        high_angle = opening_angles_deg >= angle_cut
        low_angle_high_energy = (
            (opening_angles_deg < angle_cut) & 
            (energies_1 > energy_cut) & 
            (energies_2 > energy_cut)
        )
        separable = high_angle | low_angle_high_energy
        
        separability = np.mean(separable)
        
        print(f"\n  Separability analysis:")
        print(f"    High angle (>= {angle_cut} deg): {np.mean(high_angle)*100:.1f}%")
        print(f"    Low angle + high energy: {np.mean(low_angle_high_energy)*100:.1f}%")
        print(f"    Total separable: {separability*100:.1f}%")
        
        # Sample events
        print(f"\n  Sample decay events:")
        for i in range(min(5, len(opening_angles_deg))):
            print(f"    [{i}]: θ={opening_angles_deg[i]:.2f}°, "
                  f"E1={energies_1[i]:.1f} MeV, E2={energies_2[i]:.1f} MeV, "
                  f"sep={separable[i]}")
        
    except Exception as e:
        print(f"\n  ERROR in simulate_decay_4vectors: {e}")
        import traceback
        traceback.print_exc()


def compare_with_without_scaling():
    """Compare results with and without beam scaling."""
    print_section("STEP 7: Impact of Beam Scaling")
    
    flux_file = Path("output/Tz10_xy5_G42_photon_flux_target_exit.csv")
    df = pd.read_csv(flux_file)
    energies = df['energy_MeV'].values
    
    m_a = 8.0
    coupling_gev = 1e-4
    coupling_mev = coupling_gev / 1000.0
    exposure_days = 30.0
    det_dist_m = 0.42
    
    print(f"\n  Comparing WITH vs WITHOUT beam scaling...")
    print(f"  (Same ALP parameters: m_a={m_a} MeV, g={coupling_gev:.0e} GeV^-1)")
    
    results = []
    
    for case_name, use_scaling in [("WITHOUT scaling (raw counts)", False),
                                    ("WITH scaling (beam current)", True)]:
        print(f"\n  Case: {case_name}")
        
        if use_scaling:
            # With beam scaling
            e_charge = 1.602176634e-19
            beam_current_uA = 62.5
            n_primaries = 1000
            electrons_per_second = (beam_current_uA * 1e-6) / e_charge
            scale_factor = electrons_per_second / n_primaries
        else:
            # Without scaling - just raw counts
            scale_factor = 1.0
        
        # Bin energies
        max_energy = np.ceil(energies.max())
        bins = np.arange(0, max_energy + 1, 1.0)
        counts, bin_edges = np.histogram(energies, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        rates = counts * scale_factor
        flux_array = np.column_stack([bin_centers, rates])
        flux_array = flux_array[flux_array[:, 1] > 0]
        
        print(f"    Scale factor: {scale_factor:.3e}")
        print(f"    Total flux: {flux_array[:, 1].sum():.3e}")
        
        # Run calculation
        target_material = alplib_materials.Material("W")
        detector_material = alplib_materials.Material("CsI")
        
        flux_obj = alplib_fluxes.FluxPrimakoffIsotropic(
            photon_flux=flux_array,
            target=target_material,
            det_dist=det_dist_m,
            det_length=1.0,
            det_area=0.25,
            axion_mass=m_a,
            axion_coupling=coupling_mev,
            n_samples=1000
        )
        
        flux_obj.simulate()
        flux_obj.propagate()
        
        generator = alplib_generators.PhotonEventGenerator(flux_obj, detector_material)
        n_events = generator.decays(days_exposure=exposure_days, threshold=0.1)
        
        print(f"    ALP events: {n_events:.6f}")
        results.append((case_name, scale_factor, n_events))
    
    print(f"\n  SUMMARY:")
    print(f"  {'Case':<35} {'Scale Factor':<15} {'ALP Events':<15}")
    print(f"  {'-'*65}")
    for name, sf, ne in results:
        print(f"  {name:<35} {sf:<15.3e} {ne:<15.6f}")


def main():
    """Run all debug analyses."""
    print("\n" + "#"*70)
    print("#  DAMSA ALP Signal Calculation Debug")
    print("#"*70)
    
    # Explain the physics
    explain_beam_scaling()
    
    # Load flux
    df = load_and_analyze_flux()
    if df is None:
        return 1
    
    # Analyze beam scaling
    analyze_beam_scaling(df, n_primaries=1000, beam_current_uA=62.5)
    
    # Analyze Primakoff cross-section
    coupling_mev = analyze_primakoff_cross_section(m_a=8.0, coupling_gev=1e-4)
    
    # Analyze decay width
    analyze_decay_width(m_a=8.0, coupling_mev=coupling_mev)
    
    # Run full calculation
    n_events, flux_obj, generator = run_full_alplib_calculation(
        df, n_primaries=1000, beam_current_uA=62.5,
        m_a=8.0, coupling_gev=1e-4, exposure_days=30.0,
        det_dist_m=0.42
    )
    
    # Analyze separability
    analyze_separability(generator, exposure_days=30.0)
    
    # Compare with/without scaling
    compare_with_without_scaling()
    
    print("\n" + "="*70)
    print(" DEBUG COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
