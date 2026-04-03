#!/usr/bin/env python3
"""
DAMSA Gap Scan Results Analysis
Reads gap scan results and creates visualization plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

def read_gap_scan_results(filename='gap_scan_results.txt'):
    """Read the gap scan results file"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        print("Please run the gap scan simulation first.")
        sys.exit(1)
    
    # Read all lines, skip header lines and comments
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            # Parse data line
            parts = line.split()
            if len(parts) >= 8:
                try:
                    row = [float(x) for x in parts[:8]]
                    data.append(row)
                except ValueError:
                    continue
    
    if not data:
        print("Error: No data found in file!")
        sys.exit(1)
    
    data = np.array(data)
    
    results = {
        'gap': data[:, 0],
        'csi_z': data[:, 1],
        'target_photons': data[:, 2],
        'target_neutrons': data[:, 3],
        'detector_photons': data[:, 4],
        'detector_neutrons': data[:, 5],
        'forward_photons': data[:, 6],
        'efficiency': data[:, 7]
    }
    
    return results

def plot_gap_scan_results(results, output_prefix='gap_scan'):
    """Create comprehensive visualization of gap scan results"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    optimal_idx = np.argmax(results['efficiency'])
    optimal_gap = results['gap'][optimal_idx]
    optimal_eff = results['efficiency'][optimal_idx]
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['gap'], results['efficiency'], 'o-', color='blue', 
             linewidth=2, markersize=8, label='Photon Efficiency')
    ax1.set_xlabel('Gap Distance (cm)', fontsize=12)
    ax1.set_ylabel('Efficiency (%)', fontsize=12)
    ax1.set_title('Photon Detection Efficiency vs Gap', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(optimal_gap, color='red', linestyle='--', alpha=0.5, 
                label=f'Optimal: {optimal_gap:.0f} cm')
    ax1.plot(optimal_gap, optimal_eff, 'r*', markersize=20)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['gap'], results['detector_photons'], 'o-', color='green', 
             linewidth=2, markersize=8)
    ax2.set_xlabel('Gap Distance (cm)', fontsize=12)
    ax2.set_ylabel('Number of Photons', fontsize=12)
    ax2.set_title('Photons Reaching Detector vs Gap', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(optimal_gap, color='red', linestyle='--', alpha=0.5)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(results['gap'], results['forward_photons'], 'o-', color='purple', 
             linewidth=2, markersize=8)
    ax3.set_xlabel('Gap Distance (cm)', fontsize=12)
    ax3.set_ylabel('Number of Forward Photons', fontsize=12)
    ax3.set_title('Forward Photons (0-20°) vs Gap', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(optimal_gap, color='red', linestyle='--', alpha=0.5)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(results['gap'], results['detector_neutrons'], 'o-', color='orange', 
             linewidth=2, markersize=8)
    ax4.set_xlabel('Gap Distance (cm)', fontsize=12)
    ax4.set_ylabel('Number of Neutrons', fontsize=12)
    ax4.set_title('Neutrons Reaching Detector vs Gap', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(optimal_gap, color='red', linestyle='--', alpha=0.5)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(results['gap'], results['target_photons'], 'o-', color='blue', 
             linewidth=2, markersize=8, label='Target Exit Photons', alpha=0.7)
    ax5.plot(results['gap'], results['detector_photons'], 's-', color='green', 
             linewidth=2, markersize=8, label='Detector Photons', alpha=0.7)
    ax5.set_xlabel('Gap Distance (cm)', fontsize=12)
    ax5.set_ylabel('Number of Photons', fontsize=12)
    ax5.set_title('Target Exit vs Detector Photons', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.axvline(optimal_gap, color='red', linestyle='--', alpha=0.5)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    table_data = []
    table_data.append(['Parameter', 'Value'])
    table_data.append(['Optimal Gap', f'{optimal_gap:.0f} cm'])
    table_data.append(['Optimal Efficiency', f'{optimal_eff:.2f}%'])
    table_data.append(['CsI Position', f'{results["csi_z"][optimal_idx]:.1f} cm'])
    table_data.append(['Detector Photons', f'{int(results["detector_photons"][optimal_idx])}'])
    table_data.append(['Forward Photons', f'{int(results["forward_photons"][optimal_idx])}'])
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Optimal Configuration', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('DAMSA Gap Distance Optimization Scan Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved as: {output_prefix}_analysis.png")
    
    return optimal_gap, optimal_eff

def print_recommendations(results):
    """Print detailed recommendations based on the scan"""
    
    optimal_idx = np.argmax(results['efficiency'])
    optimal_gap = results['gap'][optimal_idx]
    
    print("\n" + "="*60)
    print("DAMSA GAP SCAN RECOMMENDATIONS")
    print("="*60)
    print(f"\nOptimal Gap Distance: {optimal_gap:.0f} cm")
    print(f"Corresponding CsI Center Z: {results['csi_z'][optimal_idx]:.1f} cm")
    print(f"Maximum Efficiency: {results['efficiency'][optimal_idx]:.2f}%")
    print(f"Detector Photons at Optimal: {int(results['detector_photons'][optimal_idx])}")
    print(f"Forward Photons at Optimal: {int(results['forward_photons'][optimal_idx])}")
    
    print("\n--- Top 3 Configurations ---")
    sorted_indices = np.argsort(results['efficiency'])[::-1][:3]
    for i, idx in enumerate(sorted_indices, 1):
        print(f"{i}. Gap={results['gap'][idx]:.0f} cm, "
              f"Efficiency={results['efficiency'][idx]:.2f}%, "
              f"Photons={int(results['detector_photons'][idx])}")
    
    print("\n--- Gap Range Analysis ---")
    print(f"Minimum Efficiency: {np.min(results['efficiency']):.2f}% "
          f"at {results['gap'][np.argmin(results['efficiency'])]:.0f} cm")
    print(f"Maximum Efficiency: {np.max(results['efficiency']):.2f}% "
          f"at {results['gap'][np.argmax(results['efficiency'])]:.0f} cm")
    print(f"Average Efficiency: {np.mean(results['efficiency']):.2f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'gap_scan_results.txt'
    
    try:
        results = read_gap_scan_results(filename)
        
        output_prefix = filename.replace('.txt', '')
        optimal_gap, optimal_eff = plot_gap_scan_results(results, output_prefix)
        
        print_recommendations(results)
        
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        print("Please run the gap scan simulation first.")
    except Exception as e:
        print(f"Error: {e}")
