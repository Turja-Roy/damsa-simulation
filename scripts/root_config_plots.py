#!/usr/bin/env python3
"""
ROOT-Based Configuration Comparison Plots for DAMSA Optimization

This module generates ROOT-format summary plots for comparing different
detector configurations. It creates histograms and plots for:
1. Photon energy distributions at different scoring planes
2. Angular distributions
3. Timing distributions
4. Particle spectra (photons, neutrons, electrons)
5. Configuration comparison overlays

Outputs both ROOT files (.root) and PNG images for quick viewing.

Usage:
    python root_config_plots.py --config-file config.json
    python root_config_plots.py --results-dir optimization_results/
    python root_config_plots.py --compare config1.json config2.json config3.json

Author: DAMSA Collaboration
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import argparse

# ROOT imports
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)  # Don't display canvases
    ROOT_AVAILABLE = True
except ImportError:
    ROOT_AVAILABLE = False
    print("Warning: ROOT not available. Install PyROOT for ROOT output.")

# Matplotlib fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ConfigurationPlotter:
    """
    Generate ROOT plots for a single detector configuration.
    
    Reads simulation output and creates comprehensive histograms
    for physics distributions.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "plots"):
        """
        Initialize plotter.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with:
            - target_z, target_xy, gap: geometry parameters
            - data_file: path to simulation output (CSV or ROOT)
        output_dir : str
            Directory for output plots
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration label
        self.label = self._make_label()
        
        # Histograms storage
        self.histograms = {}
        
    def _make_label(self) -> str:
        """Create descriptive label for this configuration."""
        tz = self.config.get('target_z', 10.0)
        txy = self.config.get('target_xy', 5.0)
        gap = self.config.get('gap', 47.0)
        return f"T{tz:.0f}x{txy:.0f}_G{gap:.0f}"
    
    def load_data(self, data_file: str) -> pd.DataFrame:
        """
        Load simulation data from file.
        
        Parameters
        ----------
        data_file : str
            Path to data file (CSV or ROOT)
            
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        data_file = Path(data_file)
        
        if data_file.suffix == '.csv':
            return pd.read_csv(data_file)
        elif data_file.suffix == '.root':
            return self._load_root_file(data_file)
        else:
            raise ValueError(f"Unknown file format: {data_file.suffix}")
    
    def _load_root_file(self, root_file: Path) -> pd.DataFrame:
        """Load data from ROOT file into DataFrame."""
        if not ROOT_AVAILABLE:
            raise ImportError("ROOT required to read .root files")
        
        f = ROOT.TFile.Open(str(root_file), "READ")
        
        # Try different tree names
        tree_names = ["hits", "flux", "photons", "events"]
        tree = None
        
        for name in tree_names:
            tree = f.Get(name)
            if tree:
                break
        
        if not tree:
            f.Close()
            raise ValueError(f"No recognized tree in {root_file}")
        
        # Convert to DataFrame
        data = {}
        branches = [b.GetName() for b in tree.GetListOfBranches()]
        
        for branch in branches:
            values = []
            for i in range(tree.GetEntries()):
                tree.GetEntry(i)
                values.append(getattr(tree, branch))
            data[branch] = values
        
        f.Close()
        return pd.DataFrame(data)
    
    def create_histograms(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create ROOT histograms from data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Simulation data
            
        Returns
        -------
        dict
            Dictionary of ROOT histograms
        """
        if not ROOT_AVAILABLE:
            return self._create_matplotlib_histograms(df)
        
        histograms = {}
        label = self.label
        
        # Energy distribution
        if 'energy' in df.columns or 'E' in df.columns:
            e_col = 'energy' if 'energy' in df.columns else 'E'
            h_energy = ROOT.TH1D(f"h_energy_{label}", 
                                 f"Energy Distribution ({label});Energy [MeV];Counts",
                                 100, 0, 1000)
            for e in df[e_col]:
                h_energy.Fill(e)
            histograms['energy'] = h_energy
        
        # Log energy distribution
        if 'energy' in df.columns or 'E' in df.columns:
            e_col = 'energy' if 'energy' in df.columns else 'E'
            h_log_energy = ROOT.TH1D(f"h_log_energy_{label}",
                                     f"Log Energy Distribution ({label});log_{10}(E/MeV);Counts",
                                     100, -3, 4)
            for e in df[e_col]:
                if e > 0:
                    h_log_energy.Fill(np.log10(e))
            histograms['log_energy'] = h_log_energy
        
        # Angular distribution (theta)
        if 'theta' in df.columns:
            h_theta = ROOT.TH1D(f"h_theta_{label}",
                                f"Angular Distribution ({label});#theta [deg];Counts",
                                90, 0, 90)
            for theta in df['theta']:
                h_theta.Fill(theta)
            histograms['theta'] = h_theta
        elif 'px' in df.columns and 'py' in df.columns and 'pz' in df.columns:
            # Calculate theta from momentum
            h_theta = ROOT.TH1D(f"h_theta_{label}",
                                f"Angular Distribution ({label});#theta [deg];Counts",
                                90, 0, 90)
            for _, row in df.iterrows():
                p = np.sqrt(row['px']**2 + row['py']**2 + row['pz']**2)
                if p > 0:
                    theta = np.arccos(row['pz'] / p) * 180 / np.pi
                    h_theta.Fill(theta)
            histograms['theta'] = h_theta
        
        # Time distribution
        if 'time' in df.columns or 't' in df.columns:
            t_col = 'time' if 'time' in df.columns else 't'
            h_time = ROOT.TH1D(f"h_time_{label}",
                               f"Time Distribution ({label});Time [ns];Counts",
                               100, 0, 100)
            for t in df[t_col]:
                h_time.Fill(t)
            histograms['time'] = h_time
        
        # Position distributions
        for coord in ['x', 'y', 'z']:
            if coord in df.columns:
                h_pos = ROOT.TH1D(f"h_{coord}_{label}",
                                  f"{coord.upper()} Position ({label});{coord} [cm];Counts",
                                  100, -50, 50 if coord != 'z' else 200)
                for val in df[coord]:
                    h_pos.Fill(val)
                histograms[coord] = h_pos
        
        # XY position (2D)
        if 'x' in df.columns and 'y' in df.columns:
            h_xy = ROOT.TH2D(f"h_xy_{label}",
                             f"XY Position ({label});X [cm];Y [cm]",
                             50, -25, 25, 50, -25, 25)
            for _, row in df.iterrows():
                h_xy.Fill(row['x'], row['y'])
            histograms['xy'] = h_xy
        
        # Particle type distribution (if available)
        if 'pdg' in df.columns or 'particle' in df.columns:
            p_col = 'pdg' if 'pdg' in df.columns else 'particle'
            h_pdg = ROOT.TH1D(f"h_pdg_{label}",
                              f"Particle Types ({label});PDG Code;Counts",
                              100, -50, 50)
            for pdg in df[p_col]:
                h_pdg.Fill(pdg)
            histograms['pdg'] = h_pdg
        
        # Scoring plane distribution
        if 'plane' in df.columns or 'scoring_plane' in df.columns:
            plane_col = 'plane' if 'plane' in df.columns else 'scoring_plane'
            unique_planes = df[plane_col].unique()
            h_plane = ROOT.TH1D(f"h_plane_{label}",
                                f"Scoring Planes ({label});Plane;Counts",
                                len(unique_planes), 0, len(unique_planes))
            for plane in df[plane_col]:
                h_plane.Fill(str(plane), 1)
            histograms['plane'] = h_plane
        
        self.histograms = histograms
        return histograms
    
    def _create_matplotlib_histograms(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback: create matplotlib figures instead of ROOT histograms."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither ROOT nor matplotlib available")
        
        histograms = {}
        label = self.label
        
        # Energy distribution
        if 'energy' in df.columns or 'E' in df.columns:
            e_col = 'energy' if 'energy' in df.columns else 'E'
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df[e_col], bins=100, range=(0, 1000), alpha=0.7)
            ax.set_xlabel('Energy [MeV]')
            ax.set_ylabel('Counts')
            ax.set_title(f'Energy Distribution ({label})')
            histograms['energy'] = fig
        
        return histograms
    
    def save_histograms(self, root_filename: str = None) -> str:
        """
        Save histograms to ROOT file.
        
        Parameters
        ----------
        root_filename : str
            Output ROOT filename
            
        Returns
        -------
        str
            Path to saved file
        """
        if not ROOT_AVAILABLE:
            return self._save_matplotlib_plots()
        
        if root_filename is None:
            root_filename = self.output_dir / f"histograms_{self.label}.root"
        
        f = ROOT.TFile.Open(str(root_filename), "RECREATE")
        
        for name, hist in self.histograms.items():
            hist.Write()
        
        f.Close()
        print(f"Saved ROOT histograms: {root_filename}")
        return str(root_filename)
    
    def _save_matplotlib_plots(self) -> str:
        """Save matplotlib plots as PNGs."""
        for name, fig in self.histograms.items():
            if isinstance(fig, plt.Figure):
                path = self.output_dir / f"{name}_{self.label}.png"
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
        return str(self.output_dir)
    
    def create_summary_canvas(self, save_png: bool = True) -> Any:
        """
        Create summary canvas with all histograms.
        
        Parameters
        ----------
        save_png : bool
            Also save as PNG image
            
        Returns
        -------
        ROOT.TCanvas or matplotlib.Figure
            Summary canvas
        """
        if not ROOT_AVAILABLE:
            return self._create_matplotlib_summary()
        
        # Create canvas
        canvas = ROOT.TCanvas(f"c_summary_{self.label}",
                             f"Summary: {self.label}",
                             1200, 800)
        
        # Determine grid size
        n_hists = len(self.histograms)
        if n_hists <= 4:
            canvas.Divide(2, 2)
        elif n_hists <= 6:
            canvas.Divide(3, 2)
        else:
            canvas.Divide(3, 3)
        
        for i, (name, hist) in enumerate(self.histograms.items()):
            canvas.cd(i + 1)
            if isinstance(hist, ROOT.TH2):
                hist.Draw("COLZ")
            else:
                hist.Draw()
        
        canvas.Update()
        
        if save_png:
            png_path = self.output_dir / f"summary_{self.label}.png"
            canvas.SaveAs(str(png_path))
            print(f"Saved summary: {png_path}")
        
        return canvas
    
    def _create_matplotlib_summary(self) -> plt.Figure:
        """Create matplotlib summary figure."""
        n_hists = len(self.histograms)
        if n_hists == 0:
            return None
        
        n_cols = min(3, n_hists)
        n_rows = (n_hists + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = np.atleast_1d(axes).flatten()
        
        for ax in axes[n_hists:]:
            ax.set_visible(False)
        
        fig.suptitle(f"Summary: {self.label}")
        plt.tight_layout()
        
        png_path = self.output_dir / f"summary_{self.label}.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary: {png_path}")
        
        return fig


class ConfigurationComparator:
    """
    Compare multiple detector configurations side-by-side.
    """
    
    def __init__(self, configs: List[Dict[str, Any]], output_dir: str = "comparison_plots"):
        """
        Initialize comparator.
        
        Parameters
        ----------
        configs : list
            List of configuration dictionaries
        output_dir : str
            Output directory
        """
        self.configs = configs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plotters = []
        self.colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta,
                      ROOT.kOrange, ROOT.kCyan] if ROOT_AVAILABLE else \
                      ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    
    def load_all_data(self, data_files: List[str]) -> List[pd.DataFrame]:
        """Load data for all configurations."""
        dataframes = []
        
        for i, (config, data_file) in enumerate(zip(self.configs, data_files)):
            plotter = ConfigurationPlotter(config, str(self.output_dir / f"config_{i}"))
            df = plotter.load_data(data_file)
            plotter.create_histograms(df)
            self.plotters.append(plotter)
            dataframes.append(df)
        
        return dataframes
    
    def create_comparison_plots(self, histogram_names: List[str] = None) -> Dict[str, Any]:
        """
        Create overlay comparison plots.
        
        Parameters
        ----------
        histogram_names : list
            Names of histograms to compare (None = all)
            
        Returns
        -------
        dict
            Dictionary of comparison canvases
        """
        if not self.plotters:
            raise ValueError("No data loaded. Call load_all_data first.")
        
        if histogram_names is None:
            # Use histograms from first plotter
            histogram_names = list(self.plotters[0].histograms.keys())
        
        canvases = {}
        
        if not ROOT_AVAILABLE:
            return self._create_matplotlib_comparisons(histogram_names)
        
        for hist_name in histogram_names:
            # Check if all plotters have this histogram
            if not all(hist_name in p.histograms for p in self.plotters):
                continue
            
            # Skip 2D histograms
            if isinstance(self.plotters[0].histograms[hist_name], ROOT.TH2):
                continue
            
            canvas = ROOT.TCanvas(f"c_compare_{hist_name}",
                                 f"Comparison: {hist_name}",
                                 800, 600)
            
            legend = ROOT.TLegend(0.65, 0.7, 0.9, 0.9)
            
            max_y = 0
            for i, plotter in enumerate(self.plotters):
                hist = plotter.histograms[hist_name].Clone()
                hist.SetLineColor(self.colors[i % len(self.colors)])
                hist.SetLineWidth(2)
                
                max_y = max(max_y, hist.GetMaximum())
                
                if i == 0:
                    hist.Draw("HIST")
                else:
                    hist.Draw("HIST SAME")
                
                legend.AddEntry(hist, plotter.label, "l")
            
            # Set y-axis range
            self.plotters[0].histograms[hist_name].SetMaximum(max_y * 1.2)
            
            legend.Draw()
            canvas.Update()
            
            # Save
            png_path = self.output_dir / f"compare_{hist_name}.png"
            canvas.SaveAs(str(png_path))
            
            canvases[hist_name] = canvas
        
        print(f"Created {len(canvases)} comparison plots in {self.output_dir}")
        return canvases
    
    def _create_matplotlib_comparisons(self, histogram_names: List[str]) -> Dict[str, plt.Figure]:
        """Create matplotlib comparison plots."""
        figures = {}
        
        for hist_name in histogram_names:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for i, plotter in enumerate(self.plotters):
                # This would require extracting data from histograms
                # Simplified placeholder
                ax.set_title(f"Comparison: {hist_name}")
            
            figures[hist_name] = fig
        
        return figures
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create summary table comparing configurations.
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        rows = []
        
        for i, config in enumerate(self.configs):
            row = {
                'config_id': i,
                'target_z': config.get('target_z', 'N/A'),
                'target_xy': config.get('target_xy', 'N/A'),
                'gap': config.get('gap', 'N/A'),
            }
            
            # Add histogram statistics if available
            if i < len(self.plotters) and self.plotters[i].histograms:
                for name, hist in self.plotters[i].histograms.items():
                    if ROOT_AVAILABLE and isinstance(hist, ROOT.TH1):
                        row[f'{name}_entries'] = hist.GetEntries()
                        row[f'{name}_mean'] = hist.GetMean()
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = self.output_dir / "configuration_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved comparison table: {csv_path}")
        
        return df
    
    def save_all_to_root(self, filename: str = "comparison.root"):
        """Save all histograms to a single ROOT file."""
        if not ROOT_AVAILABLE:
            print("ROOT not available, skipping ROOT output")
            return
        
        filepath = self.output_dir / filename
        f = ROOT.TFile.Open(str(filepath), "RECREATE")
        
        for i, plotter in enumerate(self.plotters):
            # Create directory for each configuration
            dir_name = f"config_{i}_{plotter.label}"
            d = f.mkdir(dir_name)
            d.cd()
            
            for hist in plotter.histograms.values():
                hist.Write()
        
        f.Close()
        print(f"Saved all histograms: {filepath}")


def generate_config_plots(
    config: Dict[str, Any],
    data_file: str,
    output_dir: str = "config_plots"
) -> str:
    """
    Generate all plots for a single configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    data_file : str
        Path to simulation data
    output_dir : str
        Output directory
        
    Returns
    -------
    str
        Path to output directory
    """
    plotter = ConfigurationPlotter(config, output_dir)
    df = plotter.load_data(data_file)
    plotter.create_histograms(df)
    plotter.save_histograms()
    plotter.create_summary_canvas()
    
    return str(plotter.output_dir)


def compare_configurations(
    config_files: List[str],
    data_files: List[str],
    output_dir: str = "comparison_plots"
) -> str:
    """
    Compare multiple configurations.
    
    Parameters
    ----------
    config_files : list
        Paths to configuration JSON files
    data_files : list
        Paths to corresponding data files
    output_dir : str
        Output directory
        
    Returns
    -------
    str
        Path to output directory
    """
    # Load configurations
    configs = []
    for cf in config_files:
        with open(cf, 'r') as f:
            configs.append(json.load(f))
    
    comparator = ConfigurationComparator(configs, output_dir)
    comparator.load_all_data(data_files)
    comparator.create_comparison_plots()
    comparator.create_summary_table()
    comparator.save_all_to_root()
    
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ROOT plots for DAMSA configuration comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single configuration
    single_parser = subparsers.add_parser('single', help='Plot single configuration')
    single_parser.add_argument('--config', required=True, help='Configuration JSON file')
    single_parser.add_argument('--data', required=True, help='Simulation data file')
    single_parser.add_argument('--output', default='config_plots', help='Output directory')
    
    # Compare configurations
    compare_parser = subparsers.add_parser('compare', help='Compare configurations')
    compare_parser.add_argument('--configs', nargs='+', required=True, 
                               help='Configuration JSON files')
    compare_parser.add_argument('--data', nargs='+', required=True,
                               help='Corresponding data files')
    compare_parser.add_argument('--output', default='comparison_plots', help='Output directory')
    
    # From optimization results
    results_parser = subparsers.add_parser('from-results', 
                                          help='Generate plots from optimization results')
    results_parser.add_argument('results_dir', help='Optimization results directory')
    results_parser.add_argument('--top-n', type=int, default=5,
                               help='Number of top solutions to compare')
    results_parser.add_argument('--output', default=None, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        with open(args.config, 'r') as f:
            config = json.load(f)
        generate_config_plots(config, args.data, args.output)
        
    elif args.command == 'compare':
        if len(args.configs) != len(args.data):
            parser.error("Number of config files must match number of data files")
        compare_configurations(args.configs, args.data, args.output)
        
    elif args.command == 'from-results':
        # Load Pareto solutions and generate comparison
        results_dir = Path(args.results_dir)
        pareto_csv = results_dir / "pareto_solutions.csv"
        
        if not pareto_csv.exists():
            print(f"Error: {pareto_csv} not found")
            return 1
        
        df = pd.read_csv(pareto_csv)
        
        # Get top N solutions (by score if available, else first N)
        if 'score' in df.columns:
            top_df = df.nsmallest(args.top_n, 'score')
        else:
            top_df = df.head(args.top_n)
        
        print(f"Top {len(top_df)} configurations from Pareto front:")
        print(top_df)
        
        # Create config dicts
        configs = []
        for _, row in top_df.iterrows():
            configs.append({
                'target_z': row.get('target_z_cm', row.get('target_z', 10.0)),
                'target_xy': row.get('target_xy_cm', row.get('target_xy', 5.0)),
                'gap': row.get('gap_cm', row.get('gap', 47.0)),
            })
        
        output_dir = args.output or str(results_dir / "config_comparison")
        
        print(f"\nConfiguration comparison would be saved to: {output_dir}")
        print("Note: Actual simulation data files needed for histogram generation")
        
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
