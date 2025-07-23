#!/usr/bin/env python3
"""
Graph Generator for progress curves
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

# Set matplotlib parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'lines.linewidth': 2.0,
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False
})

class GraphGenerator:
    """Generate graphs from kinetics data."""
    
    def __init__(self, data_folder: str = "."):
        self.data_folder = Path(data_folder)
        self.viridis_colors = self._get_viridis_colors()
        
    def _get_viridis_colors(self) -> List[str]:
        """Get full viridis color palette matching kinetics_analyzer.py."""
        return [
            '#440154', '#482777', '#3f4a8a', '#31678e', '#26838f',
            '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825', '#f0f921',
            # Extended viridis-like colors for more datasets
            '#4c0c72', '#5a187b', '#682681', '#753581', '#824381',
            '#8e5181', '#9a5f81', '#a66c82', '#b17a83', '#bc8785'
        ]
    
    def _select_viridis_colors(self, n_datasets: int) -> List[str]:
        """Select colors from viridis palette with maximum contrast."""
        full_palette = self.viridis_colors
        
        if n_datasets <= 1:
            return [full_palette[4]]  # Middle viridis color for single dataset
        elif n_datasets == 2:
            return [full_palette[1], full_palette[8]]  # Dark purple and bright green
        elif n_datasets == 3:
            return [full_palette[0], full_palette[5], full_palette[9]]  # Dark purple, teal, yellow
        elif n_datasets == 4:
            return [full_palette[0], full_palette[3], full_palette[6], full_palette[9]]
        elif n_datasets == 5:
            return [full_palette[0], full_palette[2], full_palette[5], full_palette[7], full_palette[9]]
        else:
            # For 6+ datasets, use evenly spaced colors across the full palette
            step = len(full_palette) / n_datasets
            indices = [int(i * step) for i in range(n_datasets)]
            return [full_palette[i] for i in indices]
    
    def _load_data_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load data from a processed text file."""
        try:
            # Read the file, skipping comment lines
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Find the data start (after the header)
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('Time (s)'):
                    data_start = i
                    break
            
            # Read the data
            data_lines = lines[data_start:]
            data_content = '\n'.join(data_lines)
            
            # Parse as tab-separated values
            from io import StringIO
            df = pd.read_csv(StringIO(data_content), sep='\t')
            
            # Ensure we have the expected columns
            if len(df.columns) >= 2:
                df.columns = ['time', 'absorbance']
                df = df.dropna()
                return df
            else:
                print(f"Warning: {filepath} doesn't have expected format")
                return None
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _sort_files_numerically(self, files: List[Path]) -> List[Path]:
        """Sort files numerically by their numeric prefix."""
        def extract_number(filepath: Path) -> int:
            """Extract numeric part from filename like '1.txt' -> 1."""
            try:
                return int(filepath.stem)
            except ValueError:
                # If filename doesn't start with number, sort alphabetically
                return float('inf')
        
        return sorted(files, key=extract_number)
    
    def _select_files(self) -> Tuple[List[Path], str, str]:
        """Interactive file selection. Returns (files, title, plot_mode)."""
        if not self.data_folder.exists():
            print(f"Error: Data folder '{self.data_folder}' not found")
            return [], "", ""
        
        # Get all .txt files in the folder and sort numerically
        txt_files = list(self.data_folder.glob("*.txt"))
        txt_files = self._sort_files_numerically(txt_files)
        
        if not txt_files:
            print(f"No .txt files found in {self.data_folder}")
            return [], "", ""
        
        print("\nAvailable data files:")
        for i, file in enumerate(txt_files, 1):
            print(f"{i:2d}. {file.name}")
        
        print("\nOptions:")
        print("  - Enter specific numbers (e.g., '1,3,5' or '1 3 5')")
        print("  - Enter 'all' to select all files")
        print("  - Press Enter to select first file only")
        
        choice = input("\nEnter your choice: ").strip()
        
        selected_files = []
        if not choice:
            selected_files = [txt_files[0]]
        elif choice.lower() == 'all':
            selected_files = txt_files
        else:
            # Parse specific file numbers
            try:
                # Handle both comma and space separated
                if ',' in choice:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                else:
                    indices = [int(x) - 1 for x in choice.split()]
                
                for idx in indices:
                    if 0 <= idx < len(txt_files):
                        selected_files.append(txt_files[idx])
                    else:
                        print(f"Warning: File number {idx + 1} is out of range")
                
                if not selected_files:
                    selected_files = [txt_files[0]]
                    
            except ValueError:
                print("Invalid input. Using first file only.")
                selected_files = [txt_files[0]]
        
        # Ask for custom title
        if len(selected_files) == 1:
            default_title = f"Kinetics Data: {selected_files[0].stem}"
        else:
            default_title = "Kinetics Data Comparison"
        
        custom_title = input(f"\nGraph title (default: '{default_title}'): ").strip()
        title = custom_title if custom_title else default_title
        
        # Ask for plot mode if multiple files
        plot_mode = "single"
        if len(selected_files) > 1:
            print("\nPlot mode:")
            print("1. Overlay (all datasets on same graph)")
            print("2. Individual (separate graph for each dataset)")
            
            mode_choice = input("Choose mode (1-2, default=1): ").strip()
            plot_mode = "individual" if mode_choice == "2" else "overlay"
        
        return selected_files, title, plot_mode
    
    def _create_single_plot(self, data: pd.DataFrame, title: str, 
                           color: str = '#440154') -> plt.Figure:
        """Create a single dataset plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot with professional styling
        ax.scatter(data['time'], data['absorbance'], 
                  s=20, alpha=0.6, color=color, edgecolors='none')
        
        # Styling
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Absorbance (UA)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Remove top and right spines (already set in rcParams)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Clean white background
        ax.set_facecolor('white')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def _create_multiple_plot(self, datasets: Dict[str, pd.DataFrame], title: str) -> plt.Figure:
        """Create a multiple dataset overlay plot."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Sort dataset names numerically for proper legend order
        sorted_names = self._sort_dataset_names(list(datasets.keys()))
        
        # Get dynamic viridis colors for maximum contrast
        selected_colors = self._select_viridis_colors(len(sorted_names))
        
        # Plot each dataset with dynamic viridis colors in sorted order
        for i, filename in enumerate(sorted_names):
            data = datasets[filename]
            color = selected_colors[i % len(selected_colors)]
            
            ax.scatter(data['time'], data['absorbance'], 
                      s=15, alpha=0.6, color=color, 
                      label=filename, edgecolors='none')
        
        # Professional styling
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Absorbance (UA)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Clean legend with no background
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                          frameon=False, fancybox=False, shadow=False)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Clean white background
        ax.set_facecolor('white')
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        
        return fig
    
    def _sort_dataset_names(self, names: List[str]) -> List[str]:
        """Sort dataset names numerically (e.g., 1, 2, 3, ..., 8, 9, 10, 11)."""
        def extract_number(name: str) -> int:
            """Extract numeric part from dataset name."""
            try:
                return int(name)
            except ValueError:
                # If not a number, sort alphabetically at the end
                return float('inf')
        
        return sorted(names, key=extract_number)
    
    def _get_save_options(self) -> List[str]:
        """Get format options. Save directory is always current folder."""
        print("\nSave options:")
        print("1. PNG (default, good for presentations)")
        print("2. PDF (vector format, best for publications)")
        print("3. SVG (vector format, editable)")
        print("4. EPS (vector format, traditional publication)")
        print("5. All formats")
        
        format_choice = input("Choose format (1-5, default=1): ").strip()
        
        format_map = {
            '1': ['png'],
            '2': ['pdf'],
            '3': ['svg'],
            '4': ['eps'],
            '5': ['png', 'pdf', 'svg', 'eps']
        }
        
        return format_map.get(format_choice, ['png'])
    
    def _save_figure(self, fig: plt.Figure, base_filename: str, 
                    formats: List[str]) -> None:
        """Save figure in specified formats in current directory."""
        for fmt in formats:
            filename = f"{base_filename}.{fmt}"
            
            # Format-specific settings
            if fmt == 'png':
                fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            elif fmt in ['pdf', 'svg', 'eps']:
                fig.savefig(filename, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            
            print(f"Saved: {filename}")
    
    def generate_graphs(self) -> None:
        """Main method to generate  graphs."""
        print("Graph Generator for Kinetics Data")
        print("=" * 55)
        
        # Select files, get title, and plot mode
        selected_files, title, plot_mode = self._select_files()
        
        if not selected_files:
            print("No files selected. Exiting.")
            return
        
        # Load data
        datasets = {}
        for filepath in selected_files:
            data = self._load_data_file(filepath)
            if data is not None:
                datasets[filepath.stem] = data
            else:
                print(f"Failed to load {filepath}")
        
        if not datasets:
            print("No valid datasets loaded. Exiting.")
            return
        
        print(f"\nLoaded {len(datasets)} dataset(s)")
        
        # Get save options
        formats = self._get_save_options()
        
        # Generate plots based on mode
        if len(datasets) == 1:
            # Single dataset
            filename, data = next(iter(datasets.items()))
            # Use dynamic color selection even for single dataset
            color = self._select_viridis_colors(1)[0]
            fig = self._create_single_plot(data, title, color)
            self._save_figure(fig, f"kinetics_{filename}", formats)
            
        elif plot_mode == "overlay":
            # Multiple datasets on same graph (overlay mode)
            print(f"\nGenerating overlay plot for {len(datasets)} datasets...")
            fig = self._create_multiple_plot(datasets, title)
            self._save_figure(fig, "kinetics_comparison", formats)
            
        else:  # plot_mode == "individual"
            # Multiple datasets as separate graphs
            print(f"\nGenerating individual plots for {len(datasets)} datasets...")
            
            # Get dynamic colors for consistency across individual plots
            sorted_names = self._sort_dataset_names(list(datasets.keys()))
            selected_colors = self._select_viridis_colors(len(sorted_names))
            
            for i, filename in enumerate(sorted_names):
                data = datasets[filename]
                color = selected_colors[i % len(selected_colors)]
                # Create individual title for each dataset
                individual_title = f"{title}: {filename}" if "Comparison" in title else f"Kinetics Data: {filename}"
                fig = self._create_single_plot(data, individual_title, color)
                self._save_figure(fig, f"kinetics_{filename}", formats)
                plt.close(fig)
        
        print(f"\nGraph generation complete! Files saved in current directory.")
        
        # Show plot if requested
        show_plot = input("Display plots now? (y/N): ").strip().lower()
        if show_plot == 'y':
            # Recreate and show the appropriate figure
            if len(datasets) == 1:
                filename, data = next(iter(datasets.items()))
                color = self._select_viridis_colors(1)[0]
                fig = self._create_single_plot(data, title, color)
            elif plot_mode == "overlay":
                fig = self._create_multiple_plot(datasets, title)
            else:  # Show the first individual plot as example
                first_name = self._sort_dataset_names(list(datasets.keys()))[0]
                data = datasets[first_name]
                color = self._select_viridis_colors(len(datasets))[0]
                individual_title = f"{title}: {first_name}" if "Comparison" in title else f"Kinetics Data: {first_name}"
                fig = self._create_single_plot(data, individual_title, color)
            
            plt.show()

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Generate graphs from kinetics data'
    )
    parser.add_argument('--data-folder', '-d', default='.',
                       help='Path to folder containing processed data files (default: current directory)')
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = GraphGenerator(args.data_folder)
    generator.generate_graphs()

if __name__ == "__main__":
    main()
