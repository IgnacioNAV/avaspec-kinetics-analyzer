#!/usr/bin/env python3
"""
Enzyme Kinetics Analyzer v1 - Streamlit Web Application
Interactive web-based tool for progress curves analysis with AvaSpec integration.
Uses Streamlit for the web interface and Plotly for interactive plotting.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pint
from pathlib import Path
import re
import io
import tempfile
import subprocess
import sys
from typing import Dict, Tuple, List, Optional, Union
import base64
import random
import logging
from datetime import datetime
import warnings

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Kinetics Analyzer v1",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ColorPalette:
    """Centralized color management for consistent styling across the application."""
    
    def __init__(self):
        self.viridis_colors = [
            '#440154', '#482777', '#3f4a8a', '#31678e', '#26838f',
            '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825', '#f0f921',
            # Extended viridis-like colors for more datasets
            '#4c0c72', '#5a187b', '#682681', '#753581', '#824381',
            '#8e5181', '#9a5f81', '#a66c82', '#b17a83', '#bc8785'
        ]
        
        self.nostalgic_colors = [
            '#000000', '#FF0000', '#0000FF', '#00FF00', '#00FFFF',
            '#FF00FF', '#FFFF00', '#FF8000', '#8000FF', '#00FF80',
            # Extended nostalgic colors for more datasets
            '#800000', '#000080', '#008000', '#008080', '#800080',
            '#808000', '#804000', '#400080', '#008040', '#800040'
        ]
        
        self.viridis_palette = [
            ('#440154', 'Dark Purple'), ('#482777', 'Deep Violet'), 
            ('#3f4a8a', 'Royal Blue'), ('#31678e', 'Ocean Blue'),
            ('#26838f', 'Teal Blue'), ('#1f9d8a', 'Forest Green'),
            ('#6cce5a', 'Lime Green'), ('#b6de2b', 'Yellow Green'),
            ('#fee825', 'Golden Yellow'), ('#f0f921', 'Bright Yellow')
        ]
        
        self.nostalgic_palette = [
            ('#000000', 'Black'), ('#FF0000', 'Red'), ('#0000FF', 'Blue'), 
            ('#00FF00', 'Green'), ('#00FFFF', 'Cyan'), ('#FF00FF', 'Magenta'),
            ('#FFFF00', 'Yellow'), ('#FF8000', 'Orange'), ('#8000FF', 'Purple'),
            ('#00FF80', 'Spring Green')
        ]
    
    def get_colors(self, palette_type: str = 'viridis') -> List[str]:
        """Get color list for specified palette type."""
        if palette_type == 'nostalgic':
            return self.nostalgic_colors
        return self.viridis_colors
    
    def get_palette(self, palette_type: str = 'viridis') -> List[Tuple[str, str]]:
        """Get palette with color names for UI selection."""
        if palette_type == 'nostalgic':
            return self.nostalgic_palette
        return self.viridis_palette
    
    def select_colors(self, n_datasets: int, palette_type: str = 'viridis') -> List[str]:
        """Select colors with maximum contrast for given number of datasets."""
        full_colors = self.get_colors(palette_type)
        
        if palette_type == 'nostalgic':
            # For nostalgic palette, use sequential order
            if n_datasets <= 1:
                return [full_colors[0]]  # Black for single dataset
            else:
                return [full_colors[i % len(full_colors)] for i in range(n_datasets)]
        else:
            # Viridis logic - evenly spaced colors for maximum contrast
            if n_datasets <= 1:
                return [full_colors[len(full_colors)//2]]  # Middle color for single
            else:
                if n_datasets >= len(full_colors):
                    return [full_colors[i % len(full_colors)] for i in range(n_datasets)]
                else:
                    # Evenly distribute across spectrum
                    step = (len(full_colors) - 1) / (n_datasets - 1) if n_datasets > 1 else 0
                    indices = [int(i * step) for i in range(n_datasets)]
                    return [full_colors[i] for i in indices]

class AvaSpecIntegration:
    """Handle AvaSpec Excel file detection and conversion."""
    
    def __init__(self):
        self.conversion_script_path = Path(__file__).parent / "avaspec_excel_to_txt.py"
        logger.info(f"AvaSpec integration initialized. Script path: {self.conversion_script_path}")
    
    def is_avaspec_excel(self, uploaded_file) -> bool:
        """
        Detect if an Excel file is from AvaSpec spectrometer.
        Checks for characteristic patterns in the data structure.
        """
        try:
            # Read first few rows to check structure
            df = pd.read_excel(uploaded_file, sheet_name=0, nrows=10, header=None)
            
            # Reset file position for subsequent reads
            uploaded_file.seek(0)
            
            # AvaSpec files typically have:
            # - Timestamp in first column
            # - Time in milliseconds in second column (column B)
            # - Absorbance data in third column (column C)
            # - Data starts around row 4
            
            if len(df.columns) >= 3:
                # Check if column B contains time-like data (milliseconds, large numbers)
                col_b_data = df.iloc[3:, 1].dropna()  # Skip header rows
                if not col_b_data.empty:
                    try:
                        # Check if values look like milliseconds (typically > 1000)
                        numeric_data = pd.to_numeric(col_b_data, errors='coerce').dropna()
                        if len(numeric_data) > 0:
                            avg_value = numeric_data.mean()
                            # AvaSpec time data is typically in milliseconds
                            if avg_value > 1000 and avg_value < 1000000:  # Reasonable range
                                logger.info("Detected potential AvaSpec Excel file")
                                return True
                    except Exception:
                        pass
            
            return False
        except Exception as e:
            logger.warning(f"Error checking if file is AvaSpec Excel: {str(e)}")
            return False
    
    def convert_avaspec_excel(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Convert AvaSpec Excel file to standard format.
        Returns converted DataFrame or None if conversion fails.
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_excel_path = Path(tmp_file.name)
            
            # Read Excel file in AvaSpec format
            df = pd.read_excel(tmp_excel_path, sheet_name=0, skiprows=3, header=None)
            
            if df.empty or len(df.columns) < 3:
                logger.error("AvaSpec Excel file doesn't have expected structure")
                tmp_excel_path.unlink()  # Cleanup
                return None
            
            # Process data according to AvaSpec format
            valid_rows = []
            for idx, row in df.iterrows():
                raw_time_ms = row.iloc[1]  # Column B (time in ms)
                raw_absorbance = row.iloc[2]  # Column C (absorbance)
                
                # Check if both values are numeric and not empty
                if pd.notna(raw_time_ms) and pd.notna(raw_absorbance):
                    try:
                        # Convert time from ms to s
                        time_sec = float(raw_time_ms) / 1000.0
                        
                        # Handle absorbance - convert to string and ensure dot decimal
                        absorbance_str = str(raw_absorbance).replace(',', '.')
                        absorbance = float(absorbance_str)
                        
                        valid_rows.append([time_sec, absorbance])
                        
                    except (ValueError, TypeError):
                        continue
            
            # Cleanup temporary file
            tmp_excel_path.unlink()
            
            if not valid_rows:
                logger.error("No valid data found in AvaSpec Excel file")
                return None
            
            # Create standardized DataFrame
            result_df = pd.DataFrame(valid_rows, columns=['time', 'product'])
            logger.info(f"Successfully converted AvaSpec Excel file: {len(result_df)} data points")
            return result_df
            
        except Exception as e:
            logger.error(f"Error converting AvaSpec Excel file: {str(e)}")
            return None

class MemoryManager:
    """Handle memory management and session state cleanup."""
    
    @staticmethod
    def cleanup_removed_datasets(current_filenames: set):
        """Clean up session state for datasets that are no longer uploaded."""
        if not hasattr(st.session_state, 'data_sets'):
            return
        
        datasets_to_remove = [name for name in st.session_state.data_sets.keys() 
                            if name not in current_filenames]
        
        for dataset_name in datasets_to_remove:
            # Remove from data_sets
            if dataset_name in st.session_state.data_sets:
                del st.session_state.data_sets[dataset_name]
                logger.info(f"Cleaned up dataset: {dataset_name}")
            
            # Remove from regions
            if hasattr(st.session_state, 'regions') and dataset_name in st.session_state.regions:
                del st.session_state.regions[dataset_name]
                logger.info(f"Cleaned up regions for dataset: {dataset_name}")
            
            # Remove from manual colors
            if (hasattr(st.session_state, 'manual_colors') and 
                st.session_state.manual_colors and 
                dataset_name in st.session_state.manual_colors):
                del st.session_state.manual_colors[dataset_name]
                logger.info(f"Cleaned up manual colors for dataset: {dataset_name}")
            
            # Remove from AvaSpec converted files tracking
            if (hasattr(st.session_state, 'avaspec_converted_files') and 
                st.session_state.avaspec_converted_files and 
                dataset_name in st.session_state.avaspec_converted_files):
                del st.session_state.avaspec_converted_files[dataset_name]
                logger.info(f"Cleaned up AvaSpec converted file tracking for dataset: {dataset_name}")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, int]:
        """Get current memory usage statistics."""
        usage = {}
        if hasattr(st.session_state, 'data_sets'):
            usage['datasets'] = len(st.session_state.data_sets)
            total_points = sum(len(df) for df in st.session_state.data_sets.values())
            usage['total_data_points'] = total_points
        
        if hasattr(st.session_state, 'regions'):
            total_regions = sum(len(regions) for regions in st.session_state.regions.values())
            usage['total_regions'] = total_regions
        
        return usage

class BatchAnalyzer:
    """
    Safe batch analysis implementation to avoid infinite loops.
    Phase 4 improvement: Careful batch processing architecture.
    """
    
    def __init__(self, main_app):
        self.main_app = main_app
        self.max_iterations = 100  # Safety limit to prevent infinite loops
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def validate_batch_parameters(self, datasets: List[str], time_frame: Tuple[float, float]) -> bool:
        """
        Validate batch analysis parameters to prevent issues.
        Returns True if parameters are valid, False otherwise.
        """
        if not datasets or len(datasets) == 0:
            st.error("❌ No datasets selected for batch analysis.")
            return False
        
        if len(datasets) == 1:
            st.warning("⚠️ Only one dataset selected.")
        
        start_time, end_time = time_frame
        if start_time >= end_time:
            st.error("❌ Invalid time frame - start time must be less than end time.")
            return False
        
        # Validate that all datasets have data in the specified time range
        invalid_datasets = []
        for dataset_name in datasets:
            if dataset_name not in st.session_state.data_sets:
                invalid_datasets.append(dataset_name)
                continue
            
            data = st.session_state.data_sets[dataset_name]
            data_min_time = data['time'].min()
            data_max_time = data['time'].max()
            
            if start_time > data_max_time or end_time < data_min_time:
                invalid_datasets.append(dataset_name)
        
        if invalid_datasets:
            st.error(f"❌ Time frame ({start_time:.2f}s - {end_time:.2f}s) is outside data range for datasets: {', '.join(invalid_datasets)}")
            return False
        
        return True
    
    def perform_batch_analysis(self, datasets: List[str], time_frame: Tuple[float, float], 
                             analysis_params: Dict) -> Dict[str, Dict]:
        """
        Perform safe batch analysis across multiple datasets.
        
        Args:
            datasets: List of dataset names to analyze
            time_frame: Tuple of (start_time, end_time) for analysis
            analysis_params: Dictionary containing analysis parameters
            
        Returns:
            Dictionary with results for each dataset
        """
        results = {}
        start_time, end_time = time_frame
        
        # Safety check - prevent infinite loops
        if len(datasets) > self.max_iterations:
            st.error(f"❌ **Safety Limit**: Cannot process more than {self.max_iterations} datasets in batch mode.")
            return {}
        
        # Validate parameters before starting
        if not self.validate_batch_parameters(datasets, time_frame):
            return {}
        
        try:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_datasets = len(datasets)
            processed = 0
            
            for i, dataset_name in enumerate(datasets):
                # Update progress
                progress = (i + 1) / total_datasets
                progress_bar.progress(progress)
                status_text.text(f"Processing dataset {i + 1} of {total_datasets}: {dataset_name}")
                
                # Safety check within loop
                if i > self.max_iterations:
                    st.error(f"❌ **Safety Limit Reached**: Stopped processing after {self.max_iterations} datasets.")
                    break
                
                try:
                    # Generate unique region name for this dataset
                    batch_number = st.session_state.get('batch_counter', 0) + 1
                    region_name = f"Batch_{batch_number}_{dataset_name}"
                    
                    # Perform slope calculation for this dataset
                    result = self.main_app._calculate_slope(
                        dataset_name, 
                        start_time, 
                        end_time, 
                        region_name,
                        analysis_params.get('extinction_coeff'),
                        analysis_params.get('target_conc_unit', 'UA'),
                        analysis_params.get('target_time_unit', 's'),
                        analysis_params.get('enzyme_units')
                    )
                    
                    if result:
                        # Store result in session state
                        if dataset_name not in st.session_state.regions:
                            st.session_state.regions[dataset_name] = {}
                        
                        st.session_state.regions[dataset_name][region_name] = result
                        results[dataset_name] = result
                        processed += 1
                        logger.info(f"Successfully processed batch analysis for dataset: {dataset_name}")
                    else:
                        logger.warning(f"Failed to process batch analysis for dataset: {dataset_name}")
                        st.warning(f"⚠️ Could not analyze dataset: {dataset_name}")
                
                except Exception as e:
                    logger.error(f"Error in batch analysis for dataset {dataset_name}: {str(e)}")
                    st.error(f"❌ Error processing {dataset_name}: {str(e)}")
                    continue
            
            # Update batch counter to ensure unique region names
            st.session_state.batch_counter = st.session_state.get('batch_counter', 0) + 1
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show simple completion summary
            if processed > 0:
                st.success(f"✅ **Batch Analysis Complete**: Successfully processed {processed} out of {total_datasets} datasets.")
                if processed < total_datasets:
                    st.info(f"ℹ️ {total_datasets - processed} datasets could not be processed due to insufficient data in the specified time range.")
            else:
                st.error("❌ **Batch Analysis Failed**: No datasets could be processed. Please check your time range and data.")
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in batch analysis: {str(e)}")
            st.error(f"❌ **Critical Batch Analysis Error**: {str(e)}")
            return {}
    
    def suggest_time_frame(self, datasets: List[str]) -> Optional[Tuple[float, float]]:
        """
        Suggest an optimal time frame that works for all datasets.
        Returns None if no common time frame is found.
        """
        if not datasets:
            return None
        
        try:
            # Find the intersection of all dataset time ranges
            common_start = float('-inf')
            common_end = float('inf')
            
            for dataset_name in datasets:
                if dataset_name not in st.session_state.data_sets:
                    continue
                
                data = st.session_state.data_sets[dataset_name]
                data_start = data['time'].min()
                data_end = data['time'].max()
                
                common_start = max(common_start, data_start)
                common_end = min(common_end, data_end)
            
            if common_start < common_end and common_start != float('-inf'):
                # Suggest a time frame that covers the middle 80% of the common range
                time_range = common_end - common_start
                suggested_start = common_start + (time_range * 0.1)
                suggested_end = common_end - (time_range * 0.1)
                
                return (suggested_start, suggested_end)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error suggesting time frame: {str(e)}")
            return None

class ErrorHandler:
    """Centralized error handling with user-friendly messages."""
    
    @staticmethod
    def handle_file_load_error(filename: str, error: Exception) -> None:
        """Handle file loading errors with helpful suggestions."""
        error_msg = str(error).lower()
        
        if "permission" in error_msg:
            st.error(f"❌ **Permission Error**: Cannot access file '{filename}'. "
                    "Please check file permissions or try saving the file in a different location.")
        elif "memory" in error_msg or "size" in error_msg:
            st.error(f"❌ **Memory Error**: File '{filename}' is too large. "
                    "Try reducing the data size or splitting into smaller files.")
        elif "format" in error_msg or "excel" in error_msg:
            st.error(f"❌ **Format Error**: Cannot read file '{filename}'. "
                    "Please ensure it's a valid Excel, CSV, or TXT file.")
        else:
            st.error(f"❌ **Error loading '{filename}'**: {str(error)}")
            st.info("💡 **Suggestions**: \n"
                   "- Check that the file contains time and absorbance columns\n"
                   "- Ensure the file isn't corrupted or password-protected\n"
                   "- Try converting to a different format (CSV or TXT)")
    
    @staticmethod
    def handle_conversion_error(filename: str, error: Exception) -> None:
        """Handle AvaSpec conversion errors."""
        st.error(f"❌ **Conversion Error**: Could not convert AvaSpec file '{filename}'. "
                f"Error: {str(error)}")
        st.info("💡 **Suggestions**: \n"
               "- Ensure the file is a genuine AvaSpec export\n"
               "- Try opening the file in Excel to verify its contents")

class EnzymeKineticsStreamlit:
    """
    Main application class for enzyme kinetics analysis v1.
    """
    
    def __init__(self):
        self.ureg = pint.UnitRegistry()
        self._define_custom_units()
        
        # Initialize enhanced components
        self.color_palette = ColorPalette()
        self.avaspec_integration = AvaSpecIntegration()
        self.memory_manager = MemoryManager()
        self.error_handler = ErrorHandler()
        self.batch_analyzer = BatchAnalyzer(self)
        
        # Initialize session state
        if 'data_sets' not in st.session_state:
            st.session_state.data_sets = {}
        if 'regions' not in st.session_state:
            st.session_state.regions = {}
        if 'region_counter' not in st.session_state:
            st.session_state.region_counter = 0
        if 'avaspec_converted_files' not in st.session_state:
            st.session_state.avaspec_converted_files = {}  # Track converted AvaSpec files
        
        self.quotes = [
            "The only way to make sense out of change is to plunge into it, move with it, and join the dance - Watts",
            "What we observe is not nature itself, but nature exposed to our method of questioning - Heisenberg",
            "Kinetics tells us how fast, thermodynamics tells us how far",
            "Every reaction has its optimal path - trust the data to show you",
            "R² > 0.99: Excellent fit. R² < 0.98: Time for coffee and reconsideration",
            "The enzyme knows chemistry better than the chemist",
            "Linear kinetics: when enzymes behave like good students",
            "Why don't enzymes ever get tired? They have infinite turnover rates in their dreams",
            "Enzymes are like good friends: they lower your activation energy for getting things done",
            "What did the substrate say to the enzyme? You complete me... by breaking me apart",
            "Allosteric enzymes: proof that proteins have mood swings",
            "Fact: Some enzymes are so efficient they approach the diffusion limit - nature's speed demons",
            "Enzymes: billions of years of R&D, no patents required",
            "DNA: the world's most successful backup system (with occasional corrupted files)",
            "RNA: Jack of all trades, master of everything - sorry proteins!",
            "Ribozymes proved that RNA doesn't need protein chaperones to get things done",
            "DNA polymerase has 3' to 5' exonuclease activity because even evolution believes in proofreading",
            "Why did the ribosome break up with the polysome? Too much translation drama",
            "Fact: Your DNA contains the instructions for about 20,000-25,000 proteins (assembly required)",
            "Telomerase: the enzyme trying to make you immortal (aging has entered the chat)",
            "DNA repair mechanisms: your genome's personal IT department working 24/7",
            "Why don't DNA strands ever get lost? They always know their 5' from their 3' end",
            "LUCA (Last Universal Common Ancestor): the ultimate single parent raising 4 billion years of offspring",
            "RNA World hypothesis: back when RNA was CEO, CFO, and the entire workforce",
            "Primordial soup: Earth's first attempt at molecular cuisine (no recipe book included)",
            "Iron-sulfur clusters: the original catalysts, still going strong after 3.8 billion years",
            "Miller-Urey experiment: proof that you can make amino acids from lightning and patience",
            "Fact: Cyanobacteria invented photosynthesis and accidentally created the oxygen apocalypse",
            "Hydrothermal vents: where life possibly started in nature's pressure cookers",
            "Self-replication: the ultimate mystery - how did molecules learn to copy themselves?",
            "From chemistry to biology: the greatest startup story never fully documented",
            "Autocatalytic networks: when molecules formed the first self-sustaining businesses",
            "Evolution: 4 billion years of A/B testing with no rollback option",
            "Endosymbiotic theory: the first successful corporate merger in cellular history",
            "In God we trust, all others must bring data - Deming",
            "The best thing about being a statistician is that you get to play in everyone's backyard - Tukey",
            "Without data, you're just another person with an opinion - Edwards",
            "Correlation does not imply causation, but it does waggle its eyebrows suggestively",
            "A good plot is worth a thousand p-values",
            "Caffeine: the molecule that powers science",
            "Good coffee, good data, good science",
            "Lab rule #1: Never trust data collected before the first cup of coffee",
            "Science runs on coffee and curiosity",
            "The universal solvent for scientific problems: coffee",
            "Fact: Your morning coffee contains over 1000 different chemical compounds",
            "Fact: Ribozymes can cut and paste RNA like molecular scissors and glue",
            "Fact: The ribosome is a ribozyme - your protein factory is made of RNA",
            "Fact: Some RNA molecules can evolve in test tubes in just hours",
            "Fact: DNA repair enzymes fix about 1000 DNA damages per cell per day - busy little workers"
        ]
    
    def _define_custom_units(self) -> None:
        """Define custom units for biochemical applications."""
        try:
            self.ureg.define('enzyme_unit = 1 * umol / minute = EU')
            self.ureg.define('international_unit = 1 * umol / minute = IU')
        except Exception:
            pass
    
    def _get_plot_width(self) -> int:
        """Get plot width based on user selection."""
        width_mode = st.session_state.get('plot_width_mode', 'container')
        
        width_mapping = {
            'container': None,
            'small': 600,
            'medium': 800, 
            'large': 1000,
            'xlarge': 1200
        }
        
        return width_mapping.get(width_mode, None)
    
    def _load_data_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file with enhanced AvaSpec integration.
        Phase 1 improvement: Automatic AvaSpec detection and conversion.
        """
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            # Phase 1: AvaSpec Excel Integration
            if file_extension in ['.xlsx', '.xls']:
                # Check if this is an AvaSpec Excel file
                if self.avaspec_integration.is_avaspec_excel(uploaded_file):
                    # Show user prompt for AvaSpec conversion
                    if f"avaspec_prompt_{uploaded_file.name}" not in st.session_state:
                        st.session_state[f"avaspec_prompt_{uploaded_file.name}"] = "pending"
                    
                    if st.session_state[f"avaspec_prompt_{uploaded_file.name}"] == "pending":
                        st.info(f"🔬 **Excel File Detection**: {uploaded_file.name}")
                        st.write("**Is this Excel file exported directly from AvaSpec spectrometer?**")
                        st.caption("AvaSpec files need special processing to convert time units and decimal formats.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Yes, convert.", key=f"convert_{uploaded_file.name}"):
                                st.session_state[f"avaspec_prompt_{uploaded_file.name}"] = "convert"
                                st.rerun()
                        with col2:
                            if st.button(f"No.", key=f"standard_{uploaded_file.name}"):
                                st.session_state[f"avaspec_prompt_{uploaded_file.name}"] = "standard"
                                st.rerun()
                        
                        # Don't process the file yet, wait for user decision
                        return None
                    
                    elif st.session_state[f"avaspec_prompt_{uploaded_file.name}"] == "convert":
                        # User chose to convert AvaSpec file
                        with st.spinner(f"🔄 Converting AvaSpec file: {uploaded_file.name}..."):
                            converted_data = self.avaspec_integration.convert_avaspec_excel(uploaded_file)
                            if converted_data is not None:
                                # Track converted file for download option
                                st.session_state.avaspec_converted_files[uploaded_file.name] = uploaded_file.name
                                st.success(f"✅ **AvaSpec Conversion Successful**: {uploaded_file.name} "
                                         f"({len(converted_data)} data points)")
                                logger.info(f"Successfully converted AvaSpec file: {uploaded_file.name}")
                                return converted_data
                            else:
                                self.error_handler.handle_conversion_error(uploaded_file.name, 
                                                                         Exception("Conversion failed"))
                                return None
                    
                    elif st.session_state[f"avaspec_prompt_{uploaded_file.name}"] == "standard":
                        # User chose to load as standard Excel
                        return self._load_excel_file(uploaded_file)
                else:
                    # Regular Excel file, load normally
                    return self._load_excel_file(uploaded_file)
            
            # Handle other file types
            elif file_extension == '.txt':
                return self._load_txt_file(uploaded_file)
            elif file_extension == '.csv':
                return self._load_csv_file(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            self.error_handler.handle_file_load_error(uploaded_file.name, e)
            return None
    
    def _load_txt_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from TXT file."""
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.strip().split('\n')
        
        # Skip comment lines and find header
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#') or line.strip().startswith('%'):
                continue
            if any(keyword in line.lower() for keyword in ['time', 'ua', 'concentration']):
                data_start = i
                break
            if i > 10:  # Assume data starts within first 10 lines
                data_start = i
                break
        
        # Read data
        data_lines = lines[data_start:]
        df = pd.read_csv(io.StringIO('\n'.join(data_lines)), sep='\t', comment='#')
        
        return self._standardize_columns(df)
    
    def _load_csv_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(uploaded_file)
        return self._standardize_columns(df)
    
    def _load_excel_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from Excel file."""
        df = pd.read_excel(uploaded_file)
        return self._standardize_columns(df)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistent processing."""
        # Create a mapping of possible column names to standard names
        time_keywords = ['time', 'tiempo', 'temp', 'seconds', 'sec', 's']
        concentration_keywords = ['ua', 'concentration', 'conc', 'product', 'absorbance', 'abs']
        
        # Find time column
        time_col = None
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in time_keywords):
                time_col = col
                break
        
        # Find concentration column
        conc_col = None
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in concentration_keywords):
                conc_col = col
                break
        
        if time_col is None or conc_col is None:
            # Use first two numeric columns if keywords not found
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                time_col = numeric_cols[0]
                conc_col = numeric_cols[1]
            else:
                raise ValueError("Could not identify time and concentration columns")
        
        # Standardize column names
        result_df = pd.DataFrame()
        result_df['time'] = pd.to_numeric(df[time_col], errors='coerce')
        result_df['product'] = pd.to_numeric(df[conc_col], errors='coerce')
        
        # Remove rows with NaN values
        result_df = result_df.dropna()
        
        return result_df
    
    def _create_interactive_plot(self, dataset_name: str = None, show_regions: bool = True, overlay_datasets: List[str] = None) -> go.Figure:
        """Create interactive Plotly figure with enhanced color management."""
        fig = go.Figure()
        
        # Get current color palette
        current_palette_type = st.session_state.get('color_palette_type', 'viridis')
        
        # Theme-aware styling
        if hasattr(st.session_state, 'theme') and st.session_state.theme == 'dark':
            plot_bg = '#0e1117'
            paper_bg = '#0e1117'
            grid_color = '#2a2e3b'
            text_color = '#ffffff'
            line_color = '#4a5568'
        else:
            plot_bg = 'white'
            paper_bg = 'white' 
            grid_color = '#e2e8f0'
            text_color = '#2d3748'
            line_color = '#cbd5e0'
        
        # Determine which datasets to plot
        datasets_to_plot = []
        if overlay_datasets:
            datasets_to_plot = [ds for ds in overlay_datasets if ds in st.session_state.data_sets]
        elif dataset_name and dataset_name in st.session_state.data_sets:
            datasets_to_plot = [dataset_name]
        elif dataset_name is None:
            datasets_to_plot = list(st.session_state.data_sets.keys())
        
        # Use enhanced color selection
        if st.session_state.get('color_mode', 'auto') == 'manual' and 'manual_colors' in st.session_state:
            # Get manual colors from palette
            palette = self.color_palette.get_palette(current_palette_type)
            colors = []
            for ds_name in datasets_to_plot:
                color_idx = st.session_state.manual_colors.get(ds_name, 0)
                colors.append(palette[color_idx][0])  # Get hex color
        else:
            # Use automatic color selection
            colors = self.color_palette.select_colors(len(datasets_to_plot), current_palette_type)
        
        # Add data traces
        for i, ds_name in enumerate(datasets_to_plot):
            data = st.session_state.data_sets[ds_name]
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=data['product'],
                mode='markers',
                name=ds_name,
                marker=dict(
                    size=st.session_state.get('plot_marker_size', 3), 
                    color=color, 
                    opacity=st.session_state.get('plot_marker_opacity', 0.5)
                ),
                hovertemplate=f'{ds_name}: %{{x:.1f}}s, %{{y:.3f}} UA<extra></extra>',
                showlegend=False,
                legendgroup=ds_name,
            ))
            
            # Add invisible trace for legend with full opacity
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name=ds_name,
                marker=dict(size=8, color=color, opacity=1.0),
                showlegend=True,
                legendgroup=ds_name,
                hoverinfo='skip'
            ))
        
        # Add current selection highlight
        if hasattr(st.session_state, 'selected_start') and hasattr(st.session_state, 'selected_end'):
            if 'selected_start' in st.session_state and 'selected_end' in st.session_state:
                all_data = [st.session_state.data_sets[ds] for ds in datasets_to_plot]
                y_min = min(data['product'].min() for data in all_data) * 0.95
                y_max = max(data['product'].max() for data in all_data) * 1.05
                
                selection_color = '#6cce5a'  # Viridis green
                fig.add_shape(
                    type="rect",
                    x0=st.session_state.selected_start,
                    y0=y_min,
                    x1=st.session_state.selected_end,
                    y1=y_max,
                    fillcolor=f'rgba(108, 206, 90, 0.2)',
                    line=dict(color=selection_color, width=2, dash='dash'),
                    layer="below"
                )
                
                fig.add_annotation(
                    x=(st.session_state.selected_start + st.session_state.selected_end) / 2,
                    y=y_max * 0.98,
                    text=f"Selected: {st.session_state.selected_start:.1f}s - {st.session_state.selected_end:.1f}s",
                    showarrow=False,
                    font=dict(size=12, color=selection_color),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor=selection_color,
                    borderwidth=1
                )
        
        # Add region highlights if available
        if show_regions and hasattr(st.session_state, 'regions'):
            # Calculate global min/max for consistent shading
            all_data = [st.session_state.data_sets[ds] for ds in datasets_to_plot]
            global_min = min(data['product'].min() for data in all_data)
            global_max = max(data['product'].max() for data in all_data)
            
            if dataset_name:
                # Single dataset mode - show regions for selected dataset
                if dataset_name in st.session_state.regions:
                    for region_name, region_data in st.session_state.regions[dataset_name].items():
                        if 'start' in region_data and 'end' in region_data:
                            data = st.session_state.data_sets[dataset_name]
                            
                            # Add shaded region background
                            fig.add_shape(
                                type="rect",
                                x0=region_data['start'],
                                y0=global_min * 0.98,
                                x1=region_data['end'],
                                y1=global_max * 1.02,
                                fillcolor='rgba(128, 128, 128, 0.1)',
                                line_width=0,
                            )
                            
                            # Add fitted line if slope exists
                            if 'slope' in region_data and 'intercept' in region_data:
                                region_subset = data[
                                    (data['time'] >= region_data['start']) & 
                                    (data['time'] <= region_data['end'])
                                ]
                                if not region_subset.empty:
                                    fitted_y = region_data['slope'] * region_subset['time'] + region_data['intercept']
                                    # Get the color for this dataset
                                    dataset_color = colors[datasets_to_plot.index(dataset_name)] if dataset_name in datasets_to_plot else colors[0]
                                    # Add glow effect with wider transparent line first
                                    fig.add_trace(go.Scatter(
                                        x=region_subset['time'],
                                        y=fitted_y,
                                        mode='lines',
                                        line=dict(width=st.session_state.get('plot_line_width', 5)*2, color=f'rgba({int(dataset_color[1:3], 16)}, {int(dataset_color[3:5], 16)}, {int(dataset_color[5:7], 16)}, 0.2)'),
                                        hoverinfo='skip',
                                        showlegend=False
                                    ))
                                    # Add main line
                                    fig.add_trace(go.Scatter(
                                        x=region_subset['time'],
                                        y=fitted_y,
                                        mode='lines',
                                        name=f'{region_name} fit',
                                        line=dict(width=st.session_state.get('plot_line_width', 5), color=dataset_color),
                                        opacity=st.session_state.get('plot_line_opacity', 1.0),
                                        hovertemplate='Fit: %{y:.3f}<extra></extra>',
                                        showlegend=False
                                    ))
            else:
                # Overlay mode - show regression lines without background shading
                for dataset_idx, ds_name in enumerate(datasets_to_plot):
                    if ds_name in st.session_state.regions:
                        # Get the correct color for this dataset
                        dataset_color = colors[dataset_idx % len(colors)]
                        
                        for region_name, region_data in st.session_state.regions[ds_name].items():
                            if 'start' in region_data and 'end' in region_data:
                                # Add fitted line if slope exists
                                if 'slope' in region_data and 'intercept' in region_data:
                                    data = st.session_state.data_sets[ds_name]
                                    region_subset = data[
                                        (data['time'] >= region_data['start']) & 
                                        (data['time'] <= region_data['end'])
                                    ]
                                    if not region_subset.empty:
                                        fitted_y = region_data['slope'] * region_subset['time'] + region_data['intercept']
                                        # Add glow effect with wider transparent line first
                                        fig.add_trace(go.Scatter(
                                            x=region_subset['time'],
                                            y=fitted_y,
                                            mode='lines',
                                            line=dict(width=st.session_state.get('plot_line_width', 5)*2, color=f'rgba({int(dataset_color[1:3], 16)}, {int(dataset_color[3:5], 16)}, {int(dataset_color[5:7], 16)}, 0.2)'),
                                            hoverinfo='skip',
                                            showlegend=False
                                        ))
                                        # Add main line
                                        fig.add_trace(go.Scatter(
                                            x=region_subset['time'],
                                            y=fitted_y,
                                            mode='lines',
                                            name=f'{ds_name} - {region_name} fit',
                                            line=dict(width=st.session_state.get('plot_line_width', 5), color=dataset_color),
                                            opacity=st.session_state.get('plot_line_opacity', 1.0),
                                            hovertemplate=f'{ds_name} fit: %{{y:.3f}}<extra></extra>',
                                            showlegend=False,  # Don't show fit lines in legend
                                            legendgroup=ds_name  # Group with dataset
                                        ))
        
        layout_config = dict(
            xaxis=dict(
                title="Time (s)",
                showgrid=False,
                showline=True,
                linecolor=line_color,
                linewidth=1,
                ticks='outside',
                tickcolor=line_color,
                tickfont=dict(size=14, color=text_color),
                title_font=dict(size=16, color=text_color)
            ),
            yaxis=dict(
                title="Absorbance (UA)",
                showgrid=False,
                showline=True,
                linecolor=line_color,
                linewidth=1,
                ticks='outside',
                tickcolor=line_color,
                tickfont=dict(size=14, color=text_color),
                title_font=dict(size=16, color=text_color)
            ),
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            showlegend=len(datasets_to_plot) > 1,
            legend=dict(
                x=1.02, y=1,
                bgcolor=paper_bg,
                bordercolor=line_color,
                borderwidth=1,
                font=dict(size=10, color=text_color)
            ),
            height=st.session_state.get('plot_height', 500),
            width=self._get_plot_width(),
            margin=dict(l=60, r=100, t=40, b=60),
            font=dict(family="Arial", color=text_color),
            hovermode='closest'
        )
        
        # Enable selection tools
        layout_config.update({
            'dragmode': 'select',
            'selectdirection': 'h',
            'newselection_mode': 'immediate',
            'activeselection_fillcolor': 'rgba(0, 255, 0, 0.3)',
            'activeselection_opacity': 0.3
        })
        
        fig.update_layout(layout_config)
        return fig
    
    def _calculate_slope(self, dataset_name: str, start_time: float, end_time: float, region_name: str, 
                        extinction_coeff: float = None, target_conc_unit: str = 'UA', 
                        target_time_unit: str = 's', enzyme_units: Dict = None) -> Dict:
        """Calculate slope for selected region."""
        if dataset_name not in st.session_state.data_sets:
            return {}
        
        data = st.session_state.data_sets[dataset_name]
        
        # Filter data for selected region
        region_data = data[
            (data['time'] >= start_time) & 
            (data['time'] <= end_time)
        ].copy()
        
        if len(region_data) < 2:
            return {}
        
        # Perform linear regression
        slope, intercept, r_value, _, std_err = stats.linregress(
            region_data['time'], region_data['product']
        )
        
        # Calculate additional statistics
        n = len(region_data)
        fitted_values = slope * region_data['time'] + intercept
        residuals = region_data['product'] - fitted_values
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Calculate mean absorbance and standard deviation for the region
        mean_absorbance = np.mean(region_data['product'])
        absorbance_std_dev = np.std(region_data['product'], ddof=1)  # Sample standard deviation
        
        
        # Convert slope using extinction coefficient if provided
        converted_slope = slope
        converted_std_err = std_err
        slope_units = 'UA/s'
        
        if extinction_coeff and extinction_coeff > 0 and target_conc_unit != 'UA':
            # Convert from absorbance/time to concentration/time
            # slope is in AU/s, convert to mM/s or μM/s
            if target_conc_unit == 'mM':
                converted_slope = slope / extinction_coeff  # mM/s (assuming extinction coeff in mM⁻¹cm⁻¹)
                converted_std_err = std_err / extinction_coeff  # Transform standard error too
                slope_units = 'mM/s'
            elif target_conc_unit == 'uM':
                converted_slope = (slope / extinction_coeff) * 1000  # μM/s
                converted_std_err = (std_err / extinction_coeff) * 1000  # Transform standard error too
                slope_units = 'μM/s'
        
        # Convert time units if requested
        if target_time_unit == 'min':
            converted_slope *= 60  # per minute
            converted_std_err *= 60  # Transform standard error too
            slope_units = slope_units.replace('/s', '/min')
        
        # Convert to enzyme units if requested
        enzyme_activity = None
        enzyme_activity_units = None
        if enzyme_units and enzyme_units.get('use_enzyme_units', False):
            if target_conc_unit in ['mM', 'uM'] and target_time_unit == 'min':
                # Convert concentration rate to enzyme activity
                # 1 U = 1 μmol/min, so we need to convert to μmol/min first
                if target_conc_unit == 'mM':
                    # mM/min * reaction_volume_mL * 1000 μM/mM = μmol/min
                    umol_per_min = converted_slope * enzyme_units['reaction_volume'] * 1000
                elif target_conc_unit == 'uM':
                    # μM/min * reaction_volume_mL = μmol/min  
                    umol_per_min = converted_slope * enzyme_units['reaction_volume']
                
                # Convert to specific activity
                if enzyme_units['enzyme_unit_type'] == 'U/mL':
                    enzyme_activity = umol_per_min / enzyme_units['enzyme_volume']
                    enzyme_activity_units = 'U/mL'
                else:  # U/mg
                    enzyme_activity = umol_per_min / enzyme_units['enzyme_mass']
                    enzyme_activity_units = 'U/mg'
        
        return {
            'region_name': region_name,
            'dataset': dataset_name,
            'start': start_time,
            'end': end_time,
            'slope': slope,  # Original slope in UA/s
            'converted_slope': converted_slope,  # Converted slope
            'slope_units': slope_units,
            'extinction_coeff': extinction_coeff,
            'intercept': intercept,
            'r_squared': r_value**2,
            'std_error': std_err,  # Original standard error
            'converted_std_error': converted_std_err,  # Converted standard error
            'enzyme_activity': enzyme_activity,  # Enzyme activity if calculated
            'enzyme_activity_units': enzyme_activity_units,  # Enzyme activity units
            'n_points': n,
            'rmse': rmse,
            'mean_absorbance': mean_absorbance,  # Mean absorbance in the region
            'absorbance_std_dev': absorbance_std_dev  # Standard deviation of absorbance
        }
    
    def _suggest_regions(self, dataset_name: str) -> Dict[str, Tuple[float, float]]:
        """Suggest regions based on data analysis."""
        if dataset_name not in st.session_state.data_sets:
            return {}
        
        data = st.session_state.data_sets[dataset_name]
        
        # Simple suggestions: start, middle, end thirds
        time_min = data['time'].min()
        time_max = data['time'].max()
        time_range = time_max - time_min
        
        suggestions = {
            'Region 1': (time_min, time_min + time_range/3),
            'Region 2': (time_min + time_range/3, time_min + 2*time_range/3),
            'Region 3': (time_min + 2*time_range/3, time_max)
        }
        
        return suggestions
    
    def _convert_units(self, value: float, from_unit: str, to_unit: str, unit_type: str = 'concentration') -> float:
        """Convert units using pint unit registry."""
        try:
            # Define unit mappings
            if unit_type == 'concentration':
                unit_mappings = {
                    'UA': 'dimensionless',
                    'uM': 'micromolar',
                    'mM': 'millimolar', 
                    'M': 'molar'
                }
            elif unit_type == 'time':
                unit_mappings = {
                    's': 'second',
                    'min': 'minute',
                    'h': 'hour'
                }
            else:
                return value
            
            # Handle dimensionless units (UA)
            if from_unit == 'UA' and to_unit == 'UA':
                return value
            elif from_unit == 'UA' or to_unit == 'UA':
                # For UA conversions, just return the value as-is since it's relative
                return value
            
            # Convert using pint
            from_pint = unit_mappings.get(from_unit, from_unit)
            to_pint = unit_mappings.get(to_unit, to_unit)
            
            quantity = self.ureg.Quantity(value, from_pint)
            converted = quantity.to(to_pint)
            return float(converted.magnitude)
            
        except Exception:
            # If conversion fails, return original value
            return value
    
    def _apply_unit_conversions(self, df: pd.DataFrame, time_unit_from: str, time_unit_to: str, 
                               conc_unit_from: str, conc_unit_to: str) -> pd.DataFrame:
        """Apply unit conversions to the dataframe."""
        converted_df = df.copy()
        
        # Convert time units if different
        if time_unit_from != time_unit_to and 'Start_Time' in converted_df.columns:
            converted_df['Start_Time'] = converted_df['Start_Time'].apply(
                lambda x: self._convert_units(float(x), time_unit_from, time_unit_to, 'time')
            )
            converted_df['End_Time'] = converted_df['End_Time'].apply(
                lambda x: self._convert_units(float(x), time_unit_from, time_unit_to, 'time')
            )
        
        # Convert concentration-related units
        if conc_unit_from != conc_unit_to and 'Slope' in converted_df.columns:
            # Slope units are concentration/time, so we need to convert the concentration part
            converted_df['Slope'] = converted_df['Slope'].apply(
                lambda x: self._convert_units(float(x), conc_unit_from, conc_unit_to, 'concentration')
            )
        
        return converted_df
    
    def _create_download_link(self, df: pd.DataFrame, filename: str, 
                             time_unit_from: str = 's', time_unit_to: str = 's',
                             conc_unit_from: str = 'UA', conc_unit_to: str = 'UA') -> str:
        """Create download link for DataFrame with unit conversion options."""
        # Apply unit conversions if requested
        if time_unit_from != time_unit_to or conc_unit_from != conc_unit_to:
            df = self._apply_unit_conversions(df, time_unit_from, time_unit_to, conc_unit_from, conc_unit_to)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {filename}</a>'
    
    def _generate_txt_content(self, dataset_name: str, original_filename: str) -> str:
        """Generate TXT file content for converted AvaSpec data."""
        if dataset_name not in st.session_state.data_sets:
            return ""
        
        data = st.session_state.data_sets[dataset_name]
        
        # Create TXT content with header
        txt_content = []
        txt_content.append("##################################################")
        txt_content.append("# DATA EXPORT AND TRANSFORMATION REPORT")
        txt_content.append("#")
        txt_content.append(f"# Original File: {original_filename}")
        txt_content.append(f"# Converted by: Kinetics Analyzer v1")
        txt_content.append(f"# Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        txt_content.append("#")
        txt_content.append("# Operations Performed:")
        txt_content.append("# - Column 1 (Time): Read from Excel Column B (milliseconds), divided by 1000.")
        txt_content.append("# - Column 2 (Absorbance): Read from Excel Column C, comma decimal converted to dot decimal.")
        txt_content.append("##################################################")
        txt_content.append("")
        txt_content.append("Time (s)\tUA")
        
        # Add data rows
        for _, row in data.iterrows():
            txt_content.append(f"{row['time']:.3f}\t{row['product']:.6f}")
        
        return '\n'.join(txt_content)
    
    def _create_zip_download(self, converted_files: Dict[str, str], filename: str = "avaspec_converted_files.zip") -> bytes:
        """Create a ZIP file containing all converted TXT files."""
        import zipfile
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for dataset_name, original_filename in converted_files.items():
                # Generate TXT content
                txt_content = self._generate_txt_content(dataset_name, original_filename)
                
                # Create filename for TXT file (replace .xlsx/.xls with .txt)
                txt_filename = Path(dataset_name).stem + '.txt'
                
                # Add to ZIP
                zip_file.writestr(txt_filename, txt_content)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def run(self):
        """Main application interface with Phase 1 improvements."""
        # Initialize theme preference
        if 'theme' not in st.session_state:
            st.session_state.theme = 'auto'
        
        # Theme-aware CSS
        if st.session_state.theme == 'dark':
            theme_vars = {
                'bg_primary': '#0e1117',
                'bg_secondary': '#262730',
                'text_primary': '#ffffff',
                'text_secondary': '#a3a8b8',
                'border_color': '#464853',
                'accent_color': '#ff6b6b'
            }
        elif st.session_state.theme == 'light':
            theme_vars = {
                'bg_primary': '#ffffff',
                'bg_secondary': '#f0f2f6',
                'text_primary': '#262730',
                'text_secondary': '#6c757d',
                'border_color': '#d1d5db',
                'accent_color': '#1f77b4'
            }
        else:  # auto/system
            theme_vars = {
                'bg_primary': 'var(--background-color)',
                'bg_secondary': 'var(--secondary-background-color)',
                'text_primary': 'var(--text-color)',
                'text_secondary': 'var(--text-color-secondary)',
                'border_color': 'var(--border-color)',
                'accent_color': 'var(--primary-color)'
            }
        
        # Apply theme CSS
        st.markdown(f"""
        <style>
        .main-header {{
            background: {theme_vars['bg_secondary']};
            color: {theme_vars['text_primary']};
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            text-align: center;
            border: 1px solid {theme_vars['border_color']};
        }}
        .main-header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 600;
        }}
        .main-header p {{
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.8;
        }}
        .metric-container {{
            background: {theme_vars['bg_secondary']};
            color: {theme_vars['text_primary']};
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid {theme_vars['border_color']};
            margin: 0.5rem 0;
        }}
        .upload-section {{
            background: {theme_vars['bg_secondary']};
            padding: 1.5rem;
            border-radius: 8px;
            border: 2px dashed {theme_vars['border_color']};
            margin: 1rem 0;
        }}
        .success-msg {{
            background: {theme_vars['accent_color']};
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin: 0.25rem 0;
        }}
        
        /* Subtle hover effects and animations */
        .stButton > button {{
            transition: all 0.2s ease-in-out;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Viridis gradient accent */
        .viridis-accent {{
            background: linear-gradient(45deg, #440154, #21908c, #fde725);
            height: 3px;
            border-radius: 1.5px;
            margin: 0.5rem 0;
        }}
        
        /* Coffee cup animation for loading states */
        @keyframes coffee-steam {{
            0% {{ opacity: 0.8; transform: translateY(0px) scale(1); }}
            50% {{ opacity: 0.4; transform: translateY(-10px) scale(1.1); }}
            100% {{ opacity: 0.8; transform: translateY(0px) scale(1); }}
        }}
        
        .coffee-loading {{
            animation: coffee-steam 2s ease-in-out infinite;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Professional header with same styling as original
        selected_quote = random.choice(self.quotes)
        st.markdown(f"""
        <div class="main-header">
            <h1>☕ Kinetics Analyzer</h1>
            <p>Interactive enzyme kinetics data analysis</p>
            <div class="viridis-accent"></div>
            <div style="font-style: italic; opacity: 0.7; font-size: 0.9rem; margin-top: 0.5rem;">
                "{selected_quote}"
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Theme selection
            theme_option = st.selectbox(
                "Theme",
                ["auto", "light", "dark"],
                index=0 if st.session_state.theme == 'auto' else (1 if st.session_state.theme == 'light' else 2),
                help="Choose interface theme"
            )
            
            if theme_option != st.session_state.theme:
                st.session_state.theme = theme_option
                st.rerun()
            
            # Enhanced color palette management with v1 improvements
            st.subheader("Color Settings")
            
            # Color palette mode selection
            color_palette_type = st.radio(
                "Color Palette",
                ["Viridis", "Nostalgic"],
                index=0 if st.session_state.get('color_palette_type', 'viridis') == 'viridis' else 1,
                help="Choose color palette style"
            )
            
            # Store palette type
            st.session_state.color_palette_type = 'viridis' if color_palette_type == "Viridis" else 'nostalgic'
            
            # Get current palette using enhanced color management
            current_palette = self.color_palette.get_palette(st.session_state.color_palette_type)
            
            # Color selection mode
            color_mode = st.radio(
                "Color Selection Mode",
                ["Automatic (Dynamic)", "Manual Selection"],
                index=0 if st.session_state.get('color_mode', 'auto') == 'auto' else 1,
                help="Choose automatic color selection or manually pick colors"
            )
            
            # Store color mode and track changes
            new_color_mode = 'auto' if color_mode == "Automatic (Dynamic)" else 'manual'
            
            # Track mode changes for color preservation
            if 'color_mode' in st.session_state and st.session_state.color_mode != new_color_mode:
                st.session_state.color_mode_previous = st.session_state.color_mode
            
            st.session_state.color_mode = new_color_mode
            
            # Manual color selection with enhanced palette management
            if st.session_state.color_mode == 'manual' and st.session_state.data_sets:
                st.write("**Select colors for each dataset:**")
                
                # Initialize manual colors if not exists
                if 'manual_colors' not in st.session_state:
                    st.session_state.manual_colors = {}
                
                # Simple color selector for each dataset
                for i, dataset_name in enumerate(st.session_state.data_sets.keys()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Display color options with descriptive names from current palette
                        selected_color_idx = st.selectbox(
                            f"{dataset_name}:",
                            range(len(current_palette)),
                            index=st.session_state.manual_colors.get(dataset_name, i % len(current_palette)),
                            format_func=lambda x: current_palette[x][1],  # Use descriptive name
                            key=f"color_select_{dataset_name}_{st.session_state.color_palette_type}"
                        )
                        st.session_state.manual_colors[dataset_name] = selected_color_idx
                    
                    with col2:
                        # Show color preview
                        color_hex = current_palette[selected_color_idx][0]  # Get hex color
                        st.markdown(f"""
                        <div style="
                            width: 30px; 
                            height: 30px; 
                            background-color: {color_hex}; 
                            border-radius: 4px; 
                            border: 1px solid #ccc;
                            margin-top: 24px;
                        "></div>
                        """, unsafe_allow_html=True)
            
            st.divider()
            
            # Enhanced file upload with AvaSpec integration
            st.header("Data Upload")
            
            uploaded_files = st.file_uploader(
                "Select data files",
                type=['txt', 'csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Data must contain time (seconds) and absorbance (UA) columns. Supports automatic AvaSpec Excel conversion."
            )
            
            # Enhanced memory management with batch AvaSpec detection
            if uploaded_files:
                uploaded_filenames = {uploaded_file.name for uploaded_file in uploaded_files}
                # Clean up removed datasets
                self.memory_manager.cleanup_removed_datasets(uploaded_filenames)
                
                # Batch AvaSpec detection for multiple Excel files
                new_files = [f for f in uploaded_files if f.name not in st.session_state.data_sets]
                excel_files = [f for f in new_files if Path(f.name).suffix.lower() in ['.xlsx', '.xls']]
                
                if len(excel_files) > 1:
                    # Check if ALL Excel files are AvaSpec
                    all_avaspec = all(self.avaspec_integration.is_avaspec_excel(f) for f in excel_files)
                    
                    if all_avaspec:
                        # Batch AvaSpec prompt for multiple files
                        batch_key = f"batch_avaspec_{hash(tuple(f.name for f in excel_files))}"
                        if batch_key not in st.session_state:
                            st.session_state[batch_key] = "pending"
                        
                        if st.session_state[batch_key] == "pending":
                            st.info(f"🔬 **Excel Files Detected**: {len(excel_files)} files")
                            st.write("**Are ALL these Excel files directly exported from AvaSpec spectrometer?**")
                            st.caption(f"Files: {', '.join([f.name for f in excel_files])}")
                            
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                if st.button("Yes, convert.", key=f"batch_convert_{batch_key}"):
                                    st.session_state[batch_key] = "convert_all"
                                    st.rerun()
                            with col2:
                                if st.button("No", key=f"batch_standard_{batch_key}"):
                                    st.session_state[batch_key] = "standard_all"
                                    st.rerun()
                            with col3:
                                if st.button("🔍", key=f"batch_individual_{batch_key}"):
                                    st.session_state[batch_key] = "individual"
                                    st.rerun()
                            
                            # Don't process files yet, wait for user decision
                            return
                        
                        elif st.session_state[batch_key] == "convert_all":
                            # Batch convert all AvaSpec files
                            st.subheader("Loading Files")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, uploaded_file in enumerate(excel_files):
                                progress = (i + 1) / len(excel_files)
                                progress_bar.progress(progress)
                                status_text.text(f"Converting AvaSpec file {i + 1} of {len(excel_files)}: {uploaded_file.name}")
                                
                                with st.spinner(f"Converting {uploaded_file.name}..."):
                                    data = self.avaspec_integration.convert_avaspec_excel(uploaded_file)
                                    if data is not None:
                                        st.session_state.data_sets[uploaded_file.name] = data
                                        # Track converted file for download option
                                        st.session_state.avaspec_converted_files[uploaded_file.name] = uploaded_file.name
                                        st.success(f"✅ **AvaSpec Converted**: {uploaded_file.name} ({len(data)} points)")
                            
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"🎉 **Batch Conversion Complete**: {len(excel_files)} AvaSpec files processed")
                            
                        elif st.session_state[batch_key] == "standard_all":
                            # Load all as standard Excel
                            st.subheader("Loading Files")
                            for uploaded_file in excel_files:
                                with st.spinner(f"Loading {uploaded_file.name}..."):
                                    data = self._load_excel_file(uploaded_file)
                                    if data is not None:
                                        st.session_state.data_sets[uploaded_file.name] = data
                                        st.success(f"📁 Loaded: {uploaded_file.name}")
                
                # Process remaining files (non-Excel and individual handling)
                st.subheader("Loaded Files")
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.data_sets:
                        # Skip Excel files that are part of batch processing (unless individual handling)
                        if (len(excel_files) > 1 and uploaded_file in excel_files and 
                            f"batch_avaspec_{hash(tuple(f.name for f in excel_files))}" in st.session_state and
                            st.session_state[f"batch_avaspec_{hash(tuple(f.name for f in excel_files))}"] != "individual"):
                            continue
                        
                        with st.spinner(f"Loading {uploaded_file.name}..."):
                            data = self._load_data_file(uploaded_file)
                            if data is not None:
                                st.session_state.data_sets[uploaded_file.name] = data
                                st.success(f"📁 Loaded: {uploaded_file.name}")
                            # Error already handled by error handler if data is None
                    else:
                        st.info(f"Already loaded: {uploaded_file.name}")
            else:
                # Clear session state when no files
                if st.session_state.data_sets:
                    st.session_state.data_sets = {}
                    st.session_state.regions = {}
                    st.session_state.avaspec_converted_files = {}
                    if hasattr(st.session_state, 'manual_colors'):
                        st.session_state.manual_colors = {}
            
            st.divider()
            
            # Plot customization section (same as original)
            st.subheader("Plot Appearance")
            
            # Initialize plot customization settings in session state
            if 'plot_marker_size' not in st.session_state:
                st.session_state.plot_marker_size = 3
            if 'plot_marker_opacity' not in st.session_state:
                st.session_state.plot_marker_opacity = 1.0
            if 'plot_line_width' not in st.session_state:
                st.session_state.plot_line_width = 5
            if 'plot_line_opacity' not in st.session_state:
                st.session_state.plot_line_opacity = 0.7
            if 'plot_height' not in st.session_state:
                st.session_state.plot_height = 500
            if 'plot_width_mode' not in st.session_state:
                st.session_state.plot_width_mode = 'container'
            
            # Data points customization
            st.write("**Data Points:**")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.plot_marker_size = st.slider(
                    "Thickness",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.plot_marker_size,
                    step=1,
                    key="marker_size_slider",
                    help="Size of data point markers"
                )
            with col2:
                st.session_state.plot_marker_opacity = st.slider(
                    "Opacity",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.plot_marker_opacity,
                    step=0.1,
                    key="marker_opacity_slider",
                    help="Opacity of data points (1.0 = opaque, 0.1 = very transparent)"
                )
            
            # Fitted lines customization
            st.write("**Fitted Lines:**")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.plot_line_width = st.slider(
                    "Thickness",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.plot_line_width,
                    step=1,
                    key="line_width_slider",
                    help="Width of fitted regression lines"
                )
            with col2:
                st.session_state.plot_line_opacity = st.slider(
                    "Opacity",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.plot_line_opacity,
                    step=0.1,
                    key="line_opacity_slider",
                    help="Opacity of fitted lines (1.0 = opaque, 0.1 = very transparent)"
                )
            
            # Plot size customization
            st.write("**Plot Size:**")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.plot_height = st.slider(
                    "Height (px)",
                    min_value=300,
                    max_value=1000,
                    value=st.session_state.plot_height,
                    step=50,
                    key="plot_height_slider",
                    help="Height of the plot in pixels"
                )
            with col2:
                plot_width_options = ["Container Width", "Small (600px)", "Medium (800px)", "Large (1000px)", "Extra Large (1200px)"]
                width_mode_index = ["container", "small", "medium", "large", "xlarge"].index(st.session_state.plot_width_mode)
                
                selected_width = st.selectbox(
                    "Width",
                    plot_width_options,
                    index=width_mode_index,
                    key="plot_width_selector",
                    help="Width of the plot"
                )
                
                # Update session state based on selection
                width_mapping = {
                    "Container Width": "container",
                    "Small (600px)": "small", 
                    "Medium (800px)": "medium",
                    "Large (1000px)": "large",
                    "Extra Large (1200px)": "xlarge"
                }
                st.session_state.plot_width_mode = width_mapping[selected_width]
            
            # Reset to defaults button
            if st.button("Reset to Defaults", type="secondary", use_container_width=True):
                st.session_state.plot_marker_size = 3
                st.session_state.plot_marker_opacity = 1.0
                st.session_state.plot_line_width = 5
                st.session_state.plot_line_opacity = 0.7
                st.session_state.plot_height = 500
                st.session_state.plot_width_mode = 'container'
                st.rerun()
            
            st.divider()
            
            # Statistics section
            if st.session_state.data_sets:
                st.header("Dataset Statistics")
                st.metric("Datasets loaded", len(st.session_state.data_sets))
                total_regions = sum(len(regions) for regions in st.session_state.regions.values())
                if total_regions > 0:
                    st.metric("Regions analyzed", total_regions)
            
            st.divider()
            
            # AvaSpec converted files download section
            if (hasattr(st.session_state, 'avaspec_converted_files') and 
                st.session_state.avaspec_converted_files):
                st.header("🔬 AvaSpec Downloads")
                
                converted_count = len(st.session_state.avaspec_converted_files)
                st.metric("AvaSpec files converted", converted_count)
                
                st.write("**Download converted TXT files:**")
                st.caption("Get all converted AvaSpec files as TXT format in a ZIP archive.")
                
                # Single consolidated download button
                try:
                    # Create ZIP file
                    zip_data = self._create_zip_download(st.session_state.avaspec_converted_files)
                    
                    # Provide direct download
                    st.download_button(
                        label="📥 Download All as TXT Files (ZIP)",
                        data=zip_data,
                        file_name="avaspec_converted_files.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="primary",
                        help=f"Download {converted_count} converted AvaSpec files as TXT format in a ZIP file"
                    )
                    
                except Exception as e:
                    st.error(f"❌ **Download Error**: {str(e)}")
                    logger.error(f"Error creating AvaSpec download ZIP: {str(e)}")
                
                # Show list of converted files
                with st.expander("📋 View Converted Files", expanded=False):
                    for i, (dataset_name, original_name) in enumerate(st.session_state.avaspec_converted_files.items(), 1):
                        txt_filename = Path(dataset_name).stem + '.txt'
                        st.write(f"{i}. **{original_name}** → `{txt_filename}`")
            
        # Main content area
        if not st.session_state.data_sets:
            st.info("Please upload data files using the sidebar to begin analysis.")
            return
        
        # Dataset selection and overlay options
        st.subheader("Dataset Analysis")
        
        selected_dataset = st.selectbox(
            "Select Dataset for Analysis",
            list(st.session_state.data_sets.keys()),
            help="Choose a dataset to analyze and visualize"
        )
        
        # Overlay selection section
        st.subheader("Overlay Options")
        col1, col2 = st.columns([1, 2])
        with col1:
            enable_overlay = st.checkbox(
                "Enable Overlay Mode",
                value=False,
                help="Show multiple datasets on the same plot for comparison"
            )
        
        # Initialize overlay_datasets based on enable_overlay
        if enable_overlay:
            with col2:
                overlay_datasets = st.multiselect(
                    "Select Datasets to Overlay",
                    options=list(st.session_state.data_sets.keys()),
                    default=list(st.session_state.data_sets.keys()),
                    help="Choose which datasets to display on the overlay plot"
                )
            show_overlay = len(overlay_datasets) > 0
        else:
            overlay_datasets = []
            show_overlay = False
        
        # Phase 4: Safe Batch Analysis Section (Overlay Mode Only)
        if len(st.session_state.data_sets) > 1 and show_overlay:
            st.subheader("Batch Analysis")
            
            with st.expander("Batch Process Multiple Datasets", expanded=False):
                st.info("**Batch Analysis**: Analyze the same time region across multiple datasets simultaneously.")
                
                # Dataset selection for batch
                batch_datasets = st.multiselect(
                    "Select Datasets for Batch Analysis",
                    options=list(st.session_state.data_sets.keys()),
                    default=[],
                    help="Choose datasets to analyze in batch mode"
                )
                
                if batch_datasets and len(batch_datasets) > 1:
                    # Check for interactive selection
                    has_interactive_selection = ('selected_start' in st.session_state and 
                                               'selected_end' in st.session_state)
                    
                    # Time frame selection with manual modification capability
                    if has_interactive_selection:
                        st.success(f"✅ **Plot Selection Available**: {st.session_state.selected_start:.2f}s - {st.session_state.selected_end:.2f}s")
                        
                        # Get data ranges for validation
                        all_data_min = min([st.session_state.data_sets[ds]['time'].min() for ds in batch_datasets])
                        all_data_max = max([st.session_state.data_sets[ds]['time'].max() for ds in batch_datasets])
                        
                        # Manual time range inputs (like normal mode)
                        col1, col2 = st.columns(2)
                        with col1:
                            batch_start = st.number_input(
                                "Batch Start Time (s)",
                                min_value=float(all_data_min),
                                max_value=float(all_data_max),
                                value=float(st.session_state.selected_start),
                                step=0.1,
                                format="%.2f",
                                key="batch_start_time",
                                help="Start time for batch analysis (auto-filled from plot selection)"
                            )
                        with col2:
                            batch_end = st.number_input(
                                "Batch End Time (s)",
                                min_value=float(max(batch_start, all_data_min)),
                                max_value=float(all_data_max),
                                value=float(st.session_state.selected_end),
                                step=0.1,
                                format="%.2f",
                                key="batch_end_time",
                                help="End time for batch analysis (auto-filled from plot selection)"
                            )
                        
                    else:
                        # No selection available - show instruction
                        st.warning("📍 **No Plot Selection Found**")
                        st.info("**Please make a selection on the plot above to set the batch analysis time range.**")
                        st.caption("Use the box select tool in the plot toolbar to drag across your desired time range.")
                        
                        # Don't show batch analysis options without selection, but continue with rest of UI
                        st.info("💡 **Batch analysis will be available once you make a plot selection.**")
                        # Don't return here - let the rest of the UI continue
                    
                    # Only show batch analysis parameters if there's a selection
                    if has_interactive_selection:
                        # Batch analysis parameters (simplified)
                        st.write("**Batch Analysis Parameters:**")
                        batch_col1, batch_col2 = st.columns(2)
                        
                        with batch_col1:
                            batch_use_extinction = st.checkbox(
                                "Use Extinction Coefficient (Batch)",
                                value=False,
                                key="batch_extinction",
                                help="Apply extinction coefficient to all datasets in batch"
                            )
                            
                            batch_extinction_coeff = st.number_input(
                                "Extinction Coefficient (Batch)",
                                min_value=0.0,
                                value=1.0,
                                step=0.1,
                                format="%.3f",
                                disabled=not batch_use_extinction,
                                key="batch_extinction_value",
                                help="Extinction coefficient for batch analysis"
                            )
                        
                        with batch_col2:
                            batch_conc_unit = st.selectbox(
                                "Concentration Unit (Batch)",
                                ["UA", "mM", "uM"],
                                index=0,
                                disabled=not batch_use_extinction,
                                key="batch_conc_unit",
                                help="Target concentration unit for batch analysis"
                            )
                            
                            batch_time_unit = st.selectbox(
                                "Time Unit (Batch)",
                                ["s", "min"],
                                index=0,
                                key="batch_time_unit",
                                help="Time unit for batch analysis"
                            )
                        
                        # Run batch analysis button
                        if st.button("Run Batch Analysis", type="primary", use_container_width=True):
                            # Prepare batch analysis parameters
                            batch_params = {
                                'extinction_coeff': batch_extinction_coeff if batch_use_extinction else None,
                                'target_conc_unit': batch_conc_unit if batch_use_extinction else 'UA',
                                'target_time_unit': batch_time_unit,
                                'enzyme_units': None  # Can be extended later
                            }
                            
                            # Run safe batch analysis
                            batch_results = self.batch_analyzer.perform_batch_analysis(
                                batch_datasets,
                                (batch_start, batch_end),
                                batch_params
                            )
                            
                            if batch_results:
                                st.rerun()  # Refresh to show new regions
                
                elif batch_datasets and len(batch_datasets) == 1:
                    st.info("💡 Select 2 or more datasets for batch analysis, or use the regular region analysis above.")
                elif len(batch_datasets) == 0:
                    st.info("💡 Choose datasets above to enable batch analysis options.")
        
        elif len(st.session_state.data_sets) > 1 and not show_overlay:
            st.subheader("Batch Analysis")
            st.info("💡 Enable overlay mode above to access batch processing features.")
        
        if selected_dataset:
            # Display plot
            st.subheader("Interactive Data Visualization")
            
            # Clear selection option
            if 'selected_start' in st.session_state or 'selected_end' in st.session_state:
                if st.button("Clear Selection", type="secondary"):
                    if 'selected_start' in st.session_state:
                        del st.session_state.selected_start
                    if 'selected_end' in st.session_state:
                        del st.session_state.selected_end
                    st.rerun()
            
            # Configure plot toolbar
            config = {
                'modeBarButtonsToRemove': ['lasso2d'],
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['select2d'],
                'showTips': True,
                'scrollZoom': True
            }
            
            # Create interactive plot with enhanced color management
            if show_overlay and overlay_datasets:
                fig = self._create_interactive_plot(show_regions=True, overlay_datasets=overlay_datasets)
                plot_key = "overlay_plot"
            else:
                fig = self._create_interactive_plot(selected_dataset, show_regions=True)
                plot_key = "single_plot"
            
            # Use selection events for both single and overlay modes
            if show_overlay and overlay_datasets:
                datasets_hash = hash(tuple(sorted(overlay_datasets)))
                unique_plot_key = f"{plot_key}_{datasets_hash}"
            else:
                unique_plot_key = f"{plot_key}_{selected_dataset}"
            
            # Determine if we should use container width or fixed width
            use_container_width = st.session_state.get('plot_width_mode', 'container') == 'container'
            
            plot_selection = st.plotly_chart(fig, use_container_width=use_container_width, key=unique_plot_key, on_select="rerun", config=config)
            
            # Handle plot selection events
            if plot_selection and 'selection' in plot_selection and plot_selection['selection']:
                selection_data = plot_selection['selection']
                
                # Extract x-range from box selection
                if 'box' in selection_data and selection_data['box']:
                    boxes = selection_data['box']
                    if len(boxes) > 0 and 'x' in boxes[0] and len(boxes[0]['x']) >= 2:
                        x_range = boxes[0]['x']
                        # Update session state with selection
                        st.session_state.selected_start = float(min(x_range))
                        st.session_state.selected_end = float(max(x_range))
                        st.session_state.selected_for_dataset = selected_dataset
                        st.session_state.selection_counter = st.session_state.get('selection_counter', 0) + 1
                        st.rerun()
            
            # Display current selection
            if 'selected_start' in st.session_state and 'selected_end' in st.session_state:
                target_dataset = st.session_state.get('selected_for_dataset', 'Unknown')
                if target_dataset == selected_dataset:
                    st.info(f"Selected for {selected_dataset}: {st.session_state.selected_start:.2f}s to {st.session_state.selected_end:.2f}s")
                else:
                    st.warning(f"Selection is for {target_dataset}, currently viewing {selected_dataset}. Switch datasets to use selection.")
            
            # Region management with enhanced interface
            st.subheader("Region Analysis")
            
            # Initialize regions for this dataset if not exists
            if selected_dataset not in st.session_state.regions:
                st.session_state.regions[selected_dataset] = {}
            
            # Add new region section (keeping same interface as original)
            with st.expander("Add New Region", expanded=True):
                # Time range selection
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    data_min = float(st.session_state.data_sets[selected_dataset]['time'].min())
                    data_max = float(st.session_state.data_sets[selected_dataset]['time'].max())
                    
                    selection_counter = st.session_state.get('selection_counter', 0)
                    has_selection = ('selected_start' in st.session_state and 
                                   st.session_state.get('selected_for_dataset') == selected_dataset)
                    
                    if has_selection:
                        default_start = st.session_state.selected_start
                    else:
                        default_start = data_min
                    
                    new_start = st.number_input(
                        "Start (s)",
                        min_value=data_min,
                        max_value=data_max,
                        value=float(default_start),
                        step=0.1,
                        format="%.2f",
                        key=f"start_time_{selection_counter}",
                        help="Start time of the region (auto-filled from plot selection)"
                    )
                
                with col2:
                    if has_selection:
                        default_end = max(st.session_state.selected_end, new_start)
                    else:
                        default_end = data_max
                    
                    new_end = st.number_input(
                        "End (s)",
                        min_value=new_start,
                        max_value=data_max,
                        value=float(default_end),
                        step=0.1,
                        format="%.2f",
                        key=f"end_time_{selection_counter}",
                        help="End time of the region (auto-filled from plot selection)"
                    )
                
                with col3:
                    region_name = st.text_input(
                        "Region Name", 
                        value=f"Region_{st.session_state.region_counter + 1}",
                        key="new_region_name"
                    )
                
                # Unit conversion sections (same as original implementation)
                st.subheader("Unit Conversions (Optional)")
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    use_extinction = st.checkbox(
                        "Use Extinction Coefficient",
                        value=False,
                        help="Convert absorbance slope to concentration slope"
                    )
                
                with col2:
                    extinction_coeff = st.number_input(
                        "Extinction Coefficient (mM⁻¹cm⁻¹)",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        format="%.3f",
                        disabled=not use_extinction,
                        help="Extinction coefficient in mM⁻¹cm⁻¹"
                    )
                
                with col3:
                    target_conc_unit = st.selectbox(
                        "Concentration Unit",
                        ["UA", "mM", "uM"],
                        index=2,
                        disabled=not use_extinction,
                        help="Target concentration unit for slope"
                    )
                
                with col4:
                    target_time_unit = st.selectbox(
                        "Time Unit",
                        ["s", "min"],
                        index=0,
                        help="Time unit for slope calculation"
                    )
                
                # Enzyme units section (same as original)
                st.subheader("Enzyme Unit Conversion")
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    use_enzyme_units = st.checkbox(
                        "Convert to Enzyme Units",
                        value=False,
                        help="Convert concentration rate to enzyme activity units (U/mL or U/mg)"
                    )
                
                with col2:
                    enzyme_unit_type = st.selectbox(
                        "Enzyme Unit Type",
                        ["U/mL", "U/mg"],
                        index=0,
                        disabled=not use_enzyme_units,
                        help="Type of enzyme activity units"
                    )
                
                with col3:
                    reaction_volume = st.number_input(
                        "Reaction Volume (mL)",
                        min_value=0.001,
                        value=1.0,
                        step=0.001,
                        format="%.3f",
                        disabled=not use_enzyme_units,
                        help="Total reaction volume in mL"
                    )
                
                with col4:
                    if enzyme_unit_type == "U/mL":
                        enzyme_volume = st.number_input(
                            "Enzyme Volume (mL)",
                            min_value=0.001,
                            value=0.010,
                            step=0.001,
                            format="%.3f",
                            disabled=not use_enzyme_units,
                            help="Volume of enzyme added in mL"
                        )
                    else:  # U/mg
                        enzyme_mass = st.number_input(
                            "Enzyme Mass (mg)",
                            min_value=0.001,
                            value=1.0,
                            step=0.001,
                            format="%.3f",
                            disabled=not use_enzyme_units,
                            help="Mass of enzyme in mg"
                        )
                
                # Add region button
                if st.button("Add Region", type="primary", use_container_width=True):
                    if region_name and region_name not in st.session_state.regions[selected_dataset]:
                        ext_coeff = extinction_coeff if use_extinction else None
                        conc_unit = target_conc_unit if use_extinction else 'UA'
                        
                        # Prepare enzyme units parameters
                        enzyme_units_params = None
                        if use_enzyme_units:
                            enzyme_units_params = {
                                'use_enzyme_units': True,
                                'enzyme_unit_type': enzyme_unit_type,
                                'reaction_volume': reaction_volume
                            }
                            if enzyme_unit_type == "U/mL":
                                enzyme_units_params['enzyme_volume'] = enzyme_volume
                            else:  # U/mg
                                enzyme_units_params['enzyme_mass'] = enzyme_mass
                        
                        # Add loading animation
                        with st.spinner('☕ Analyzing kinetics...'):
                            result = self._calculate_slope(
                                selected_dataset, new_start, new_end, region_name,
                                ext_coeff, conc_unit, target_time_unit, enzyme_units_params
                            )
                        
                        if result:
                            st.session_state.regions[selected_dataset][region_name] = result
                            st.session_state.region_counter += 1
                            
                            # Clear the selection after successful region addition
                            if 'selected_start' in st.session_state:
                                del st.session_state.selected_start
                            if 'selected_end' in st.session_state:
                                del st.session_state.selected_end
                            
                            st.success(f"✅ Added region: {region_name}")
                            st.rerun()
                    else:
                        st.error("Region name already exists or is empty")
            
            # Show existing regions (same interface as original)
            if st.session_state.regions[selected_dataset]:
                st.subheader("📊 Analysis Results")
                
                # Display regions with delete buttons
                for i, (region_name, region_data) in enumerate(st.session_state.regions[selected_dataset].items()):
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        
                        with col1:
                            # Format region display
                            if 'converted_slope' in region_data and 'slope_units' in region_data:
                                if region_data['slope_units'] in ['UA/s', 'UA/min']:
                                    slope_display = f"{region_data['converted_slope']:.3e} {region_data['slope_units']}"
                                else:
                                    slope_display = f"{region_data['converted_slope']:.6f} {region_data['slope_units']}"
                            else:
                                slope_display = f"{region_data['slope']:.3e} UA/s"
                            
                            # R² feedback
                            r_squared_val = region_data['r_squared']
                            if r_squared_val >= 0.99:
                                r_squared_icon = "🟢"
                            elif r_squared_val < 0.98:
                                r_squared_icon = "🔴"
                            else:
                                r_squared_icon = ""
                            
                            # Additional info
                            enzyme_display = ""
                            if region_data.get('enzyme_activity') is not None:
                                enzyme_display = f" | Enzyme Activity: {region_data['enzyme_activity']:.6f} {region_data['enzyme_activity_units']}"
                            
                            mean_abs_display = ""
                            if 'mean_absorbance' in region_data and 'absorbance_std_dev' in region_data:
                                mean_abs_display = f" | Mean Abs: {region_data['mean_absorbance']:.4f} ± {region_data['absorbance_std_dev']:.4f} UA"
                            
                            st.info(f"**{region_name}** | {region_data['start']:.2f}s - {region_data['end']:.2f}s | "
                                   f"Slope: {slope_display} | R²: {r_squared_icon}{' ' if r_squared_icon else ''}{region_data['r_squared']:.4f} | "
                                   f"Points: {region_data['n_points']}{mean_abs_display}{enzyme_display}")
                        
                        with col2:
                            if st.button("🗑️", key=f"delete_{selected_dataset}_{region_name}", 
                                       help=f"Delete {region_name}", type="secondary"):
                                del st.session_state.regions[selected_dataset][region_name]
                                st.success(f"🗑️ Deleted region: {region_name}")
                                st.rerun()
        
        # Batch Analysis Results Section
        # Collect all batch regions across all datasets
        batch_regions = {}
        for dataset_name, regions_dict in st.session_state.regions.items():
            for region_name, region_data in regions_dict.items():
                if region_name.startswith('Batch_'):
                    # Extract batch number from region name (e.g., "Batch_1_1.xlsx" -> "1")
                    batch_parts = region_name.split('_')
                    if len(batch_parts) >= 2:
                        batch_number = batch_parts[1]
                        if batch_number not in batch_regions:
                            batch_regions[batch_number] = []
                        batch_regions[batch_number].append((region_name, region_data, dataset_name))
        
        # Display batch results if any exist
        if batch_regions:
            st.subheader("Batch Analysis Results")
            
            for batch_number in sorted(batch_regions.keys()):
                batch_data = batch_regions[batch_number]
                st.write(f"**Batch {batch_number} Results** ({len(batch_data)} datasets processed):")
                
                for region_name, region_data, dataset_name in batch_data:
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        
                        with col1:
                            # Format batch region display to match individual results exactly
                            if 'converted_slope' in region_data and 'slope_units' in region_data:
                                if region_data['slope_units'] in ['UA/s', 'UA/min']:
                                    slope_display = f"{region_data['converted_slope']:.3e} {region_data['slope_units']}"
                                else:
                                    slope_display = f"{region_data['converted_slope']:.6f} {region_data['slope_units']}"
                            else:
                                slope_display = f"{region_data['slope']:.3e} UA/s"
                            
                            # R² feedback
                            r_squared_val = region_data['r_squared']
                            if r_squared_val >= 0.99:
                                r_squared_icon = "🟢"
                            elif r_squared_val < 0.98:
                                r_squared_icon = "🔴"
                            else:
                                r_squared_icon = ""
                            
                            # Additional info
                            enzyme_display = ""
                            if region_data.get('enzyme_activity') is not None:
                                enzyme_display = f" | Enzyme Activity: {region_data['enzyme_activity']:.6f} {region_data['enzyme_activity_units']}"
                            
                            mean_abs_display = ""
                            if 'mean_absorbance' in region_data and 'absorbance_std_dev' in region_data:
                                mean_abs_display = f" | Mean Abs: {region_data['mean_absorbance']:.4f} ± {region_data['absorbance_std_dev']:.4f} UA"
                            
                            # Format
                            st.info(f"**{region_name}** | {region_data['start']:.2f}s - {region_data['end']:.2f}s | "
                                   f"Slope: {slope_display} | R²: {r_squared_icon}{' ' if r_squared_icon else ''}{region_data['r_squared']:.4f} | "
                                   f"Points: {region_data['n_points']}{mean_abs_display}{enzyme_display}")
                        
                        with col2:
                            if st.button("🗑️", key=f"delete_batch_{dataset_name}_{region_name}", 
                                       help=f"Delete {region_name}", type="secondary"):
                                del st.session_state.regions[dataset_name][region_name]
                                st.success(f"🗑️ Deleted batch region: {region_name}")
                                st.rerun()
                
                st.divider()  # Separate different batches
        
        # Global results and export (same as original implementation)
        all_regions = []
        for dataset_name, regions_dict in st.session_state.regions.items():
            for region_name, region_data in regions_dict.items():
                all_regions.append(region_data)
        
        if all_regions:
            st.divider()
            st.subheader("Export All Results")
            
            # Create comprehensive results table
            results_data = []
            for region in all_regions:
                # Format data for export
                if 'converted_slope' in region and 'slope_units' in region:
                    if region['slope_units'] in ['UA/s', 'UA/min']:
                        slope_display = f"{region['converted_slope']:.3e}"
                    else:
                        slope_display = f"{region['converted_slope']:.6f}"
                    slope_units = region['slope_units']
                else:
                    slope_display = f"{region['slope']:.3e}"
                    slope_units = 'UA/s'
                
                converted_std_err = region.get('converted_std_error', region['std_error'])
                
                result_entry = {
                    'Dataset': region['dataset'],
                    'Region': region['region_name'],
                    'Start_Time': f"{region['start']:.2f}",
                    'End_Time': f"{region['end']:.2f}",
                    'Slope': slope_display,
                    'Slope_Units': slope_units,
                    'Std_Error_Converted': f"{converted_std_err:.3e}",
                    'Original_Slope_UA_s': f"{region['slope']:.3e}",
                    'Original_Std_Error': f"{region['std_error']:.3e}",
                    'Extinction_Coeff': region.get('extinction_coeff', 'N/A'),
                    'R_Squared': f"{region['r_squared']:.4f}",
                    'Number_of_Points': region['n_points'],
                    'RMSE': f"{region['rmse']:.4f}",
                    'Mean_Absorbance': f"{region.get('mean_absorbance', 0):.4f}",
                    'Absorbance_Std_Dev': f"{region.get('absorbance_std_dev', 0):.4f}"
                }
                
                # Add enzyme activity if available
                if region.get('enzyme_activity') is not None:
                    result_entry['Enzyme_Activity'] = f"{region['enzyme_activity']:.6f}"
                    result_entry['Enzyme_Activity_Units'] = region['enzyme_activity_units']
                
                results_data.append(result_entry)
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Export options
                st.subheader("Export Options")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Select optional data to export:**")
                    st.caption("Basic information, statistical data (R², Number of Points), and RMSE are always included")
                    
                    available_columns = list(results_df.columns)
                    mandatory_columns = {
                        'Dataset', 'Region', 'Start_Time', 'End_Time', 
                        'R_Squared', 'Number_of_Points', 'RMSE'
                    }
                    
                    export_cols = {}
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        export_cols['Slope'] = st.checkbox(
                            "Transformed Slope",
                            value=True,
                            key="export_transformed_slope",
                            help="Includes slope values, units, and standard error"
                        )
                        
                        export_cols['Original_Slope_UA_s'] = st.checkbox(
                            "Original Slope",
                            value=True,
                            key="export_original_slope",
                            help="Includes original slope (UA/s) and standard error"
                        )
                    
                    with col_b:
                        export_cols['Mean_Absorbance'] = st.checkbox(
                            "Mean Absorbance",
                            value=True,
                            key="export_mean_absorbance",
                            help="Mean absorbance value and standard deviation for each region"
                        )
                        
                        export_cols['Extinction_Coeff'] = st.checkbox(
                            "Extinction Coefficient",
                            value=False,
                            key="export_extinction_coeff",
                            help="Extinction coefficient used in calculations"
                        )
                        
                        if 'Enzyme_Activity' in available_columns:
                            export_cols['Enzyme_Activity'] = st.checkbox(
                                "Enzyme Activity",
                                value=True,
                                key="export_enzyme_activity",
                                help="Includes enzyme activity values and units"
                            )
                
                with col2:
                    st.write("**Export Settings:**")
                    
                    custom_filename = st.text_input(
                        "Filename (without extension)",
                        value="kinetics_analysis_results",
                        help="Enter filename without extension"
                    )
                    
                    export_format = st.selectbox(
                        "Export Format",
                        ["Excel (.xlsx)", "CSV (.csv)"],
                        index=0,
                        help="File format for export"
                    )
                
                # Handle column dependencies and create download
                selected_columns = [col for col, selected in export_cols.items() if selected]
                final_columns = set(mandatory_columns.intersection(available_columns))
                
                # Add selected columns and dependencies
                for col in selected_columns:
                    final_columns.add(col)
                    
                    if col == 'Slope':
                        if 'Slope_Units' in available_columns:
                            final_columns.add('Slope_Units')
                        if 'Std_Error_Converted' in available_columns:
                            final_columns.add('Std_Error_Converted')
                    elif col == 'Original_Slope_UA_s':
                        if 'Original_Std_Error' in available_columns:
                            final_columns.add('Original_Std_Error')
                    elif col == 'Mean_Absorbance':
                        if 'Absorbance_Std_Dev' in available_columns:
                            final_columns.add('Absorbance_Std_Dev')
                    elif col == 'Enzyme_Activity':
                        if 'Enzyme_Activity_Units' in available_columns:
                            final_columns.add('Enzyme_Activity_Units')
                
                final_columns_list = [col for col in available_columns if col in final_columns]
                
                if final_columns_list:
                    filtered_df = results_df[final_columns_list]
                    
                    # Create file data based on format
                    if export_format == "CSV (.csv)":
                        file_extension = ".csv"
                        mime_type = "text/csv"
                    else:  # Excel
                        file_extension = ".xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    
                    filename = f"{custom_filename}{file_extension}"
                    
                    output = io.BytesIO()
                    if export_format == "CSV (.csv)":
                        csv_string = filtered_df.to_csv(index=False)
                        output.write(csv_string.encode('utf-8'))
                    else:  # Excel
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            filtered_df.to_excel(writer, index=False, sheet_name='Results')
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=output.getvalue(),
                        file_name=filename,
                        mime=mime_type,
                        use_container_width=True,
                        type="primary"
                    )
                    
                    # Show auto-included columns info
                    auto_included = [col for col in final_columns_list if col not in selected_columns]
                    if auto_included:
                        st.info(f"Auto-included: {', '.join(auto_included)}")
                    
                    st.success(f"Ready to download {len(filtered_df)} rows with {len(final_columns_list)} columns")
                else:
                    st.warning("Please select at least one optional column to export")

# Run the application
if __name__ == "__main__":
    app = EnzymeKineticsStreamlit()
    app.run()
