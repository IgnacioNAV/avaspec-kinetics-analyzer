#!/usr/bin/env python3
"""
Enzyme Kinetics Analyzer - Streamlit Web Application
Interactive web-based tool for enzyme kinetics data analysis with region selection.
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
from typing import Dict, Tuple, List, Optional, Union
import base64

# Set page configuration
st.set_page_config(
    page_title="Enzyme Kinetics Analyzer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnzymeKineticsStreamlit:
    """
    Main application class for enzyme kinetics analysis using Streamlit and Plotly.
    """
    
    def __init__(self):
        # Initialize unit registry
        self.ureg = pint.UnitRegistry()
        self._define_custom_units()
        
        # Initialize session state
        if 'data_sets' not in st.session_state:
            st.session_state.data_sets = {}
        if 'regions' not in st.session_state:
            st.session_state.regions = {}
        if 'region_counter' not in st.session_state:
            st.session_state.region_counter = 0
    
    def _define_custom_units(self) -> None:
        """Define custom units for biochemical applications."""
        try:
            self.ureg.define('enzyme_unit = 1 * umol / minute = EU')
            self.ureg.define('international_unit = 1 * umol / minute = IU')
        except Exception:
            pass
    
    def _load_data_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file."""
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.txt':
                return self._load_txt_file(uploaded_file)
            elif file_extension == '.csv':
                return self._load_csv_file(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                return self._load_excel_file(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
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
    
    def _create_interactive_plot(self, dataset_name: str = None, show_regions: bool = True) -> go.Figure:
        """Create interactive Plotly figure with modern scientific styling and overlay support."""
        fig = go.Figure()
        
        # Modern scientific color palette using viridis-like colors
        viridis_colors = [
            '#440154', '#482777', '#3f4a8a', '#31678e', '#26838f',
            '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825', '#f0f921'
        ]
        
        # Additional colors for overlay mode
        overlay_colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#1b9e77'
        ]
        
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
        if dataset_name and dataset_name in st.session_state.data_sets:
            # Single dataset mode
            datasets_to_plot = [dataset_name]
        elif dataset_name is None:
            # Overlay mode - plot all datasets
            datasets_to_plot = list(st.session_state.data_sets.keys())
        
        # Choose color palette based on number of datasets
        if len(datasets_to_plot) > 1:
            colors = overlay_colors
        else:
            colors = viridis_colors
        
        # Add data traces
        for i, ds_name in enumerate(datasets_to_plot):
            data = st.session_state.data_sets[ds_name]
            color = colors[i % len(colors)]
            
            # Different line styles for overlay mode
            if len(datasets_to_plot) > 1:
                line_style = dict(width=2, color=color)
                if i >= 5:  # Use dashed lines for datasets 6+
                    line_style['dash'] = 'dash'
            else:
                line_style = dict(width=2, color=color)
            
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=data['product'],
                mode='lines+markers' if len(datasets_to_plot) <= 3 else 'lines',
                name=ds_name,
                line=line_style,
                marker=dict(size=4) if len(datasets_to_plot) <= 3 else None,
                hovertemplate=f'{ds_name}: %{{x:.1f}}s, %{{y:.3f}}UA<extra></extra>',
                showlegend=True
            ))
        
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
                                    fig.add_trace(go.Scatter(
                                        x=region_subset['time'],
                                        y=fitted_y,
                                        mode='lines',
                                        name=f'{region_name} fit',
                                        line=dict(width=3, color='red', dash='dash'),
                                        hovertemplate='Fit: %{y:.3f}<extra></extra>',
                                        showlegend=False
                                    ))
            else:
                # Overlay mode - show regions for all datasets with different colors
                region_colors = ['rgba(255, 0, 0, 0.1)', 'rgba(0, 255, 0, 0.1)', 'rgba(0, 0, 255, 0.1)', 
                               'rgba(255, 255, 0, 0.1)', 'rgba(255, 0, 255, 0.1)', 'rgba(0, 255, 255, 0.1)']
                color_idx = 0
                
                for ds_name in datasets_to_plot:
                    if ds_name in st.session_state.regions:
                        for region_name, region_data in st.session_state.regions[ds_name].items():
                            if 'start' in region_data and 'end' in region_data:
                                # Add shaded region background with dataset-specific color
                                fig.add_shape(
                                    type="rect",
                                    x0=region_data['start'],
                                    y0=global_min * 0.98,
                                    x1=region_data['end'],
                                    y1=global_max * 1.02,
                                    fillcolor=region_colors[color_idx % len(region_colors)],
                                    line_width=0,
                                )
                                
                                # Add fitted line if slope exists
                                if 'slope' in region_data and 'intercept' in region_data:
                                    data = st.session_state.data_sets[ds_name]
                                    region_subset = data[
                                        (data['time'] >= region_data['start']) & 
                                        (data['time'] <= region_data['end'])
                                    ]
                                    if not region_subset.empty:
                                        fitted_y = region_data['slope'] * region_subset['time'] + region_data['intercept']
                                        fig.add_trace(go.Scatter(
                                            x=region_subset['time'],
                                            y=fitted_y,
                                            mode='lines',
                                            name=f'{ds_name} - {region_name} fit',
                                            line=dict(width=2, color=colors[color_idx % len(colors)], dash='dot'),
                                            hovertemplate=f'{ds_name} fit: %{{y:.3f}}<extra></extra>',
                                            showlegend=True
                                        ))
                                color_idx += 1
        
        # Modern scientific layout
        fig.update_layout(
            xaxis=dict(
                title="Time (s)",
                showgrid=True,
                gridcolor=grid_color,
                gridwidth=0.5,
                showline=True,
                linecolor=line_color,
                linewidth=1,
                ticks='outside',
                tickcolor=line_color,
                tickfont=dict(size=11, color=text_color),
                title_font=dict(size=12, color=text_color)
            ),
            yaxis=dict(
                title="Concentration (UA)",
                showgrid=True,
                gridcolor=grid_color,
                gridwidth=0.5,
                showline=True,
                linecolor=line_color,
                linewidth=1,
                ticks='outside',
                tickcolor=line_color,
                tickfont=dict(size=11, color=text_color),
                title_font=dict(size=12, color=text_color)
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
            height=500,
            margin=dict(l=60, r=100, t=40, b=60),
            font=dict(family="Arial", color=text_color),
            hovermode='closest'
        )
        
        return fig
    
    def _calculate_slope(self, dataset_name: str, start_time: float, end_time: float, region_name: str) -> Dict:
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
        
        return {
            'region_name': region_name,
            'dataset': dataset_name,
            'start': start_time,
            'end': end_time,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'std_error': std_err,
            'n_points': n,
            'rmse': rmse
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
    
    def run(self):
        """Main application interface."""
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
        </style>
        """, unsafe_allow_html=True)
        
        # Professional header
        st.markdown("""
        <div class="main-header">
            <h1>Enzyme Kinetics Analyzer</h1>
            <p>Professional web-based tool for interactive enzyme kinetics data analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for configuration and file upload
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
            
            # Unit configuration
            col1, col2 = st.columns(2)
            with col1:
                time_unit = st.selectbox(
                    "Time Unit",
                    ["s", "min", "h"],
                    index=0
                )
            with col2:
                conc_unit = st.selectbox(
                    "Concentration",
                    ["UA", "uM", "mM", "M"],
                    index=0
                )
            
            st.divider()
            
            # File upload section
            st.header("Data Upload")
            
            uploaded_files = st.file_uploader(
                "Select data files",
                type=['txt', 'csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Supported formats: TXT, CSV, Excel (.xlsx, .xls)"
            )
            
            # Process uploaded files
            if uploaded_files:
                st.subheader("Loaded Files")
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.data_sets:
                        with st.spinner(f"Loading {uploaded_file.name}..."):
                            data = self._load_data_file(uploaded_file)
                            if data is not None:
                                st.session_state.data_sets[uploaded_file.name] = data
                                st.success(f"Loaded: {uploaded_file.name}")
                            else:
                                st.error(f"Failed to load: {uploaded_file.name}")
                    else:
                        st.info(f"Already loaded: {uploaded_file.name}")
            
            st.divider()
            
            # Statistics section
            if st.session_state.data_sets:
                st.header("Dataset Statistics")
                st.metric("Datasets loaded", len(st.session_state.data_sets))
                total_regions = sum(len(regions) for regions in st.session_state.regions.values())
                if total_regions > 0:
                    st.metric("Regions analyzed", total_regions)
            
            # Clear data button
            if st.button("Clear All Data", type="secondary", use_container_width=True):
                st.session_state.data_sets = {}
                st.session_state.regions = {}
                st.session_state.region_counter = 0
                st.rerun()
        
        # Main content area
        if not st.session_state.data_sets:
            st.info("Please upload data files using the sidebar to begin analysis.")
            return
        
        # Dataset selection and overlay options
        st.subheader("Dataset Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_dataset = st.selectbox(
                "Select Dataset for Analysis",
                list(st.session_state.data_sets.keys()),
                help="Choose a dataset to analyze and visualize"
            )
        
        with col2:
            show_overlay = st.checkbox(
                "Overlay All Datasets",
                value=False,
                help="Show all datasets on the same plot for comparison"
            )
        
        if selected_dataset:
            # Display plot
            st.subheader("Interactive Data Visualization")
            if show_overlay:
                fig = self._create_interactive_plot(show_regions=True)
            else:
                fig = self._create_interactive_plot(selected_dataset, show_regions=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Region management
            st.subheader("Region Analysis")
            
            # Initialize regions for this dataset if not exists
            if selected_dataset not in st.session_state.regions:
                st.session_state.regions[selected_dataset] = {}
            
            # Add new region section
            with st.expander("Add New Region", expanded=True):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    new_start = st.number_input(
                        "Start (s)",
                        min_value=float(st.session_state.data_sets[selected_dataset]['time'].min()),
                        max_value=float(st.session_state.data_sets[selected_dataset]['time'].max()),
                        value=float(st.session_state.data_sets[selected_dataset]['time'].min()),
                        step=1.0,
                        format="%.2f",
                        key="new_start"
                    )
                
                with col2:
                    new_end = st.number_input(
                        "End (s)",
                        min_value=new_start,
                        max_value=float(st.session_state.data_sets[selected_dataset]['time'].max()),
                        value=float(st.session_state.data_sets[selected_dataset]['time'].max()),
                        step=1.0,
                        format="%.2f",
                        key="new_end"
                    )
                
                with col3:
                    region_name = st.text_input(
                        "Region Name", 
                        value=f"Region_{st.session_state.region_counter + 1}",
                        key="new_region_name"
                    )
                
                with col4:
                    st.write("")  # Spacer
                    if st.button("Add Region", type="primary"):
                        if region_name and region_name not in st.session_state.regions[selected_dataset]:
                            result = self._calculate_slope(selected_dataset, new_start, new_end, region_name)
                            if result:
                                st.session_state.regions[selected_dataset][region_name] = result
                                st.session_state.region_counter += 1
                                st.success(f"Added region: {region_name}")
                                st.rerun()
                        else:
                            st.error("Region name already exists or is empty")
            
            # Show existing regions
            if st.session_state.regions[selected_dataset]:
                st.subheader("Analysis Results")
                
                # Create editable table for regions
                regions_data = []
                for region_name, region_data in st.session_state.regions[selected_dataset].items():
                    regions_data.append({
                        'Region': region_name,
                        'Start': f"{region_data['start']:.2f}",
                        'End': f"{region_data['end']:.2f}",
                        'Slope': f"{region_data['slope']:.6f}",
                        'R²': f"{region_data['r_squared']:.4f}",
                        'RMSE': f"{region_data['rmse']:.4f}",
                        'Points': region_data['n_points']
                    })
                
                if regions_data:
                    regions_df = pd.DataFrame(regions_data)
                    
                    # Display table with option to delete
                    st.dataframe(regions_df, use_container_width=True)
                    
                    # Delete region section
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        region_to_delete = st.selectbox(
                            "Select region to delete",
                            options=[""] + list(st.session_state.regions[selected_dataset].keys())
                        )
                    with col2:
                        st.write("")  # Spacer
                        if st.button("Delete Region", type="secondary") and region_to_delete:
                            del st.session_state.regions[selected_dataset][region_to_delete]
                            st.success(f"Deleted region: {region_to_delete}")
                            st.rerun()
        
        # Global results summary
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
                results_data.append({
                    'Dataset': region['dataset'],
                    'Region': region['region_name'],
                    'Start_Time': f"{region['start']:.2f}",
                    'End_Time': f"{region['end']:.2f}",
                    'Slope': f"{region['slope']:.6f}",
                    'R_Squared': f"{region['r_squared']:.4f}",
                    'Std_Error': f"{region['std_error']:.6f}",
                    'N_Points': region['n_points'],
                    'RMSE': f"{region['rmse']:.4f}"
                })
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Unit conversion options for export
                with st.expander("Export Options & Unit Conversion", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        export_time_unit = st.selectbox(
                            "Export Time Unit",
                            ["s", "min", "h"],
                            index=0,
                            key="export_time_unit"
                        )
                    
                    with col2:
                        export_conc_unit = st.selectbox(
                            "Export Concentration Unit", 
                            ["UA", "uM", "mM", "M"],
                            index=0,
                            key="export_conc_unit"
                        )
                    
                    with col3:
                        current_time_unit = st.selectbox(
                            "Current Time Unit",
                            ["s", "min", "h"],
                            index=0,
                            help="Current time unit in your data"
                        )
                    
                    with col4:
                        current_conc_unit = st.selectbox(
                            "Current Concentration Unit",
                            ["UA", "uM", "mM", "M"], 
                            index=0,
                            help="Current concentration unit in your data"
                        )
                    
                    # Preview converted units
                    if current_time_unit != export_time_unit or current_conc_unit != export_conc_unit:
                        st.info(f"Units will be converted from {current_conc_unit}/{current_time_unit} to {export_conc_unit}/{export_time_unit}")
                
                # Download link with unit conversion
                st.markdown(
                    self._create_download_link(
                        results_df, 
                        "kinetics_analysis_results.xlsx",
                        current_time_unit if 'current_time_unit' in locals() else 's',
                        export_time_unit if 'export_time_unit' in locals() else 's',
                        current_conc_unit if 'current_conc_unit' in locals() else 'UA',
                        export_conc_unit if 'export_conc_unit' in locals() else 'UA'
                    ),
                    unsafe_allow_html=True
                )

# Run the application
if __name__ == "__main__":
    app = EnzymeKineticsStreamlit()
    app.run()