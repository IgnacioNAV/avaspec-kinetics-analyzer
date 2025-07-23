# Kinetics Data Analysis - AvaSpec-NEXOS Spectrometer
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  [![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-orange)](https://streamlit.io/)  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A user-friendly web application for analyzing enzyme kinetics progress curves. It enables interactive visualization, region selection, and statistical analysis of time-course absorbance data. The tool specializes in linear regression of selected regions and supports unit conversions.

## Features

### **Data Analysis**
- **Interactive Region Selection**: Direct box selection from plots using selection tools
- **Extinction Coefficient Support**: Convert absorbance slopes to concentration units
- **Statistical Validation**: R², RMSE, and standard error calculations
- **Multiple Dataset Overlay**: Compare multiple experiments simultaneously
- **Enzyme Activity Calculations**: Convert rates to U/mL or U/mg with volume/mass parameters

### **Visualization**
- **Interactive Tools**: Zoom, pan, hover, and direct region selection
- **Customizable Plot Appearance**: Adjustable point size, opacity, and line thickness
- **Multi-dataset Comparison**: Dynamic color coding for overlay mode

### **Robust Unit Management**
- **Extinction Coefficient Integration**: Automatic absorbance-to-concentration conversion
- **Enzyme Units**: Calculate specific activity (U/mL, U/mg)
- **Transformed Standard Errors**: Statistical accuracy maintained through unit conversions
- **Dual Export Format**: Both original absorbance and converted slopes in results

### **Data Import & Export**
- **Multiple Formats**: TXT, CSV, Excel (.xlsx, .xls) support
- **Standard Input Units**: Time in seconds, absorbance in UA (absorbance units)
- **Smart Column Detection**: Automatic identification of time and absorbance columns
- **Customizable Export**: Excel files with selectable columns and professional formatting
- **AvaSpec Integration**: Direct conversion from spectrometer Excel output to analysis-ready format

## Installation

### Method 1: Using Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv kinetics_env

# Activate the virtual environment
# On Windows:
kinetics_env\Scripts\activate
# On macOS/Linux:
source kinetics_env/bin/activate

# Install dependencies (choose one):
pip install streamlit plotly pandas numpy scipy pint openpyxl
# OR use requirements file:
pip install -r requirements.txt

# Run the application
streamlit run kinetics_analyzer.py
```

### Method 2: Using Conda

```bash
# Create a new conda environment
conda create -n kinetics python=3.10

# Activate the environment
conda activate kinetics

# Install dependencies (choose one):
pip install streamlit plotly pandas numpy scipy pint openpyxl
# OR use requirements file:
pip install -r requirements.txt

# Run the application
streamlit run kinetics_analyzer.py
```

### Method 3: Direct Installation (Global Environment)

```bash
# Install dependencies (choose one):
pip install streamlit plotly pandas numpy scipy pint openpyxl
# OR use requirements file:
pip install -r requirements.txt

# Run the web application
streamlit run kinetics_analyzer.py
```

The application will automatically open in your web browser at `http://localhost:8501`

### Note for Windows Users

If you're using **Windows** and have Python installed via the **Microsoft Store**, it's recommended to **uninstall that version** before proceeding, as it may cause issues with PATH configuration and package execution.

#### Recommended Installation on Windows

1. Download and install Python from the official website: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. During installation, make sure to **check the box** that says **"Add Python to PATH"**


### Troubleshooting Installation

**Virtual environment not activating:**
- Ensure you're in the correct directory
- On Windows, try using `kinetics_env\Scripts\activate.bat` instead
- For PowerShell on Windows, use `kinetics_env\Scripts\Activate.ps1`

**Package installation issues:**
- Update pip: `python -m pip install --upgrade pip`
- If you encounter permission errors, ensure the virtual environment is activated
- For conda users, try: `conda install pandas numpy scipy` before pip installing other packages



## Usage

### 1. **Data Preparation**

#### **For AvaSpec-NEXOS Spectrometer Users**
If you have raw Excel files from the AvaSpec-NEXOS spectrometer, use the provided conversion tool first:

```bash
# Navigate to the data_processing folder
cd data_processing

# Run the conversion script
python avaspec_excel_to_txt.py
```

The `avaspec_excel_to_txt.py` script is specifically designed for AvaSpec-NEXOS spectrometer output

The script will:
- Scan for all Excel files (.xlsx, .xls) in the folder
- Present an menu to process all files or select specific ones
- Convert time from milliseconds to seconds
- Extract absorbance data with proper decimal formatting
- Generate tab-separated text files ready for kinetics analysis

#### **General Data Preparation**
- Prepare files with time (seconds) and absorbance (UA) columns
- Supported formats: TXT, CSV, Excel
- Use descriptive filenames for experimental conditions
- Data input is standardized: time in seconds, absorbance in UA

### 2. **Analysis Workflow**
1. **Upload Data**: Use sidebar file uploader for multiple datasets
2. **Select Dataset**: Choose from dropdown (or enable overlay mode)
3. **Interactive Selection**: 
   - Use box select tool in plot toolbar
   - Drag across desired time range
   - Selected region appears highlighted in green
4. **Configure Analysis**:
   - Set extinction coefficient for concentration conversions
   - Choose target units (mM, μM, per second/minute)
5. **Add Region**: Click "Add Region" to perform analysis
6. **Export Results**: Download comprehensive Excel reports

### 3. **Features**
- **Multiple Regions**: Analyze different phases of the same experiment
- **Unit Conversions**: Transform slopes using extinction coefficients
- **Enzyme Activity**: Calculate specific activity with volume/mass parameters
- **Statistical Validation**: View R², RMSE, and standard errors
- **Multi-dataset Comparison**: Overlay multiple experiments for comparison
- **Plot Customization**: Adjust point size, opacity, and line thickness

## Data Processing Workflow

### **Complete Analysis Pipeline**
1. **Raw Data Collection**: AvaSpec-NEXOS spectrometer generates Excel files
2. **Data Conversion**: Use `avaspec_excel_to_txt.py` to convert to analysis format
3. **Kinetics Analysis**: Load converted files into `kinetics_analyzer.py` web interface
4. **Interactive Analysis**: Select regions, calculate rates, export results


## Example Data Format

Final input data format for kinetics analysis (always in seconds and UA):
```
# Time (s)    Absorbance (UA)
0.0          0.000
1.0          0.015
2.0          0.032
3.0          0.048
...
```


## Authors

- **Ignacio Aravena Valenzuela**
- **Felipe González Ordenes**

*Facultad de Ciencias, Universidad de Chile*

## Acknowledgments

This project was supported by project FONDECYT Postdoctorado N° 3205890 awarded to Felipe González Ordenes.
