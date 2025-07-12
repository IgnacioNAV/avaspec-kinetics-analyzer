# AvaSpec-NEXOS Kinetics Analyzer

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  [![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-orange)](https://streamlit.io/)  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A cross-platform, Streamlit-based Python application for automated processing, fitting, and visualization of enzyme kinetics data recorded with AvaSpec spectrometers.**

---

## Features

- **Automated Data Import**: Read `.txt`, `.csv`, or Excel exports directly from AvaSpec (*.txt/*.csv/*.xlsx/.xls).
- **Region Selection**: Define, highlight, and fit individual reaction phases with slope, intercept, R², RMSE, and point count.
- **Unit Handling**: Conversions between UA (absorbance units), µM, mM, M, seconds, minutes, hours via Pint.
- **Batch Processing**: Upload multiple datasets, overlay for comparison, and generate consolidated results.
- **Zero-Install GUI**: Streamlit interface accessible in any browser on Windows, macOS, or Linux.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/IgnacioNAV/avaspec-kinetics-analyzer.git
cd avaspec-kinetics-analyzer
````

### 2. Install Dependencies

You can either install via the bundled script or manually:

* **Via script**:

  ```bash
  python3 install_streamlit_version.py
  ```
* **Manually**:

  ```bash
  pip install -r requirements.txt
  ```

> The install script installs: `streamlit`, `plotly`, `pandas`, `numpy`, `scipy`, `pint`, `openpyxl`, and `pathlib2`. fileciteturn1file0

### 3. Run the App

```bash
streamlit run enzyme_kinetics_streamlit.py
```

This will launch the web app at `http://localhost:8501/` in your browser.

## Configuration & Usage

1. **Theme & Units**: Select light/dark/auto theme, choose time units (s/min/h) and concentration (UA/µM/mM/M) in the sidebar.
2. **Upload Data**: Drag-and-drop or browse to load `.txt`, `.csv`, `.xlsx`, or `.xls` files.
3. **Visualize**: Use the main panel to view interactive Plotly graphs. Toggle overlay to compare datasets.
4. **Region Analysis**:

   * Expand **Add New Region** to specify start/end times and name.
   * Click **Add Region** to compute slope, intercept, R², RMSE.
   * Review results in the interactive table and remove regions as needed.
5. **Export Results**:

   * Scroll to **Export All Results** for a summary table.
   * Use **Unit Conversion** options and click the download link to get an Excel report.

## Project Structure

```text
avaspec-kinetics-analyzer/
├── install_streamlit_version.py   # Installs required Python packages fileciteturn1file0
├── enzyme_kinetics_streamlit.py  # Streamlit application entrypoint fileciteturn1file1
├── requirements.txt              # Pin dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```
