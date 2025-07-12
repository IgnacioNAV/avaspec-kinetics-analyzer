#!/usr/bin/env python3
"""
Installation and setup script for the Streamlit version
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install required packages for the Streamlit version"""
    
    print("Installing packages for Enzyme Kinetics Analyzer (Streamlit Version)")
    print("=" * 70)
    
    # Required packages
    packages = [
        "streamlit",
        "plotly", 
        "pandas", 
        "numpy",
        "scipy",
        "pint",
        "openpyxl",
        "pathlib2"  # For Python < 3.4 compatibility
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            failed_packages.append(package)
        print()
    
    print("Installation Summary:")
    print("=" * 30)
    
    if not failed_packages:
        print("✓ All packages installed successfully!")
        print("\nYou can now run the analyzer with:")
        print("streamlit run enzyme_kinetics_streamlit.py")
        print("\nThis will open the web application in your browser.")
    else:
        print(f"✗ Failed to install: {', '.join(failed_packages)}")
        print("\nPlease try installing manually:")
        for package in failed_packages:
            print(f"pip install {package}")
    
    print("\nPackage descriptions:")
    print("- streamlit: Web application framework for interactive apps")
    print("- plotly: Interactive plotting library") 
    print("- pandas: Data manipulation and analysis")
    print("- numpy: Numerical computing")
    print("- scipy: Scientific computing (for linear regression)")
    print("- pint: Physical units handling")
    print("- openpyxl: Excel file support")
    print("- pathlib2: Cross-platform file path operations")
    
    print("\nUsage Instructions:")
    print("1. Open terminal/command prompt")
    print("2. Navigate to the directory containing enzyme_kinetics_streamlit.py")
    print("3. Run: streamlit run enzyme_kinetics_streamlit.py")
    print("4. Your web browser will open with the application")
    print("5. Upload data files and start analyzing!")

if __name__ == "__main__":
    main()