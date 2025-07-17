#!/usr/bin/env python3
"""
Installation and setup script for Enzyme Kinetics Analyzer

This script helps set up the project with proper virtual environment support
and installs all required dependencies.
"""

import subprocess
import sys
import os
import venv
from pathlib import Path

def check_virtual_env():
    """Check if running in a virtual environment"""
    return (hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def create_virtual_env(venv_path="kinetics_env"):
    """Create a virtual environment"""
    try:
        print(f"Creating virtual environment: {venv_path}")
        venv.create(venv_path, with_pip=True)
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def get_venv_activation_command(venv_path="kinetics_env"):
    """Get the appropriate activation command for the current OS"""
    if os.name == 'nt':  # Windows
        return f"{venv_path}\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        return f"source {venv_path}/bin/activate"

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_requirements_file():
    """Check if requirements.txt exists and create it if not"""
    req_file = Path("requirements.txt")
    packages = [
        "streamlit",
        "plotly", 
        "pandas", 
        "numpy",
        "scipy",
        "pint",
        "openpyxl"
    ]
    
    if not req_file.exists():
        print("Creating requirements.txt...")
        with open(req_file, 'w') as f:
            for package in packages:
                f.write(f"{package}\n")
        print("✓ requirements.txt created")
    return packages

def main():
    """Main installation script"""
    
    print("Enzyme Kinetics Analyzer - Setup Script")
    print("=" * 70)
    print("This script will help you set up the Enzyme Kinetics Analyzer")
    print("with proper virtual environment support.\n")
    
    # Check if in virtual environment
    in_venv = check_virtual_env()
    if in_venv:
        print("✓ Running in virtual environment")
    else:
        print("⚠ Not running in a virtual environment")
        
        # Ask user about virtual environment
        response = input("\nWould you like to create a virtual environment? (recommended) [Y/n]: ").strip().lower()
        
        if response in ['', 'y', 'yes']:
            venv_name = input("Enter virtual environment name [kinetics_env]: ").strip()
            if not venv_name:
                venv_name = "kinetics_env"
            
            if create_virtual_env(venv_name):
                print(f"\n✓ Virtual environment '{venv_name}' created successfully!")
                print("\nTo activate the virtual environment, run:")
                print(f"  {get_venv_activation_command(venv_name)}")
                print("\nThen re-run this script to install packages:")
                print("  python dependencies.py")
                return
            else:
                print("Continuing with global installation...")
        else:
            print("Continuing with global installation (not recommended)...")
    
    print("\n" + "=" * 50)
    print("Installing packages...")
    print("=" * 50)
    
    # Check/create requirements.txt
    packages = check_requirements_file()
    
    failed_packages = []
    
    # Update pip first
    print("Updating pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✓ pip updated successfully")
    except subprocess.CalledProcessError:
        print("⚠ Failed to update pip, continuing...")
    print()
    
    # Install packages
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
        print("\n" + "=" * 50)
        print("SETUP COMPLETE!")
        print("=" * 50)
        print("\nYou can now run the analyzer with:")
        print("  streamlit run kinetics_analyzer.py")
        print("\nThis will open the web application at: http://localhost:8501")
        
        if in_venv:
            print(f"\n✓ Running in virtual environment: {sys.prefix}")
        else:
            print("\n⚠ Running in global Python environment")
            
    else:
        print(f"✗ Failed to install: {', '.join(failed_packages)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have an active internet connection")
        print("2. Try updating pip: python -m pip install --upgrade pip")
        print("3. Install manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
    
    print("\n" + "=" * 50)
    print("Package Information:")
    print("=" * 50)
    print("- streamlit: Web application framework for interactive apps")
    print("- plotly: Interactive plotting library") 
    print("- pandas: Data manipulation and analysis")
    print("- numpy: Numerical computing")
    print("- scipy: Scientific computing (for linear regression)")
    print("- pint: Physical units handling")
    print("- openpyxl: Excel file support")
    
    print("\n" + "=" * 50)
    print("Usage Instructions:")
    print("=" * 50)
    print("1. Open terminal/command prompt")
    print("2. Navigate to the directory containing kinetics_analyzer.py")
    if not in_venv and not check_virtual_env():
        print("3. Activate virtual environment (if created):")
        print("   - Windows: kinetics_env\\Scripts\\activate")
        print("   - macOS/Linux: source kinetics_env/bin/activate")
        print("4. Run: streamlit run kinetics_analyzer.py")
    else:
        print("3. Run: streamlit run kinetics_analyzer.py")
    print("5. Your web browser will open with the application")
    print("6. Upload data files and start analyzing!")
    
    print("\n" + "=" * 50)
    print("Virtual Environment Commands:")
    print("=" * 50)
    print("Create: python -m venv kinetics_env")
    print("Activate (Windows): kinetics_env\\Scripts\\activate")
    print("Activate (macOS/Linux): source kinetics_env/bin/activate")
    print("Deactivate: deactivate")
    print("Install from requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
