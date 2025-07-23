#!/usr/bin/env python3
"""
AvaSpec Excel to Text Converter
===============================

This script processes Excel files (.xlsx, .xls) from AvaSpec equipment,
converting them to tab-separated text format for kinetics analysis.

Operations performed:
- Column 1 (Time): Read from Excel Column B (milliseconds), divided by 1000
- Column 2 (Absorbance): Read from Excel Column C, comma decimal converted to dot decimal
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import re
import random

# Configuration
START_ROW = 4  # Row number where data begins (1-indexed)

# Inspirational quotes (matching kinetics_analyzer.py style)
QUOTES = [
    "The only way to make sense out of change is to plunge into it, move with it, and join the dance - Watts",
    "What we observe is not nature itself, but nature exposed to our method of questioning - Heisenberg",
    "Science is not only compatible with spirituality; it is a profound source of spirituality - Sagan",
    "Kinetics tells us how fast, thermodynamics tells us how far",
    "Every reaction has its optimal path - trust the data to show you",
    "R² > 0.99: Excellent fit. R² < 0.98: Time for coffee and reconsideration",
    "The enzyme knows chemistry better than the chemist",
    "Linear kinetics: when enzymes behave like good students",
    "Why don't enzymes ever get tired? They have infinite turnover rates in their dreams",
    "Enzymes are like good friends: they lower your activation energy for getting things done",
    "What did the substrate say to the enzyme? You complete me... by breaking me apart",
    "Allosteric enzymes: proof that proteins have mood swings",
    "Enzymes: billions of years of R&D, no patents required",
    "DNA: the world's most successful backup system (with occasional corrupted files)",
    "RNA: Jack of all trades, master of everything - sorry proteins!",
    "Ribozymes proved that RNA doesn't need protein chaperones to get things done",
    "DNA polymerase has 3' to 5' exonuclease activity because even evolution believes in proofreading",
    "Why did the ribosome break up with the polysome? Too much translation drama",
    "Fact: If you stretched out all the DNA in your body, it would reach the sun and back 300 times",
    "RNA splicing: where introns go to disappear and exons get their moment to shine",
    "Telomerase: the enzyme trying to make you immortal (aging has entered the chat)",
    "DNA repair mechanisms: your genome's personal IT department working 24/7",
    "Why don't DNA strands ever get lost? They always know their 5' from their 3' end",
    "LUCA (Last Universal Common Ancestor): the ultimate single parent raising 4 billion years of offspring",
    "RNA World hypothesis: back when RNA was CEO, CFO, and the entire workforce",
    "Primordial soup: Earth's first attempt at molecular cuisine (no recipe book included)",
    "Iron-sulfur clusters: the original catalysts, still going strong after 3.8 billion years",
    "Miller-Urey experiment: proof that you can make amino acids from lightning and patience",
    "Fact: Cyanobacteria invented photosynthesis and accidentally created the oxygen apocalypse",
    "Self-replication: the ultimate mystery - how did molecules learn to copy themselves?",
    "Fact: All life shares the same genetic code because we all descended from one successful experiment",
    "From chemistry to biology: the greatest startup story never fully documented",
    "Autocatalytic networks: when molecules formed the first self-sustaining businesses",
    "Evolution: 4 billion years of A/B testing with no rollback option",
    "In God we trust, all others must bring data - Deming",
    "The best thing about being a statistician is that you get to play in everyone's backyard - Tukey",
    "Without data, you're just another person with an opinion - Edwards",
    "Correlation does not imply causation, but it does waggle its eyebrows suggestively",
    "Caffeine: the molecule that powers science",
    "Good coffee, good data, good science",
    "Lab rule #1: Never trust data collected before the first cup of coffee",
    "Science runs on coffee and curiosity",
    "The universal solvent for scientific problems: coffee",
    "Fact: The Michaelis-Menten equation was derived in 1913 - still going strong!",
    "Fact: Your morning coffee contains over 1000 different chemical compounds",
    "Fact: Data conversion is the bridge between raw measurements and scientific insights",
    "Fact: Ribozymes can cut and paste RNA like molecular scissors and glue",
    "Fact: The ribosome is a ribozyme - your protein factory is made of RNA",
    "Fact: Some RNA molecules can evolve in test tubes in just hours",
    "Fact: DNA repair enzymes fix about 1000 DNA damages per cell per day - busy little workers"
]

def process_excel_file(file_path):
    """
    Process a single Excel file and convert it to tab-separated text format.
    
    Args:
        file_path (Path): Path to the Excel file to process
    """
    try:
        print(f"-> Processing file: {file_path.name}")
        
        # Read Excel file - start at row 4
        # Use first sheet (index 0), read all columns
        df = pd.read_excel(file_path, sheet_name=0, skiprows=3, header=None)
        
        # Assign column names
        # Column A (index 0): Experiment timestamp
        # Column B (index 1): Time in milliseconds  
        # Column C (index 2): Absorbance data
        
        if df.empty or len(df.columns) < 3:
            print(f"   ERROR: File {file_path.name} doesn't have enough columns or is empty. Skipping.")
            return
            
        # Extract experiment date from first row, column A (was cell A4 in Excel)
        experiment_date = df.iloc[0, 0] if not df.empty else "Unknown"
        
        # Create output file path
        output_path = file_path.with_suffix('.txt')
        
        # Process data
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
                    # Skip rows with invalid data
                    continue
        
        if not valid_rows:
            print(f"   WARNING: No valid data found in {file_path.name}")
            return
            
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header (matching VBS format exactly)
            f.write("##################################################\n")
            f.write("# DATA EXPORT AND TRANSFORMATION REPORT\n")
            f.write("#\n")
            f.write(f"# Original File: {file_path.name}\n")
            f.write(f"# Experiment Start Date/Time: {experiment_date}\n")
            f.write(f"# Processing Date: {datetime.now()}\n")
            f.write("#\n")
            f.write("# Operations Performed:\n")
            f.write("# - Column 1 (Time): Read from Excel Column B (milliseconds), divided by 1000.\n")
            f.write("# - Column 2 (Absorbance): Read from Excel Column C, comma decimal converted to dot decimal.\n")
            f.write("##################################################\n")
            f.write("\n")
            
            # Write column headers
            f.write("Time (s)\tUA\n")
            
            # Write data rows
            for time_sec, absorbance in valid_rows:
                f.write(f"{time_sec}\t{absorbance}\n")
                
        print(f"   Created: {output_path.name} ({len(valid_rows)} data points)")
        
    except Exception as e:
        print(f"   ERROR processing {file_path.name}: {str(e)}")

def show_menu_and_get_choice(excel_files):
    """
    Display menu options and get user choice.
    
    Args:
        excel_files (list): List of Excel files found
        
    Returns:
        str: User choice ('all', file number, or 'quit')
    """
    print("\nAvailable Excel files:")
    for i, file in enumerate(excel_files, 1):
        print(f"  {i}. {file.name}")
    
    print("\nOptions:")
    print("  - Enter 'all' to process all Excel files")
    print(f"  - Enter a number (1-{len(excel_files)}) to process a specific file")
    print("  - Enter 'quit' to exit")
    
    while True:
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'quit':
            return 'quit'
        elif choice == 'all':
            return 'all'
        elif choice.isdigit():
            file_num = int(choice)
            if 1 <= file_num <= len(excel_files):
                return str(file_num)
            else:
                print(f"Please enter a number between 1 and {len(excel_files)}, 'all', or 'quit'.")
        else:
            print("Please enter a valid option: number, 'all', or 'quit'.")

def main():
    """
    Main function - interactive Excel file processor.
    """
    print("================================================")
    print("     AvaSpec Excel to Text Converter")
    print("================================================")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    print(f"Scanning for Excel files in: {script_dir}")
    
    # Find all Excel files in the directory
    excel_files = []
    for pattern in ['*.xlsx', '*.xls']:
        excel_files.extend(script_dir.glob(pattern))
    
    if not excel_files:
        print("\nNo Excel files found in the directory.")
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass
        return
    
    # Show menu and get user choice
    choice = show_menu_and_get_choice(excel_files)
    
    if choice == 'quit':
        print("\nExiting...")
        return
    
    print("\n" + "="*50)
    print("Starting processing...")
    print("="*50)
    
    # Process files based on user choice
    if choice == 'all':
        for excel_file in excel_files:
            process_excel_file(excel_file)
    else:
        # Process single file
        file_index = int(choice) - 1
        process_excel_file(excel_files[file_index])
    
    print("\n" + "-"*50)
    print("Processing complete!")
    print("Text files have been created in the same folder.")
    
    # Add a random inspirational quote
    selected_quote = random.choice(QUOTES)
    print(f"\n💭 \"{selected_quote}\"")
    
    # Only ask for input if running interactively (not in a pipe/redirect)
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        # Handle case when script is run non-interactively
        pass

if __name__ == "__main__":
    main()
