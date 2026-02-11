import os
import glob
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

# Configuration
SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'raw', 'ecospeclib')
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'raw', 'spectral_libraries')

# Keywords mapping: Keyword -> Material Class
KEYWORDS_MAP = {
    'quartz': 'quartz',
    'kaolinite': 'kaolinite',
    'calcite': 'calcite',
    'feldspar': 'feldspar',
    'albite': 'feldspar',
    'anorthite': 'feldspar',
    'orthoclase': 'feldspar',
    'concrete': 'urban_concrete',
    'asphalt': 'urban_asphalt',
    'leaf': 'vegetation_green_leaf',
    'grass': 'vegetation_green_leaf',
    'tree': 'vegetation_green_leaf',
    'vegetation': 'vegetation_green_leaf'
}

def identify_material(filename: str) -> str | None:
    """
    Identifies the material class based on filename keywords.
    
    Args:
        filename (str): The name of the file.
        
    Returns:
        str | None: The material class if found, else None.
    """
    fname_lower = filename.lower()
    for keyword, material_class in KEYWORDS_MAP.items():
        if keyword in fname_lower:
            return material_class
    return None

def parse_ecostress_file(filepath: str) -> tuple[pd.DataFrame | None, dict | str]:
    """
    Parses ECOSTRESS file handling headers and units.

    Args:
        filepath (str): Path to the source file.

    Returns:
        tuple[pd.DataFrame | None, dict | str]: 
            - DataFrame with 'Wavelength' and 'Emissivity' columns (or None if failed).
            - Metadata dict or Error string.
    """
    try:
        # 1. Read Header Metadata (Manual parsing to be robust)
        meta = {'unit': 'unknown', 'type': 'unknown'}
        
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
            
        header_end_idx = 0
        data_start_found = False
        
        for i, line in enumerate(lines):
            # Try to detect units from standard ECOSTRESS headers
            line_lower = line.lower()
            if 'y units' in line_lower or 'unit' in line_lower:
                if 'percent' in line_lower:
                    meta['unit'] = 'percent'
                elif '0-1' in line_lower or 'unity' in line_lower:
                    meta['unit'] = 'unity'
            
            if 'reflectance' in line_lower:
                meta['type'] = 'reflectance'
            elif 'emissivity' in line_lower:
                meta['type'] = 'emissivity'
                
            # Heuristic for data start: Finding a line with two numbers
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    # If we found numbers, this is likely data start
                    if not data_start_found:
                        header_end_idx = i
                        data_start_found = True
                        break
                except ValueError:
                    pass
        
        # 2. Load Data skipping header
        # Using pandas with python engine for flexible whitespace
        try:
            df = pd.read_csv(filepath, sep='\s+', header=None, skiprows=header_end_idx, on_bad_lines='skip', engine='python')
        except Exception:
             # Fallback: strict tab
             df = pd.read_csv(filepath, sep='\t', header=None, skiprows=header_end_idx, on_bad_lines='skip')
             
        if df.shape[1] < 2:
            return None, "Invalid Columns"
            
        wav = df.iloc[:, 0].values
        val = df.iloc[:, 1].values
        
        # Ensure numeric
        wav = pd.to_numeric(wav, errors='coerce')
        val = pd.to_numeric(val, errors='coerce')
        
        # Remove NaNs
        mask = ~np.isnan(wav) & ~np.isnan(val)
        wav = wav[mask]
        val = val[mask]
        
        # 3. Conversions
        # Default assume Reflectance/Percent if not detected?
        # ASTER/ECOSTRESS typical is Reflectance Percent.
        # Let's use metadata but have sensible defaults or heuristics if unknown
        # Heuristic: if mean value > 1.0, it's likely Percent
        if np.nanmean(val) > 1.05 and meta['unit'] == 'unknown':
            meta['unit'] = 'percent'
        
        # Conversion to 0-1
        if meta['unit'] == 'percent':
            val = val / 100.0
            
        # Conversion R -> E
        if meta['type'] == 'reflectance' or meta['type'] == 'unknown':
            # Kirchhoff's Law: E = 1 - R (assuming opaque)
            # Most libs are Reflectance.
            # If val is generally high (>0.8 for non-metals), it might be Emissivity already?
            # Rocks typically have E ~ 0.8-1.0. R ~ 0.0-0.2.
            # If mean val < 0.5, likely Reflectance.
            if np.nanmean(val) < 0.5:
                val = 1.0 - val
                meta['type'] = 'reflectance_converted'
            else:
                meta['type'] = 'emissivity_assumed'

        # Clip [0, 1]
        val = np.clip(val, 0.0, 1.0)
        
        # 4. Filter Wavelengths
        mask_range = (wav >= 0.4) & (wav <= 15.0)
        wav = wav[mask_range]
        val = val[mask_range]
        
        if len(wav) < 10:
            return None, "Empty after filter"
            
        return pd.DataFrame({'Wavelength': wav, 'Emissivity': val}), meta
        
    except Exception as e:
        return None, str(e)

def process_library():
    print(f"Scanning source: {SOURCE_DIR}")
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        # Create it to guide user
        os.makedirs(SOURCE_DIR)
        print("Created empty source folder. Please place data there.")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Init Summary Log
    log_path = os.path.join(TARGET_DIR, 'summary_log.csv')
    log_data = []
    
    files = glob.glob(os.path.join(SOURCE_DIR, "**", "*.*"), recursive=True)
    # Filter only txt or dat
    files = [f for f in files if f.lower().endswith(('.txt', '.dat', '.asc'))]
    
    print(f"Found {len(files)} total files. Filtering for targets...")
    
    processed_count = 0
    
    for fpath in tqdm(files, desc="Curating Library"):
        filename = os.path.basename(fpath)
        material = identify_material(filename)
        
        if not material:
            continue
            
        # Process Target
        df, info = parse_ecostress_file(fpath)
        
        if df is None:
            log_data.append({'Original': filename, 'New': 'SKIPPED', 'Material': material, 'Status': f"Error: {info}"})
            continue
            
        # Save
        # Unique name: material_originalBase.txt
        safe_name = os.path.splitext(filename)[0].replace(' ', '_').replace('.', '_')
        new_name = f"{material}_{safe_name}.txt"
        out_path = os.path.join(TARGET_DIR, new_name)
        
        # Write clean txt
        np.savetxt(out_path, df.values, fmt='%.6f', header='Wavelength Emissivity', comments='')
        
        log_data.append({
            'Original': filename, 
            'New': new_name, 
            'Material': material, 
            'Status': 'Success',
            'Type': info.get('type'),
            'Unit': info.get('unit')
        })
        processed_count += 1

    # Save Log
    pd.DataFrame(log_data).to_csv(log_path, index=False)
    print(f"Curation complete. Processed {processed_count} files.")
    print(f"Summary saved to {log_path}")

if __name__ == "__main__":
    process_library()
