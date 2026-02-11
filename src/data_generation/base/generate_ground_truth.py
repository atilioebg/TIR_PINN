import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.constants import h, c, k
from tqdm import tqdm
from typing import Optional, Tuple

# Constants for Planck's Law
# h = 6.626e-34 J*s (Planck constant)
# c = 2.998e8 m/s (Speed of light)
# k = 1.380e-23 J/K (Boltzmann constant)

ECOSTRESS_PATH = os.path.join('..', '..', '..', 'data', 'raw', 'spectral_libraries')

def planck_law(wavelength_um: np.ndarray, temp_k: float) -> np.ndarray:
    """
    Calculates Blackbody Radiance (B) using Planck's Law.
    
    Args:
        wavelength_um (np.array): Wavelength in micrometers.
        temp_k (float): Temperature in Kelvin.
        
    Returns:
        np.array: Radiance values (W/m^2/sr/um).
    """
    w_m = wavelength_um * 1e-6  # Convert um to meters
    
    c1 = 2 * h * c**2
    c2 = h * c / k
    
    # Avoid division by zero or overflow
    # B(lambda, T) = (2hc^2 / lambda^5) * (1 / (exp(hc/lambda*k*T) - 1))
    
    # Safe exponent calculation
    exponent = c2 / (w_m * temp_k)
    # If exponent is too large, result approaches 0
    
    val = (c1 / (w_m**5)) / (np.exp(exponent) - 1)
    
    # Convert from W/m^2/sr/m to W/m^2/sr/um for standard remote sensing units
    return val * 1e-6

def load_ecostress_spectrum(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parses an ECOSTRESS library text file.
    Supports both legacy headerless format and new curated format with headers.
    
    Args:
        filepath (str): Path to the .txt file.
        
    Returns:
        tuple: (wavelengths in um, emissivity values)
    """
    try:
        # Try reading with whitespace separator (handling new format with header)
        # The new format has 'Wavelength Emissivity' as header
        try:
            df = pd.read_csv(filepath, sep=r'\s+', comment='#')
            
            # Check if likely headerless (all numeric columns but read as str due to first line?)
            # If the first row was read as header but was actually data, columns will be numbers (as strings or floats)
            # If standard curated format, columns are ['Wavelength', 'Emissivity']
            
            if df.shape[1] < 2:
                 raise ValueError("Not enough columns")

            # Validate numeric data
            # If we read a header incorrectly (e.g. file had no header), the first row is lost.
            # But our curated files HAVE headers.
            
            # If columns are not numeric (i.e., we correctly picked up text header), perfect.
            # If columns ARE numeric, maybe we lost the first row?
            # But we prioritize the curated format.
            
            # Ensure we are working with numeric data
            # Convert to numeric, coercing errors (like if header was read as data in a different mode)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
        except Exception:
            # Fallback for old tab-separated or headerless files
            df = pd.read_csv(filepath, sep='\t', header=None, comment='H')
            if df.shape[1] < 2:
                 df = pd.read_csv(filepath, sep=r'\s+', header=None, comment='H')

        if df.shape[1] < 2:
            return None, None
        
        # dynamic column selection: assume 0 is wav, 1 is emis
        wavelengths = df.iloc[:, 0].values
        emissivity = df.iloc[:, 1].values
        
        # Ensure numeric (again, just in case)
        wavelengths = pd.to_numeric(wavelengths, errors='coerce')
        emissivity = pd.to_numeric(emissivity, errors='coerce')
        
        mask = ~np.isnan(wavelengths) & ~np.isnan(emissivity)
        wavelengths = wavelengths[mask]
        emissivity = emissivity[mask]

        if len(wavelengths) == 0:
            return None, None

        # Normalize emissivity to 0-1 range if it is in percent
        if np.max(emissivity) > 1.05: # Allow small margin for 1.0 noise
            emissivity = emissivity / 100.0
            
        # Clip to ensure valid range [0, 1] physically
        emissivity = np.clip(emissivity, 0.0, 1.0)
            
        return wavelengths, emissivity
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def generate_ground_truth(
    input_dir: str, 
    output_dir: str, 
    temp_min: float = 280, 
    temp_max: float = 330, 
    temp_step: float = 0.1
) -> None:
    """
    Generates ground truth radiance data from emissivity and temperature.
    
    Args:
        input_dir (str): Directory containing emissivity .txt files.
        output_dir (str): Directory to save processed data.
        temp_min (float): Min temperature in Kelvin.
        temp_max (float): Max temperature in Kelvin.
        temp_step (float): Step size for temperature iteration.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mineral_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not mineral_files:
        print(f"No .txt files found in {input_dir}")
        return

    temperatures = np.arange(temp_min, temp_max + temp_step, temp_step)
    
    print(f"Found {len(mineral_files)} mineral files.")
    print(f"Generating data for {len(temperatures)} temperature steps each.")
    
    results = []
    
    for m_file in tqdm(mineral_files, desc="Processing Minerals"):
        mineral_name = os.path.basename(m_file).replace('.txt', '')
        
        wavs, emis = load_ecostress_spectrum(m_file)
        if wavs is None:
            continue
            
        for T in temperatures:
            blackbody_rad = planck_law(wavs, T)
            ground_radiance = emis * blackbody_rad
            
            # Store everything needed for training/validation
            entry = {
                'mineral_name': mineral_name,
                'temperature_k': T,
                'wavelengths': wavs,
                'emissivity': emis,
                'blackbody_radiance': blackbody_rad,
                'ground_radiance': ground_radiance # L_g = epsilon * B(T)
            }
            results.append(entry)
            
    output_path = os.path.join(output_dir, 'ground_truth_dataset.pkl')
    print(f"Saving {len(results)} samples to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print("Ground Truth Generation Complete.")

if __name__ == "__main__":
    # Define absolute paths based on project structure relative to this script
    # Script is in src/data_generation
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    # Resolve ECOSTRESS_PATH relative to the script location
    if not os.path.isabs(ECOSTRESS_PATH):
        RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ECOSTRESS_PATH))
    else:
        RAW_DIR = ECOSTRESS_PATH
        
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ground_truth')
    
    if not os.path.exists(RAW_DIR):
        print(f"Warning: Raw data directory {RAW_DIR} does not exist.")
        os.makedirs(RAW_DIR)
        print("Created it. Please populate with ECOSTRESS .txt files.")
    
    generate_ground_truth(RAW_DIR, PROCESSED_DIR)
