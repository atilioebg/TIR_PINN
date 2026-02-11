import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse
import sys
from typing import Dict, List, Optional, Any

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.config import SENSOR_CONFIG

# Constants
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'processed')

def load_real_hytes_parameters(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Loads real HyTES center wavelengths and FWHMs from processed CSV.
    
    Args:
        csv_path (str): Path to HyTES parameters CSV.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with 'centers', 'fwhms', 'band_indices'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real HyTES parameters file not found at {csv_path}. Please run src/utils/process_hytes_calibration.py first.")
    
    df = pd.read_csv(csv_path)
    df = df.sort_values('band_idx')
    
    return {
        'centers': df['center_wavelength'].values,
        'fwhms': df['fwhm'].values,
        'band_indices': df['band_idx'].values
    }

def load_aster_srf(srf_dir: str, band_indices: List[int]) -> Optional[Dict[str, np.ndarray]]:
    """
    Loads ASTER SRF from .txt files (Wavenumber vs Response).
    Converts to Wavelength and interpolates to common grid.
    
    Args:
        srf_dir (str): Directory containing SRF files.
        band_indices (List[int]): List of band indices to load.
        
    Returns:
        Optional[Dict[str, np.ndarray]]: Dict with 'centers', 'wavelengths', 'responses' or None if failed.
    """
    responses = []
    centers = []
    
    # Define a high-res grid covering ASTER TIR range (approx 8-12um)
    # We use a slightly wider range to be safe
    wav_grid = np.arange(7.5, 12.5, 0.001)
    
    for idx in band_indices:
        # Pattern: rtcoef_eos_1_aster_srf_ch10.txt
        pattern = os.path.join(srf_dir, f"*_ch{idx:02d}.txt")
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: SRF file for ASTER Band {idx} not found.")
            return None
        
        fpath = files[0]
        
        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()
            
            raw_data = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Try to parse first two columns as floats
                        wn = float(parts[0])
                        resp = float(parts[1])
                        raw_data.append((wn, resp))
                    except ValueError:
                        # Skip header lines or malformed lines
                        continue
                        
            if not raw_data:
                print(f"Warning: No valid data found in {fpath}")
                return None

            raw_data = np.array(raw_data)
            
            # Convert Wavenumber (cm-1) to Wavelength (um)
            # lambda = 10000 / nu
            # Note: Higher wavenumber = Lower wavelength. So order flips.
            wavenumbers = raw_data[:, 0]
            raw_resps = raw_data[:, 1]
            
            wavelengths = 10000.0 / wavenumbers
            
            # Sort by wavelength (ascending)
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            raw_resps = raw_resps[sort_idx]
            
            # Interpolate to common grid
            f = interp1d(wavelengths, raw_resps, bounds_error=False, fill_value=0.0)
            resampled_resp = f(wav_grid)
            
            # Normalize
            total = np.sum(resampled_resp)
            if total > 0:
                resampled_resp /= total
            
            responses.append(resampled_resp)
            
            # Calculate Center Wavelength (Weighted Mean)
            center = np.average(wav_grid, weights=resampled_resp)
            centers.append(center)
            
        except Exception as e:
            print(f"Error processing ASTER file {fpath}: {e}")
            return None

    return {
        'centers': np.array(centers),
        'wavelengths': wav_grid,
        'responses': np.array(responses)
    }

def generate_gaussian_srf_from_params(centers: np.ndarray, fwhms: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generates Gaussian SRFs based on specific Center and FWHM per band.
    
    Args:
        centers (np.ndarray): Center wavelengths.
        fwhms (np.ndarray): Full Width at Half Maximums.
        
    Returns:
        Dict[str, np.ndarray]: Simulated SRF data.
    """
    n_bands = len(centers)
    min_wav = np.min(centers) - 1.0
    max_wav = np.max(centers) + 1.0
    
    # High res grid
    res = 0.001
    wav_grid = np.arange(min_wav, max_wav, res)
    
    responses = []
    
    for mu, fwhm in zip(centers, fwhms):
        sigma = fwhm / 2.35482 
        resp = np.exp(-0.5 * ((wav_grid - mu) / sigma)**2)
        
        total = np.sum(resp)
        if total > 0:
            resp /= total
        
        responses.append(resp)
        
    return {
        'centers': centers,
        'wavelengths': wav_grid,
        'responses': np.array(responses)
    }

def get_sensor_srf(sensor_name: str) -> Dict[str, np.ndarray]:
    """
    Retrieves SRF data for the specified sensor.
    
    Args:
        sensor_name (str): 'hytes' or 'aster'.
        
    Returns:
        Dict[str, np.ndarray]: SRF dictionary.
    """
    cfg = SENSOR_CONFIG[sensor_name]
    
    if sensor_name == 'hytes':
        # Load real params
        params_file = cfg['processed_params_file']
        try:
            params = load_real_hytes_parameters(params_file)
            return generate_gaussian_srf_from_params(params['centers'], params['fwhms'])
        except FileNotFoundError:
            print("HyTES parameters missing. Falling back is not allowed for HyTES anymore.")
            raise

    elif sensor_name == 'aster':
        # Try loading real SRF files
        srf_dir = cfg['srf_dir']
        indices = cfg['band_indices']
        
        srf_data = load_aster_srf(srf_dir, indices)
        
        if srf_data is not None:
            print("Loaded real ASTER SRF files.")
            return srf_data
        else:
            print("Could not load ASTER SRF files. Using Gaussian fallback.")
            fallback = cfg['fallback_params']
            return generate_gaussian_srf_from_params(fallback['centers'], fallback['fwhms'])
            
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}")

def load_modtran_lut(lut_dir: str) -> List[Dict[str, Any]]:
    """
    Loads MODTRAN tables. 
    Falls back to synthetic data if real tables are missing.
    
    Args:
        lut_dir (str): Directory containing LUT files.
        
    Returns:
        List[Dict[str, Any]]: List of atmospheric profiles.
    """
    catalog_path = os.path.join(lut_dir, 'lut_catalog.csv')
    luts = []
    
    # Synthetic Atmosphere Generation fallback
    wav_grid = np.linspace(7.0, 13.0, 2000)
    tau_base = np.exp(-0.1 * 1.5 * (1 + np.abs(wav_grid - 9.6)))
    tau_base = np.clip(tau_base, 0.0, 1.0)
    bb_atm = 8.0 
    
    for i in range(10):
        factor = 1.0 + (i * 0.1)
        tau = np.power(tau_base, factor)
        l_up = (1 - tau) * bb_atm * 0.5 * factor
        l_down = (1 - tau) * bb_atm * 0.3 * factor
        
        luts.append({
            'run_id': f'syn_atm_{i}',
            'wavelengths': wav_grid,
            'transmission': tau,
            'path_radiance_up': l_up,
            'path_radiance_down': l_down
        })
            
    print(f"Generated {len(luts)} synthetic atmospheric profiles for simulation.")
    return luts

def run_simulation(sensor: str, ground_truth_path: str, lut_dir: str, output_dir: str) -> None:
    """
    Main simulation loop.
    
    Args:
        sensor (str): Sensor name.
        ground_truth_path (str): Path to ground truth pickle.
        lut_dir (str): Path to LUT directory.
        output_dir (str): Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Ground Truth...")
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
        
    print(f"Preparing {sensor.upper()} Sensor Model...")
    srf_data = get_sensor_srf(sensor)
    print(f"Sensor ready with {len(srf_data['centers'])} bands.")

    print("Loading/Synthesizing Atmospheric LUTs...")
    luts = load_modtran_lut(lut_dir)
    
    final_dataset = []
    
    print("Starting Radiative Transfer Simulation...")
    
    import random
    
    for gt in tqdm(ground_truth, desc=f"Simulating {sensor.upper()}"):
        # Pick a random atmosphere
        atm = random.choice(luts)
        
        # 1. Prepare High Res Grid (SRF grid)
        common_grid = srf_data['wavelengths']
        
        # Interpolate components to common grid
        f_emis = interp1d(gt['wavelengths'], gt['emissivity'], bounds_error=False, fill_value='extrapolate')
        emis_res = f_emis(common_grid)
        
        f_lg = interp1d(gt['wavelengths'], gt['ground_radiance'], bounds_error=False, fill_value=0.0)
        l_g_res = f_lg(common_grid)
        
        f_tau = interp1d(atm['wavelengths'], atm['transmission'], bounds_error=False, fill_value=1.0)
        tau_res = f_tau(common_grid)
        
        f_up = interp1d(atm['wavelengths'], atm['path_radiance_up'], bounds_error=False, fill_value=0.0)
        l_up_res = f_up(common_grid)
        
        f_down = interp1d(atm['wavelengths'], atm['path_radiance_down'], bounds_error=False, fill_value=0.0)
        l_down_res = f_down(common_grid)
        
        # 2. Apply RTE
        l_toa_high_res = (tau_res * l_g_res) + l_up_res + (tau_res * (1 - emis_res) * l_down_res)
        
        # 3. Apply Sensor Response (Convolve)
        l_toa_sensor = []
        l_emis_sensor = []
        
        # Matrix mult logic could be faster but loop is fine
        for i in range(len(srf_data['centers'])):
            resp = srf_data['responses'][i]
            val_rad = np.sum(l_toa_high_res * resp)
            l_toa_sensor.append(val_rad)
            
            val_emis = np.sum(emis_res * resp)
            l_emis_sensor.append(val_emis)
            
        atm_tau_sensor = []
        atm_up_sensor = []
        atm_down_sensor = []
        
        for i in range(len(srf_data['centers'])):
             resp = srf_data['responses'][i]
             atm_tau_sensor.append(np.sum(tau_res * resp))
             atm_up_sensor.append(np.sum(l_up_res * resp))
             atm_down_sensor.append(np.sum(l_down_res * resp))

        # 4. Store Data
        sample = {
            'input_radiance': torch.tensor(l_toa_sensor, dtype=torch.float32),
            'target_temp': torch.tensor([gt['temperature_k']], dtype=torch.float32),
            'target_emissivity': torch.tensor(l_emis_sensor, dtype=torch.float32),
            'atm_params': {
                'tau': torch.tensor(atm_tau_sensor, dtype=torch.float32),
                'up': torch.tensor(atm_up_sensor, dtype=torch.float32),
                'down': torch.tensor(atm_down_sensor, dtype=torch.float32)
            },
            'mineral_name': gt['mineral_name'],
            'wavelengths': torch.tensor(srf_data['centers'], dtype=torch.float32) # Store centers for viz
        }
        final_dataset.append(sample)
        
    out_file = os.path.join(output_dir, f'synthetic_dataset_{sensor}.pt')
    print(f"Saving {len(final_dataset)} samples to {out_file}")
    torch.save(final_dataset, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sensor Simulation')
    parser.add_argument('--sensor', type=str, default='hytes', choices=['hytes', 'aster'],
                        help='Sensor to simulate')
    args = parser.parse_args()
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    GT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ground_truth', 'ground_truth_dataset.pkl')
    LUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'atmospheric_luts')
    OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'synthetic_dataset')
    
    if not os.path.exists(GT_PATH):
        print(f"Error: Ground Truth file not found at {GT_PATH}")
    else:
        run_simulation(args.sensor, GT_PATH, LUT_DIR, OUT_DIR)
