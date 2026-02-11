import os
import pandas as pd
import numpy as np
import torch
import argparse
import sys
from scipy.interpolate import interp1d
from tqdm import tqdm
import glob
from scipy.constants import h, c, k
from typing import Optional, Dict, Tuple, List, Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.config import SENSOR_CONFIG

# --- Constants & Helpers ---

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LIBRARY_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'spectral_libraries')
LOG_PATH = os.path.join(LIBRARY_DIR, 'summary_log_cleaned.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def planck_law(wavelength_um: np.ndarray, temp_k: float) -> np.ndarray:
    """
    Calculates Blackbody Radiance (W/m^2/sr/um) using Planck's Law.
    
    Args:
        wavelength_um (np.ndarray): Wavelengths in micrometers.
        temp_k (float): Temperature in Kelvin.
        
    Returns:
        np.ndarray: Blackbody radiance spectrum.
    """
    w_m = wavelength_um * 1e-6
    c1 = 2 * h * c**2
    c2 = h * c / k
    exponent = c2 / (w_m * temp_k)
    val = (c1 / (w_m**5)) / (np.exp(exponent) - 1)
    return val * 1e-6

def load_ecostress_spectrum(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads Emissivity and Wavelength from an ECOSTRESS library file.
    
    Args:
        filepath (str): Path to the .txt file.
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (Wavelengths, Emissivity) or (None, None).
    """
    try:
        df = pd.read_csv(filepath, sep=r'\s+', comment='#')
        # Check if columns are numeric (headerless) or named (curated)
        if 'Wavelength' in df.columns and 'Emissivity' in df.columns:
             wav = df['Wavelength'].values
             emis = df['Emissivity'].values
        elif df.shape[1] >= 2:
             # Try converting to numeric, if fail, maybe header was read as data?
             df = df.apply(pd.to_numeric, errors='coerce').dropna()
             wav = df.iloc[:, 0].values
             emis = df.iloc[:, 1].values
        else:
             return None, None
        
        # Ensure numeric
        wav = pd.to_numeric(wav, errors='coerce')
        emis = pd.to_numeric(emis, errors='coerce')
        mask = ~np.isnan(wav) & ~np.isnan(emis)
        return wav[mask], emis[mask]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def augment_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Applies random perturbation to the spectrum.
    Formula: new = original * uniform(0.995, 1.005) + normal(0, 0.002)
    Reduced noise to encourage learning of fine spectral features (kaolinite dublet).
    
    Args:
        spectrum (np.ndarray): Input emissivity spectrum (0-1).
        
    Returns:
        np.ndarray: Augmented spectrum.
    """
    scale = np.random.uniform(0.995, 1.005)
    noise = np.random.normal(0.0, 0.002, size=spectrum.shape)
    new_spec = spectrum * scale + noise
    # Clip to keep physics valid [0, 1]
    return np.clip(new_spec, 0.0, 1.0)

# --- SRF Logic ---

def load_real_hytes_parameters(csv_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real HyTES parameters file not found at {csv_path}")
    df = pd.read_csv(csv_path).sort_values('band_idx')
    return {'centers': df['center_wavelength'].values, 'fwhms': df['fwhm'].values}

def load_aster_srf(srf_dir: str, band_indices: List[int]) -> Optional[Dict[str, np.ndarray]]:
    responses = []
    centers = []
    wav_grid = np.arange(7.5, 14.5, 0.001) # Wide grid for ASTER TIR
    
    for idx in band_indices:
        files = glob.glob(os.path.join(srf_dir, f"*_ch{idx:02d}.txt"))
        if not files: return None
        
        try:
            raw = []
            with open(files[0], 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        try: raw.append((float(parts[0]), float(parts[1])))
                        except: continue
            if not raw: return None
            raw = np.array(raw)
            # cm-1 to um: 10000/nu
            w_um = 10000.0 / raw[:, 0]
            resp = raw[:, 1]
            
            # Sort ascending wavelength
            sidx = np.argsort(w_um)
            w_um = w_um[sidx]
            resp = resp[sidx]
            
            f = interp1d(w_um, resp, bounds_error=False, fill_value=0.0)
            resampled = f(wav_grid)
            if np.sum(resampled) > 0: resampled /= np.sum(resampled)
            
            responses.append(resampled)
            centers.append(np.average(wav_grid, weights=resampled))
        except: return None
            
    return {'centers': np.array(centers), 'wavelengths': wav_grid, 'responses': np.array(responses)}

def generate_gaussian_srf(centers: np.ndarray, fwhms: np.ndarray) -> Dict[str, np.ndarray]:
    min_w, max_w = np.min(centers) - 1.0, np.max(centers) + 1.0
    wav_grid = np.arange(min_w, max_w, 0.001)
    responses = []
    for mu, fwhm in zip(centers, fwhms):
        sigma = fwhm / 2.35482
        resp = np.exp(-0.5 * ((wav_grid - mu) / sigma)**2)
        if np.sum(resp) > 0: resp /= np.sum(resp)
        responses.append(resp)
    return {'centers': centers, 'wavelengths': wav_grid, 'responses': np.array(responses)}

def get_sensor_srf(sensor: str) -> Optional[Dict[str, np.ndarray]]:
    cfg = SENSOR_CONFIG[sensor]
    if sensor == 'hytes':
        try: 
            p = load_real_hytes_parameters(cfg['processed_params_file'])
            return generate_gaussian_srf(p['centers'], p['fwhms'])
        except:
            raise ValueError("HyTES parameters not found.")
            
    elif sensor == 'aster':
        srf = load_aster_srf(cfg['srf_dir'], cfg['band_indices'])
        if srf: return srf
        fb = cfg['fallback_params']
        return generate_gaussian_srf(fb['centers'], fb['fwhms'])
    return None

# --- Main Logic ---

def create_dataset(sensor: str) -> None:
    print(f"--- Generating Smart Augmented Dataset for {sensor.upper()} ---")
    
    # 1. Load Summary Log
    if not os.path.exists(LOG_PATH):
        print(f"Cleaned log not found at {LOG_PATH}. Run clean_and_tag_library.py first.")
        return
    
    df = pd.read_csv(LOG_PATH)
    
    # 2. Get Sensor SRF
    srf_data = get_sensor_srf(sensor)
    sensor_centers = srf_data['centers']
    srf_wavs = srf_data['wavelengths']
    srf_resps = srf_data['responses']
    
    print(f"Sensor Centers: {len(sensor_centers)} bands.")
    
    # 3. Process Splits
    for split_name in ['train', 'val']:
        
        final_data = [] # List to store dataset items
        
        if split_name == 'train':
            df_split = df[df['situacao'] == 'treino']
            if df_split.empty:
                print("No training data found!")
                continue
                
            # Smart Oversampling with Augmentation
            counts = df_split['Material'].value_counts()
            max_count = counts.max()
            print(f"  Training Set: Max count merged to {max_count} per class (Augmentation ON).")
            
            for mat, group in df_split.groupby('Material'):
                current_count = len(group)
                needed = max_count - current_count
                
                # 1. Add Original Samples (No Augmentation)
                for _, row in group.iterrows():
                    final_data.append({'row': row, 'augment': False})
                    
                # 2. Add Augmented Clones
                if needed > 0:
                    # Sample replacements from group to augment
                    samples_to_aug = group.sample(needed, replace=True)
                    for _, row in samples_to_aug.iterrows():
                        final_data.append({'row': row, 'augment': True})
                        
        else: # Validation
            df_split = df[df['situacao'] == 'teste/validacao']
            print(f"  Validation Set: {len(df_split)} samples (Augmentation OFF).")
            for _, row in df_split.iterrows():
                final_data.append({'row': row, 'augment': False})

        # 5. Generate Tensors from List
        dataset_tensors = []
        
        # shuffle training data only
        if split_name == 'train':
            np.random.shuffle(final_data)
            
        desc = f"Gen {split_name}"
        for item in tqdm(final_data, desc=desc):
            row = item['row']
            augment = item['augment']
            filename = row['New']
            
            try:
                fpath = os.path.join(LIBRARY_DIR, filename)
                wav_lib, emis_lib = load_ecostress_spectrum(fpath)
                
                if wav_lib is None: continue
                
                # Interpolate to high-res
                f_emis = interp1d(wav_lib, emis_lib, bounds_error=False, fill_value=(emis_lib[0], emis_lib[-1]))
                emis_high_res = f_emis(srf_wavs)
                
                # Apply Augmentation if needed (BEFORE Radiance calc)
                if augment:
                    emis_high_res = augment_spectrum(emis_high_res)
                
                # Sample Random Temperature
                temp_k = np.random.uniform(280.0, 330.0)
                
                # Calculate Radiance
                rad_high_res = planck_law(srf_wavs, temp_k) * emis_high_res
                
                # Convolve to Sensor Bands
                l_sensor = []
                e_sensor = []
                
                for i in range(len(sensor_centers)):
                    resp = srf_resps[i]
                    norm = np.sum(resp)
                    if norm > 0:
                        val_rad = np.sum(rad_high_res * resp) / norm
                        val_emis = np.sum(emis_high_res * resp) / norm
                    else:
                        val_rad = 0.0
                        val_emis = 0.0 # Should probably be handled better but 0 safe for now
                    
                    l_sensor.append(val_rad)
                    e_sensor.append(val_emis)
                
                # Store
                tensor_item = {
                    'input_radiance': torch.tensor(l_sensor, dtype=torch.float32),
                    'target_temp': torch.tensor([temp_k], dtype=torch.float32),
                    'target_emissivity': torch.tensor(e_sensor, dtype=torch.float32),
                    'material': row['Material'],
                    'filename': filename,
                    'is_augmented': augment
                }
                dataset_tensors.append(tensor_item)
                
            except Exception as e:
                continue
                
        # Save
        out_name = f"{split_name}_dataset_{sensor}.pt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        torch.save(dataset_tensors, out_path)
        print(f"Saved {out_name} with {len(dataset_tensors)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', type=str, required=True, choices=['hytes', 'aster'])
    args = parser.parse_args()
    
    create_dataset(args.sensor)
