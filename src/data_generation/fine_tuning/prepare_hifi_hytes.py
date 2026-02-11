import os
import pandas as pd
import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.constants import h, c, k
import sys

# Define Project Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(PROJECT_ROOT)

from src.config import SENSOR_CONFIG

# --- Constants ---
LIBRARY_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'spectral_libraries')
LOG_PATH = os.path.join(LIBRARY_DIR, 'summary_log_cleaned.csv')
# Output to local dir (fine_tuning data folder)
OUTPUT_DIR = SCRIPT_DIR

from typing import Optional, Tuple, Dict
import numpy as np

# --- Helper Functions (Copied for Isolation) ---

def planck_law(wavelength_um: np.ndarray, temp_k: float) -> np.ndarray:
    """Calculates Blackbody Radiance (W/m^2/sr/um)"""
    w_m = wavelength_um * 1e-6
    c1 = 2 * h * c**2
    c2 = h * c / k
    exponent = c2 / (w_m * temp_k)
    val = (c1 / (w_m**5)) / (np.exp(exponent) - 1)
    return val * 1e-6

def load_ecostress_spectrum(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        df = pd.read_csv(filepath, sep=r'\s+', comment='#')
        if 'Wavelength' in df.columns and 'Emissivity' in df.columns:
             wav = df['Wavelength'].values
             emis = df['Emissivity'].values
        elif df.shape[1] >= 2:
             df = df.apply(pd.to_numeric, errors='coerce').dropna()
             wav = df.iloc[:, 0].values
             emis = df.iloc[:, 1].values
        else:
             return None, None
        
        wav = pd.to_numeric(wav, errors='coerce')
        emis = pd.to_numeric(emis, errors='coerce')
        mask = ~np.isnan(wav) & ~np.isnan(emis)
        return wav[mask], emis[mask]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def augment_spectrum_micro(spectrum: np.ndarray) -> np.ndarray:
    """
    Micro-Augmentation for High Fidelity.
    10x smaller noise than training:
    - Amplitude: +/- 0.1% (vs 1%)
    - Additive: +/- 0.0005 (vs 0.005)
    
    Args:
        spectrum (np.ndarray): Input spectrum.
        
    Returns:
        np.ndarray: Augmented spectrum.
    """
    scale = np.random.uniform(0.999, 1.001)
    noise = np.random.normal(0.0, 0.0005, size=spectrum.shape)
    new_spec = spectrum * scale + noise
    return np.clip(new_spec, 0.0, 1.0)

# --- SRF Logic (Simplified for HyTES) ---
def load_real_hytes_parameters(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real HyTES parameters file not found at {csv_path}")
    df = pd.read_csv(csv_path).sort_values('band_idx')
    return {'centers': df['center_wavelength'].values, 'fwhms': df['fwhm'].values}

def generate_gaussian_srf(centers, fwhms):
    min_w, max_w = np.min(centers) - 1.0, np.max(centers) + 1.0
    wav_grid = np.arange(min_w, max_w, 0.001)
    responses = []
    for mu, fwhm in zip(centers, fwhms):
        sigma = fwhm / 2.35482
        resp = np.exp(-0.5 * ((wav_grid - mu) / sigma)**2)
        if np.sum(resp) > 0: resp /= np.sum(resp)
        responses.append(resp)
    return {'centers': centers, 'wavelengths': wav_grid, 'responses': np.array(responses)}

def get_hytes_srf():
    cfg = SENSOR_CONFIG['hytes']
    try: 
        p = load_real_hytes_parameters(cfg['processed_params_file'])
        return generate_gaussian_srf(p['centers'], p['fwhms'])
    except:
        raise ValueError("HyTES parameters not found.")

def create_hifi_dataset():
    print("--- Generating High-Fidelity Dataset for Fine-Tuning (HyTES) ---")
    
    # 1. Load Summary Log
    if not os.path.exists(LOG_PATH):
        print(f"Log not found at {LOG_PATH}")
        return
    
    df = pd.read_csv(LOG_PATH)
    
    # 2. Get HyTES SRF (Hardcoded for this task)
    srf_data = get_hytes_srf()
    sensor_centers = srf_data['centers']
    srf_wavs = srf_data['wavelengths']
    srf_resps = srf_data['responses']
    
    print(f"Sensor Centers: {len(sensor_centers)} bands.")
    
    final_data = [] 
    
    # 3. Filter Training Data
    df_split = df[df['situacao'] == 'treino']
    if df_split.empty:
        print("No training data found!")
        return
        
    # 4. Smart Oversampling Logic
    counts = df_split['Material'].value_counts()
    max_count = counts.max()
    print(f"  Oversampling: Merging to {max_count} samples per class.")
    
    for mat, group in df_split.groupby('Material'):
        current_count = len(group)
        needed = max_count - current_count
        
        # Add Original
        for _, row in group.iterrows():
            final_data.append({'row': row, 'augment': False})
            
        # Add Clones with Micro-Augmentation
        if needed > 0:
            samples_to_aug = group.sample(needed, replace=True)
            for _, row in samples_to_aug.iterrows():
                final_data.append({'row': row, 'augment': True})
                
    # 5. Generate Tensors
    dataset_tensors = []
    np.random.shuffle(final_data)
    
    for item in tqdm(final_data, desc="Gen HiFi Data"):
        row = item['row']
        augment = item['augment']
        filename = row['New']
        
        try:
            fpath = os.path.join(LIBRARY_DIR, filename)
            wav_lib, emis_lib = load_ecostress_spectrum(fpath)
            
            if wav_lib is None: continue
            
            # Interpolate
            f_emis = interp1d(wav_lib, emis_lib, bounds_error=False, fill_value=(emis_lib[0], emis_lib[-1]))
            emis_high_res = f_emis(srf_wavs)
            
            # Apply Micro-Augmentation
            if augment:
                emis_high_res = augment_spectrum_micro(emis_high_res)
            
            # Random Temp (Same range as training)
            temp_k = np.random.uniform(280.0, 330.0)
            
            # Radiance
            rad_high_res = planck_law(srf_wavs, temp_k) * emis_high_res
            
            # Convolve
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
                    val_emis = 0.0
                
                l_sensor.append(val_rad)
                e_sensor.append(val_emis)
            
            # Store
            tensor_item = {
                'input_radiance': torch.tensor(l_sensor, dtype=torch.float32),
                'target_temp': torch.tensor([temp_k], dtype=torch.float32),
                'target_emissivity': torch.tensor(e_sensor, dtype=torch.float32),
                'material': row['Material'],
                'filename': filename
            }
            dataset_tensors.append(tensor_item)
            
        except Exception as e:
            continue
            
    # Save Local
    out_path = os.path.join(OUTPUT_DIR, 'hifi_train_dataset.pt')
    torch.save(dataset_tensors, out_path)
    print(f"Saved HiFi Dataset to {out_path} with {len(dataset_tensors)} samples.")

if __name__ == "__main__":
    create_hifi_dataset()
