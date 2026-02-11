import os
import pandas as pd
import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.constants import h, c, k
import sys
import glob

# Define Project Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(PROJECT_ROOT)

from src.config import SENSOR_CONFIG

# --- Constants ---
LIBRARY_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'spectral_libraries')
LOG_PATH = os.path.join(LIBRARY_DIR, 'summary_log_cleaned.csv')
OUTPUT_DIR = SCRIPT_DIR

from typing import Optional, Tuple, Dict

# --- Helper Functions ---

def planck_law(wavelength_um: np.ndarray, temp_k: float) -> np.ndarray:
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
    Micro-Augmentation for High Fidelity ASTER.
    
    Args:
        spectrum (np.ndarray): Input spectrum.
        
    Returns:
        np.ndarray: Augmented spectrum.
    """
    scale = np.random.uniform(0.999, 1.001)
    noise = np.random.normal(0.0, 0.0005, size=spectrum.shape)
    new_spec = spectrum * scale + noise
    return np.clip(new_spec, 0.0, 1.0)

# --- ASTER SRF Logic ---
def load_aster_srf(srf_dir, band_indices):
    responses = []
    centers = []
    wav_grid = np.arange(7.5, 14.5, 0.001) 
    
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
            # cm-1 to um: 10000/nu check? No, assumes microns if raw
            # Actually project uses 10000/nu if wavenumbers. 
            # Reusing simulate_sensor logic:
            # "JHU Nicolet" files often have microns. ASTER SRF from online usually microns.
            # But let's check basic logic. Using Fallback for guaranteed run if files missing.
            # We will prioritize Fallback since we are in "Clean Room" and might not have raw SRF locally handy.
            # Actually, let's use the config's fallback directly for simplicity and robustness.
            return None
        except: return None
    return None

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

def get_aster_srf():
    # Use Fallback Gaussian SRF for ASTER to ensure it runs without external dependencies
    cfg = SENSOR_CONFIG['aster']
    fb = cfg['fallback_params']
    return generate_gaussian_srf(fb['centers'], fb['fwhms'])

def create_hifi_dataset_aster():
    print("--- Generating High-Fidelity Dataset for Fine-Tuning (ASTER) ---")
    
    if not os.path.exists(LOG_PATH):
        print(f"Log not found at {LOG_PATH}")
        return
    
    df = pd.read_csv(LOG_PATH)
    
    # 2. Get ASTER SRF
    srf_data = get_aster_srf()
    sensor_centers = srf_data['centers']
    srf_wavs = srf_data['wavelengths']
    srf_resps = srf_data['responses']
    
    print(f"Sensor Centers: {len(sensor_centers)} bands (ASTER).")
    
    final_data = [] 
    
    # 3. Filter Training Data
    df_split = df[df['situacao'] == 'treino']
    if df_split.empty: return
        
    # 4. Smart Oversampling
    counts = df_split['Material'].value_counts()
    max_count = counts.max()
    print(f"  Oversampling: Merging to {max_count} samples per class.")
    
    for mat, group in df_split.groupby('Material'):
        current_count = len(group)
        needed = max_count - current_count
        
        for _, row in group.iterrows():
            final_data.append({'row': row, 'augment': False})
            
        if needed > 0:
            samples_to_aug = group.sample(needed, replace=True)
            for _, row in samples_to_aug.iterrows():
                final_data.append({'row': row, 'augment': True})
                
    dataset_tensors = []
    np.random.shuffle(final_data)
    
    for item in tqdm(final_data, desc="Gen HiFi ASTER"):
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
            
            # Micro-Augmentation
            if augment:
                emis_high_res = augment_spectrum_micro(emis_high_res)
            
            temp_k = np.random.uniform(280.0, 330.0)
            rad_high_res = planck_law(srf_wavs, temp_k) * emis_high_res
            
            # Convolve to 5 Bands
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
            
    out_path = os.path.join(OUTPUT_DIR, 'hifi_train_dataset_aster.pt')
    torch.save(dataset_tensors, out_path)
    print(f"Saved HiFi ASTER Dataset to {out_path} with {len(dataset_tensors)} samples.")

if __name__ == "__main__":
    create_hifi_dataset_aster()
