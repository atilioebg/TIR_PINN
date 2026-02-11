import os
import pickle
import numpy as np
from tqdm import tqdm

def create_synthetic_dataset(ground_truth_path: str, lut_dir: str, output_path: str) -> None:
    """
    Combines Ground Truth (Surface Params) + Atmospheric LUTs -> Sensor Radiance.
    
    Equation: L_sensor = L_up + tau * [ eps * B(T) + (1-eps)*L_down ]
    
    Args:
        ground_truth_path (str): Path to ground truth pickle.
        lut_dir (str): Directory with atmospheric tables.
        output_path (str): Output path for the dataset.
    """
    print("--- Starting Synthetic Data Generation ---")
    
    # 1. Load Ground Truth (Surface: T, Emis, Wavs)
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        return
        
    with open(ground_truth_path, 'rb') as f:
        gt_data = pickle.load(f) # List of dicts
        
    # 2. Iterate/Load Atmospheric LUTs
    # For simplicity, we mock a single generic atmosphere or minimal set if files exist.
    # Ideally, we read the catalog csv from automate_modtran.
    
    # MOCKING ATMOSPHERE FOR DEMONSTRATION if no real files yet
    # In real usage: parse the 'atmospheric_luts' folder.
    
    synthetic_samples = []
    
    # Verify we have at least one sample to get wavelengths
    if len(gt_data) == 0:
        return
    
    sample_wavs = gt_data[0]['wavelengths']
    num_bands = len(sample_wavs)
    
    # Create a dummy atmosphere if none exists (just to allow training code to run)
    # L_up approx 0, Trans approx 0.8, L_down approx 0
    dummy_tau = np.ones(num_bands) * 0.9
    dummy_up = np.zeros(num_bands)
    dummy_down = np.ones(num_bands) * 1e-6 # Small downwelling
    
    # We mix each GT sample with an atmosphere
    # To scale up: Loop through multiple atmospheres per mineral sample
    
    print(f"combining {len(gt_data)} surface samples with atmospheric models...")
    
    for item in tqdm(gt_data):
        # Unpack surface
        T = item['temperature_k']
        emis = item['emissivity']
        B_T = item['blackbody_radiance'] # Already computed
        
        # Apply RTE
        # L_surf = emis * B_T + (1 - emis) * L_down
        # L_sensor = L_up + tau * L_surf
        
        # Using dummy atm for now (Replace with real LUT loop)
        L_surf = emis * B_T + (1.0 - emis) * dummy_down
        L_sensor = dummy_up + dummy_tau * L_surf
        
        sample = {
            'sensor_radiance': L_sensor, # INPUT TO NN
            'wavelengths': item['wavelengths'],
            'true_temp': T,               # TARGET
            'true_emis': emis,            # TARGET
            'tau': dummy_tau,             # FOR PHYSICS LOSS
            'up': dummy_up,               # FOR PHYSICS LOSS
            'down': dummy_down            # FOR PHYSICS LOSS
        }
        synthetic_samples.append(sample)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(synthetic_samples, f)
        
    print(f"Saved synthetic dataset to {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    GT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'ground_truth', 'ground_truth_dataset.pkl')
    LUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'atmospheric_luts')
    OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'synthetic_dataset', 'train_dataset.pkl')
    
    create_synthetic_dataset(GT_PATH, LUT_DIR, OUT_PATH)
