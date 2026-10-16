import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pinn import PhysicsInformedNN
from src.config import SENSOR_CONFIG

from typing import Optional, Tuple, Any

def load_model(sensor: str, type: str = 'base', device: str = 'cpu') -> Optional[Any]:
    """
    Loads a trained model checkpoint.
    
    Args:
        sensor (str): Sensor name.
        type (str): 'base' or 'refined'.
        device (str): Torch device.
        
    Returns:
        Optional[nn.Module]: Loaded model or None.
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_bands = SENSOR_CONFIG[sensor]['n_bands']
    
    if type == 'base':
        path = os.path.join(PROJECT_ROOT, 'outputs', f'final_model_{sensor}.pth')
    else:
        primary = os.path.join(PROJECT_ROOT, 'src', 'training', 'fine_tuning', f'refined_model_{sensor}.pth')
        fallback_data = os.path.join(PROJECT_ROOT, 'src', 'data_generation', 'fine_tuning', f'refined_model_{sensor}.pth')
        fallback_out = os.path.join(PROJECT_ROOT, 'outputs', f'refined_model_{sensor}.pth')
        if os.path.exists(primary):
            path = primary
        elif os.path.exists(fallback_data):
            path = fallback_data
        elif os.path.exists(fallback_out):
            path = fallback_out
        else:
            path = primary
        
    if not os.path.exists(path):
        print(f"Warning: Model not found at {path}")
        return None

    model = PhysicsInformedNN(input_dim=n_bands, hidden_dim=512, n_layers=4).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {sensor} {type}: {e}")
        return None

def find_sample_index(dataset: Any, keyword: str) -> Tuple[Optional[int], Optional[Any]]:
    """
    Finds a sample in the dataset matching the material keyword.
    
    Args:
        dataset (List[Dict]): The dataset.
        keyword (str): Substring to search in material name.
        
    Returns:
        Tuple[Optional[int], Optional[Any]]: (Index, Sample) or (None, None).
    """
    for i, sample in enumerate(dataset):
        name = sample.get('material', sample.get('mineral_name', '')).lower()
        if keyword.lower() in name:
            return i, sample
    return None, None

def generate_figures() -> None:
    """
    Generates publication-quality comparison figures (Grid Plot & Bar Chart).
    Saves 'fig1_emissivity_comparison.png' and 'fig2_temperature_accuracy.png'.
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Materials to hunt for
    target_materials = ['feldspar', 'quartz', 'kaolinite']
    material_labels = {'feldspar': 'Best Case (Feldspar)', 'quartz': 'Medium Case (Quartz)', 'kaolinite': 'Worst Case (Kaolinite)'}

    # Data container: {sensor: {material: {truth, base, refined, wavs}}}
    results = {'hytes': {}, 'aster': {}}
    
    # Load everything first
    for sensor in ['hytes', 'aster']:
        print(f"--- Processing {sensor.upper()} ---")
        
        # Load Dataset
        data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets', f'val_dataset_{sensor}.pt')
        if not os.path.exists(data_path):
            print(f"Dataset missing for {sensor}")
            continue
        dataset = torch.load(data_path)
        
        # Load Models
        base_model = load_model(sensor, 'base', device)
        refined_model = load_model(sensor, 'refined', device)
        
        # Wavelengths
        if sensor == 'hytes':
            wavs = np.linspace(7.5, 12.0, 256)
        else:
            wavs = np.array(SENSOR_CONFIG['aster']['fallback_params']['centers'])
            
        for mat in target_materials:
            idx, sample = find_sample_index(dataset, mat)
            if idx is None:
                print(f"Material {mat} not found in {sensor} dataset")
                continue
                
            l_obs = sample['input_radiance'].unsqueeze(0).to(device)
            e_true = sample['target_emissivity'].cpu().numpy()
            t_true = sample['target_temp'].item()
            
            # Inference
            res = {'e_true': e_true, 't_true': t_true, 'wavs': wavs}
            
            if base_model:
                with torch.no_grad():
                    t, e = base_model(l_obs)
                    res['e_base'] = e.squeeze(0).cpu().numpy()
                    res['t_base'] = t.item()
            
            if refined_model:
                with torch.no_grad():
                    t, e = refined_model(l_obs)
                    res['e_refined'] = e.squeeze(0).cpu().numpy()
                    res['t_refined'] = t.item()
            
            results[sensor][mat] = res

    # --- Figure 1: Emissivity Comparison (Grid 3x2) ---
    print("Generating Figure 1...")
    fig1, axes = plt.subplots(3, 2, figsize=(12, 12), sharex='col')
    
    # Rows: Materials, Cols: Sensors
    for row, mat in enumerate(target_materials):
        for col, sensor in enumerate(['hytes', 'aster']):
            ax = axes[row, col]
            data = results[sensor].get(mat)
            
            if not data:
                ax.text(0.5, 0.5, "Data Missing", ha='center')
                continue
                
            wavs = data['wavs']
            
            # Plot Curves
            marker = 'o' if sensor == 'aster' else None
            
            # Truth
            ax.plot(wavs, data['e_true'], color='black', linestyle='--', alpha=0.8, label='Ground Truth', marker=marker)
            
            # Base
            if 'e_base' in data:
                rmse_base = np.sqrt(np.mean((data['e_base'] - data['e_true'])**2))
                ax.plot(wavs, data['e_base'], color='royalblue', linestyle=':', alpha=0.7, linewidth=2, label='Base Model', marker=marker)
                
            # Refined
            if 'e_refined' in data:
                rmse_ref = np.sqrt(np.mean((data['e_refined'] - data['e_true'])**2))
                ax.plot(wavs, data['e_refined'], color='crimson', linestyle='-', linewidth=2.0, alpha=0.9, label='Refined Model', marker=marker)
                
                # Annotation
                text = f"RMSE Refined: {rmse_ref:.4f}"
                if 'e_base' in data:
                     text += f"\nRMSE Base: {rmse_base:.4f}"
                ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            # Styling
            if row == 0: ax.set_title(f"Sensor: {sensor.upper()}", fontsize=14, fontweight='bold')
            if col == 0: ax.set_ylabel(f"{material_labels[mat]}\nEmissivity", fontsize=11)
            if row == 2: ax.set_xlabel(r'Wavelength ($\mu m$)')
            
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_ylim(0, 1.05)
            
            # Legend only on first plot
            if row == 0 and col == 1:
                ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig('fig1_emissivity_comparison.png', dpi=300)
    print("Saved fig1_emissivity_comparison.png")
    plt.close()

    # --- Figure 2: Temperature Accuracy (Bar Chart) ---
    print("Generating Figure 2...")
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(target_materials))
    width = 0.2
    
    # Data extraction
    hytes_base = []
    hytes_ref = []
    aster_base = []
    aster_ref = []
    
    for mat in target_materials:
        # HyTES
        h_data = results['hytes'].get(mat)
        if h_data and 't_base' in h_data: hytes_base.append(abs(h_data['t_base'] - h_data['t_true']))
        else: hytes_base.append(0)
            
        if h_data and 't_refined' in h_data: hytes_ref.append(abs(h_data['t_refined'] - h_data['t_true']))
        else: hytes_ref.append(0)
        
        # ASTER
        a_data = results['aster'].get(mat)
        if a_data and 't_base' in a_data: aster_base.append(abs(a_data['t_base'] - a_data['t_true']))
        else: aster_base.append(0)
            
        if a_data and 't_refined' in a_data: aster_ref.append(abs(a_data['t_refined'] - a_data['t_true']))
        else: aster_ref.append(0)

    # Bars
    rects1 = ax.bar(x - 1.5*width, hytes_base, width, label='HyTES Base', color='skyblue')
    rects2 = ax.bar(x - 0.5*width, hytes_ref, width, label='HyTES Refined', color='navy')
    rects3 = ax.bar(x + 0.5*width, aster_base, width, label='ASTER Base', color='lightgray')
    rects4 = ax.bar(x + 1.5*width, aster_ref, width, label='ASTER Refined', color='darkred')
    
    # Labels
    ax.set_ylabel('Absolute Temperature Error (K)', fontsize=12)
    ax.set_title('Temperature Retrieval Accuracy: Generalist vs Specialist', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in target_materials], fontsize=11)
    ax.legend()
    
    # Threshold Line
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target < 1.0 K')
    ax.text(x[-1], 1.05, 'Target Accuracy (1.0 K)', color='green', fontsize=9, ha='right')

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    try:
        plt.tight_layout()
        plt.savefig('fig2_temperature_accuracy.png', dpi=300)
        print("Saved fig2_temperature_accuracy.png")
    except Exception as e:
        print(f"Error saving Fig 2: {e}")

if __name__ == "__main__":
    generate_figures()
