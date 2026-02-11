import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

# Add project root to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pinn import PhysicsInformedNN
from src.config import SENSOR_CONFIG

def run_visualization(
    mode: str = 'emissivity', 
    sensor: str = 'hytes', 
    model_type: str = 'base', 
    data: str = 'single', 
    save_only: bool = False
) -> None:
    """
    Generates validation plots for the trained models.
    
    Args:
        mode (str): Visualization mode ('prediction', 'emissivity', 'temperature').
        sensor (str): Sensor to visualize ('hytes', 'aster').
        model_type (str): 'base' (Generalist) or 'refined' (Specialist).
        data (str): 'single' or 'combined' (overlay both models).
        save_only (bool): If True, saves plot without opening window.
    """
    # --- Configuration ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if model_type == 'base':
        MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs', f'final_model_{sensor}.pth')
    else: # refined
        MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'model_fine_tuning', f'refined_model_{sensor}.pth')
        
    # Optional Base Model for comparison
    MODEL_PATH_BASE = os.path.join(PROJECT_ROOT, 'outputs', f'final_model_{sensor}.pth')

    # Dynamic dataset path based on sensor (Use Validation Set for visualization)
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets', f'val_dataset_{sensor}.pt')
    
    # Determine output name based on mode
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PLOT = os.path.join(OUTPUT_DIR, f'validation_plot_{mode}_{sensor}_{model_type}_{data}.png')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Visualizing: {sensor.upper()} | {mode} | {model_type} | {data} ---")
    
    # --- Load Configuration ---
    if sensor not in SENSOR_CONFIG:
        print(f"Error: Unknown sensor '{sensor}'. Available: {list(SENSOR_CONFIG.keys())}")
        return
    
    n_bands = SENSOR_CONFIG[sensor]['n_bands']

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Skipping.")
        return

    # Initialize model with correct dimensions
    model = PhysicsInformedNN(input_dim=n_bands, hidden_dim=512, n_layers=4).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()
    
    # --- Load Base Model (Comparison) ---
    model_base = None
    # Updated Logic: 'combined' triggers overlay
    if data == 'combined' and model_type == 'refined':
        if os.path.exists(MODEL_PATH_BASE):
            model_base = PhysicsInformedNN(input_dim=n_bands, hidden_dim=512, n_layers=4).to(device)
            try:
                model_base.load_state_dict(torch.load(MODEL_PATH_BASE, map_location=device))
                model_base.eval()
            except RuntimeError:
                print("Warning: Failed to load base model for comparison. Skipping overlay.")
                model_base = None
        else:
            print("Warning: Base model file not found. Skipping overlay.")

    # --- Load Dataset ---
    if not os.path.exists(DATA_PATH):
        print("Dataset file not found.")
        return
        
    dataset = torch.load(DATA_PATH)

    # --- Select Samples ---
    unique_targets = {}
    for idx, sample in enumerate(dataset):
        name = sample.get('material', sample.get('mineral_name', 'Unknown'))
        if name not in unique_targets:
            unique_targets[name] = idx
            
    target_names = sorted(unique_targets.keys())
    indices = [unique_targets[name] for name in target_names]
    n_samples = len(indices)
    
    # --- Setup Plot ---
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 5 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        min_name = sample.get('material', sample.get('mineral_name', f'Sample {idx}'))
        
        l_obs = sample['input_radiance'].unsqueeze(0).to(device)
        t_true = sample['target_temp'].item()
        e_true = sample['target_emissivity'].cpu().numpy()

        with torch.no_grad():
            t_pred_tensor, e_pred_tensor = model(l_obs)
            t_pred = t_pred_tensor.item()
            e_pred = e_pred_tensor.squeeze(0).cpu().numpy()
            
            # Base Model Prediction (Only if loaded)
            t_pred_base, e_pred_base = None, None
            if model_base:
                t_b, e_b = model_base(l_obs)
                t_pred_base = t_b.item()
                e_pred_base = e_b.squeeze(0).cpu().numpy()

        # Determine Wavelengths
        if 'wavelengths' in sample:
            wavs = sample['wavelengths']
            if isinstance(wavs, torch.Tensor):
                wavs = wavs.cpu().numpy()
        else:
             if sensor == 'hytes':
                wavs = np.linspace(7.5, 12.0, 256)
             elif sensor == 'aster':
                wavs = np.array(SENSOR_CONFIG['aster']['fallback_params']['centers'])
             else:
                 wavs = np.arange(n_bands)

        ax = axes[i]
        
        # --- Plotting Logic ---
        if mode == 'prediction':
            marker = 'o' if n_bands < 20 else None
            ax.plot(wavs, e_true, label='Truth', color='black', linestyle='--', linewidth=2, marker=marker)
            
            if e_pred_base is not None:
                ax.plot(wavs, e_pred_base, label='Base (Generalist)', color='royalblue', linestyle=':', linewidth=2, alpha=0.8, marker=marker)
            
            ax.plot(wavs, e_pred, label=f'Refined (Specialist)', color='crimson', linewidth=2, alpha=0.9, marker=marker)
            
            title_text = f"{min_name} - Emissivity Retrieval ({sensor.upper()})"
            ax.set_title(title_text, fontsize=12, fontweight='bold')
            ax.set_xlabel(r'Wavelength ($\mu m$)')
            ax.set_ylabel('Emissivity')
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower left', fontsize=9)
            
        elif mode == 'emissivity':
            e_true_mean = np.mean(e_true)
            e_pred_mean = np.mean(e_pred)
            
            labels = ['True', 'Refined']
            values = [e_true_mean, e_pred_mean]
            colors = ['gray', 'crimson']
            
            if e_pred_base is not None:
                e_base_mean = np.mean(e_pred_base)
                labels.insert(1, 'Base')
                values.insert(1, e_base_mean)
                colors.insert(1, 'royalblue')

            bars = ax.bar(labels, values, color=colors, width=0.4)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"{min_name} - Mean Emissivity", fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Emissivity')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            rmse = np.sqrt(np.mean((e_pred - e_true)**2))
            info_text = f'RMSE Refined = {rmse:.4f}'
            if e_pred_base is not None:
                rmse_base = np.sqrt(np.mean((e_pred_base - e_true)**2))
                info_text += f'\nRMSE Base = {rmse_base:.4f}'
                
            ax.text(0.5, 0.9, info_text, transform=ax.transAxes, ha='center', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        elif mode == 'temperature':
            labels = ['True', 'Refined']
            values = [t_true, t_pred]
            colors = ['gray', 'dodgerblue']

            if t_pred_base is not None:
                labels.insert(1, 'Base')
                values.insert(1, t_pred_base)
                colors.insert(1, 'lightblue')

            bars = ax.bar(labels, values, color=colors, width=0.4)
            
            # Dynamic Zoom Y
            all_vals = values
            min_val = min(all_vals) - 5
            max_val = max(all_vals) + 5
            ax.set_ylim(min_val, max_val)
            
            ax.set_title(f"{min_name} - Temperature Retrieval", fontsize=12, fontweight='bold')
            ax.set_ylabel('Kelvin')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.2f} K', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            delta_t = t_pred - t_true
            info_text = f'$\\Delta T$ Refined = {delta_t:.2f} K'
            if t_pred_base is not None:
                delta_t_base = t_pred_base - t_true
                info_text += f'\n$\\Delta T$ Base = {delta_t_base:.2f} K'
                
            ax.text(0.5, 0.9, info_text, transform=ax.transAxes, ha='center', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
            ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    try:
        plt.savefig(OUTPUT_PLOT, dpi=300)
        print(f"Validation plot saved to: {OUTPUT_PLOT}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    if not save_only:
        print("Opening interactive plot window... Close it to finish.")
        plt.show()
    else:
        plt.close(fig) # Explicitly close figure to free memory in loops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results')
    parser.add_argument('--mode', type=str, default='prediction', 
                        choices=['prediction', 'emissivity', 'temperature', 'all'],
                        help='Visualization mode')
    parser.add_argument('--sensor', type=str, default='hytes', 
                        choices=['hytes', 'aster', 'all'],
                        help='Sensor to visualize')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'refined', 'all'],
                        help='Model type: base (generalist) or refined (specialist)')
    parser.add_argument('--data', type=str, default='single',
                        choices=['single', 'combined', 'all'],
                        help="Data overlay: 'single' (only selected model) or 'combined' (overlay refined + base)")
    parser.add_argument('--save_only', action='store_true', help="Save plot without opening interactive window")
                        
    args = parser.parse_args()
    
    # --- Dispatcher Logic for 'all' arguments ---
    
    sensors = ['hytes', 'aster'] if args.sensor == 'all' else [args.sensor]
    modes = ['prediction', 'emissivity', 'temperature'] if args.mode == 'all' else [args.mode]
    models = ['base', 'refined'] if args.model_type == 'all' else [args.model_type]
    
    # Data logic maps 'all' to ['single', 'combined']
    # 'combined' is only valid if model_type is 'refined'
    expanded_data = []
    if args.data == 'all':
        expanded_data = ['single', 'combined']
    else:
        expanded_data = [args.data]
        
    for s in sensors:
        for m in modes:
            for mt in models:
                for d in expanded_data:
                    # VALIDATION LOGIC
                    
                    # 1. Base model cannot have 'combined' (Base + Base overlay is redundant/undefined)
                    if mt == 'base' and d == 'combined':
                        continue 
                        
                    # Execute
                    run_visualization(mode=m, sensor=s, model_type=mt, data=d, save_only=args.save_only)
