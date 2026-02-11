import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

# Add project root
# Now 4 levels up: src/training/base/train.py -> base -> training -> src -> IATIR
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss
from src.config import SENSOR_CONFIG

def train_model(sensor: str = 'hytes', epochs: int = 300, w_curvature: float = 0.5) -> None:
    """
    Trains the Generalist PINN Model (Base Model).
    
    Args:
        sensor (str): Sensor name ('hytes' or 'aster').
        epochs (int): Number of training epochs.
        w_curvature (float): Weight for the curvature loss component (Shape Awareness).
                             Use 0.0 for ASTER (5 bands) to avoid instability.
    """
    # --- Configuration ---
    # Now 4 levels up
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'synthetic_dataset', f'synthetic_dataset_{sensor}.pt')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if sensor not in SENSOR_CONFIG:
        print(f"Unknown sensor: {sensor}")
        return

    n_bands = SENSOR_CONFIG[sensor]['n_bands']
    print(f"Training for Sensor: {sensor.upper()} ({n_bands} bands)")
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Load Data ---
    # --- Load Data --- (Updated for Final Datasets)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    TRAIN_PATH = os.path.join(DATA_DIR, f'train_dataset_{sensor}.pt')
    VAL_PATH = os.path.join(DATA_DIR, f'val_dataset_{sensor}.pt')
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        print(f"Datasets not found at {DATA_DIR}. Run create_final_dataset.py --sensor {sensor} first.")
        return
        
    print(f"Loading training data from {TRAIN_PATH}...")
    train_dataset = torch.load(TRAIN_PATH)
    print(f"Loading validation data from {VAL_PATH}...")
    val_dataset = torch.load(VAL_PATH)
    
    # train_dataset and val_dataset are already lists of dicts

    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Initialize Model ---
    # Dynamic input dimension
    model = PhysicsInformedNN(input_dim=n_bands, output_dim=n_bands, hidden_dim=512, n_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # --- Initialize Shape Loss ---
    # Adaptive weights: ASTER (5 bands) shouldn't use high curvature weight
    if sensor == 'aster':
        print(f"Sensor is ASTER: Forcing w_curvature=0.0 to prevent instability.")
        w_curve = 0.0
    else:
        # Use user provided weight for HyTES
        w_curve = w_curvature
        
    criterion = SpectralShapeLoss(w_curvature=w_curve).to(device)
    print(f"Loss Config: w_curvature={w_curve}")
    
    # --- Training Loop ---
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            l_obs = batch['input_radiance'].to(device)
            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)
            
            optimizer.zero_grad()
            
            t_pred, e_pred = model(l_obs)
            
            # Shape Loss Check
            loss = criterion(t_pred, t_true, e_pred, e_true)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                l_obs = batch['input_radiance'].to(device)
                t_true = batch['target_temp'].to(device)
                e_true = batch['target_emissivity'].to(device)
                
                t_pred, e_pred = model(l_obs)
                
                loss = criterion(t_pred, t_true, e_pred, e_true)
                val_running_loss += loss.item()
                
        avg_val_loss = val_running_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Checkpoint (only periodically)
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f'model_{sensor}_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    print(f"Saving final model to {final_path} (Note: This overwrites previous model. Use --sensor to load correct one during inference implies we might need specific naming)")
    # Actually, let's allow overwrite for now as per user workflow, but maybe warn?
    # User requested `final_model.pth` usage in viz.
    torch.save(model.state_dict(), final_path)
    
    # Also save sensor specific for safety
    sensor_model_path = os.path.join(OUTPUT_DIR, f'final_model_{sensor}.pth')
    torch.save(model.state_dict(), sensor_model_path)
    print(f"Also saved to {sensor_model_path}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Convergence ({sensor.upper()})')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'loss_curve_{sensor}.png'))
    plt.close() # Ensure close
    print("Training complete. Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--sensor', type=str, default='hytes', choices=['hytes', 'aster'],
                        help='Sensor to train for')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--w_curvature', type=float, default=0.5, help='Weight for curvature loss (Shape Awareness)')
    args = parser.parse_args()
    
    train_model(sensor=args.sensor, epochs=args.epochs, w_curvature=args.w_curvature)
