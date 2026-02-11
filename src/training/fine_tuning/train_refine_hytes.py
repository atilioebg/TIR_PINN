import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os
import argparse
from tqdm import tqdm

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss

# --- Config ---
# Data is now in src/data_generation/fine_tuning
DATA_DIR = os.path.join(PROJECT_ROOT, 'src', 'data_generation', 'fine_tuning')
HIFI_DATA_PATH = os.path.join(DATA_DIR, 'hifi_train_dataset.pt')
PRETRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'final_model_hytes.pth')
# Save refined model locally in training folder or outputs? 
# User wanted structure organization. Let's save in outputs/fine_tuning or strictly in this folder.
# Let's save in this folder for now as per original pattern, but renaming.
OUTPUT_MODEL_PATH = os.path.join(SCRIPT_DIR, 'refined_model_hytes.pth')

LEARNING_RATE = 1e-5  # Extremely Low
EPOCHS = 50           # Short Cycle
BATCH_SIZE = 32

from typing import Dict, Any

class HiFiDataset(Dataset):
    """Custom Dataset for High-Fidelity Training Data."""
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

def train_refine() -> None:
    """
    Executes surgical Fine-Tuning on the HyTES model using HiFi data.
    Uses ultra-low LR and high curvature weights to refine spectral shape.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading HiFi Data from {HIFI_DATA_PATH}...")
    if not os.path.exists(HIFI_DATA_PATH):
        print("Dataset not found. Run prepare_hifi_data.py first.")
        return

    dataset = HiFiDataset(HIFI_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Load Pre-trained Model
    print(f"Loading Pre-trained Model form {PRETRAINED_MODEL_PATH}...")
    model = PhysicsInformedNN(input_dim=256, output_dim=256).to(device)
    
    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 3. Setup Surgical Training
    # Aggressive Shape Loss: Curvature Weight = 2.0 (vs 0.1 originally)
    # Gradient Weight = 2.0 (vs 0.5)
    criterion = SpectralShapeLoss(w_mse_eps=100.0, w_mse_temp=1.0, w_sam=1.0, w_grad=2.0, w_curvature=2.0)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    
    print(f"Starting Fine-Tuning (Refinement) for {EPOCHS} epochs...")
    print(f"LR: {LEARNING_RATE} | Shape/Curve Weights: HIGH")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Refine Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch in pbar:
            l_in = batch['input_radiance'].to(device)
            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)
            
            # Forward
            t_pred, e_pred = model(l_in)
            
            # Loss
            loss = criterion(t_pred, t_true, e_pred, e_true)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Refine Loss: {avg_loss:.4f}")
        
    # Save Refined Model
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"Refined model saved to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    train_refine()
