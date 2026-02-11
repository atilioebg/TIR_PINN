import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedNN(nn.Module):
    """
    Physically Informed Neural Network for Temperature and Emissivity retrieval.
    Architecture:
        Input: At-sensor Radiance (N_bands)
        Backbone: MLP
        Head 1: Surface Temperature (Scalar) - Constrained to physical range
        Head 2: Surface Emissivity (N_bands) - Constrained to [0, 1]
    """
    def __init__(self, 
                 input_dim: int = 256, 
                 output_dim: int = None, 
                 hidden_dim: int = 512, 
                 n_layers: int = 4):
        """
        Initializes the PINN architecture.

        Args:
            input_dim (int): Number of input bands (e.g., 256 for HyTES, 5 for ASTER).
            output_dim (int, optional): Dimension of the emissivity output. If None, equals input_dim.
            hidden_dim (int): Number of neurons in hidden layers.
            n_layers (int): Total number of layers (backbone + heads).
        """
        super(PhysicsInformedNN, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            
        self.backbone = nn.Sequential(*layers)
        
        # Head 1: Temperature
        # Outputs a single scalar
        self.temp_head = nn.Linear(hidden_dim, 1)
        
        # Head 2: Emissivity
        # Outputs vector of same size as input (default) or specific output dimension
        out_d = output_dim if output_dim is not None else input_dim
        self.emis_head = nn.Linear(hidden_dim, out_d)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input radiance tensor of shape (Batch, N_bands).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - t_pred (Batch, 1): Predicted Surface Temperature (Kelvin).
                - e_pred (Batch, N_bands): Predicted Spectral Emissivity (0-1).
        """
        features = self.backbone(x)
        
        # Temperature Prediction
        # Enforce physical constraints: T > 0 K. 
        # Earth surface usually > 200 K.
        # Softplus is smooth ReLU, good for gradients.
        # Adding 200.0 offset ensures we start in a valid physical regime.
        t_raw = self.temp_head(features)
        t_pred = F.softplus(t_raw) + 200.0 
        
        # Emissivity Prediction
        # Must be in [0, 1]. Sigmoid is perfect.
        e_raw = self.emis_head(features)
        e_pred = torch.sigmoid(e_raw)
        
        return t_pred, e_pred
