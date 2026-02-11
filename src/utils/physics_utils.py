import torch
import numpy as np
from scipy.constants import h, c, k

# Constants
C1 = 2 * h * c**2
C2 = h * c / k

def planck_law_torch(wavelength_um: torch.Tensor, temp_k: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Planck's Law for PyTorch Tensors.
    
    Args:
        wavelength_um (torch.Tensor): Wavelengths in micrometers.
        temp_k (torch.Tensor): Temperature in Kelvin (Batch, 1).
        
    Returns:
        torch.Tensor: Spectral Radiance [W/m^2/sr/um].
    """
    # Ensure inputs are tensors
    if not isinstance(wavelength_um, torch.Tensor):
        wavelength_um = torch.tensor(wavelength_um)
        
    w_m = wavelength_um * 1e-6  # um to m
    
    # B(lambda, T) = (2hc^2 / lambda^5) * 1 / (exp(hc/lambda*k*T) - 1)
    # Note: temp_k shape (Batch, 1), wavelength (N_bands) -> Result (Batch, N_bands)
    
    val = (C1 / (w_m**5)) / (torch.exp(C2 / (w_m * temp_k)) - 1)
    
    # Scale to W/m^2/sr/um (factor of 1e-6)
    return val * 1e-6

def radiative_transfer_equation(emis: torch.Tensor, 
                                temp: torch.Tensor, 
                                wavelength: torch.Tensor, 
                                tau: torch.Tensor, 
                                up: torch.Tensor, 
                                down: torch.Tensor) -> torch.Tensor:
    """
    Calculates At-Sensor Radiance using the RTE.
    L_sensor = tau * eps * B(T) + L_u + tau * (1 - eps) * L_d
    
    Args:
        emis (torch.Tensor): Emissivity (0-1).
        temp (torch.Tensor): Temperature (K).
        wavelength (torch.Tensor): Band wavelengths.
        tau (torch.Tensor): Transmission.
        up (torch.Tensor): Upwelling Radiance (Path).
        down (torch.Tensor): Downwelling Radiance (Sky).
        
    Returns:
        torch.Tensor: Simulated Sensor Radiance.
    """
    B_T = planck_law_torch(wavelength, temp)
    
    # Standard RTE
    # Surface Leaving = eps * B(T) + (1-eps) * L_down
    # At Sensor = L_up + tau * Surface_Leaving
    
    L_surf = emis * B_T + (1.0 - emis) * down
    L_sensor = up + tau * L_surf
    
    return L_sensor
