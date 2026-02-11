import torch
import torch.nn as nn
import numpy as np

class SpectralShapeLoss(nn.Module):
    """
    Spectral Shape Loss V2:
    Enforces shape similarity using:
    1. MSE (Amplitude)
    2. SAM (Vector Alignment)
    3. Gradient L1 (1st Derivative / Slope)
    4. Curvature L1 (2nd Derivative / Concavity)
    
    Loss = w1*MSE + w2*MSE_Temp + w3*SAM + w4*Grad_L1 + w5*Curve_L1
    """
    def __init__(self, 
                 w_mse_eps: float = 100.0, 
                 w_mse_temp: float = 1.0, 
                 w_sam: float = 1.0, 
                 w_grad: float = 0.5, 
                 w_curvature: float = 0.1):
        """
        Initializes the loss weights.

        Args:
            w_mse_eps (float): Weight for Emissivity MSE loss.
            w_mse_temp (float): Weight for Temperature MSE loss.
            w_sam (float): Weight for Spectral Angle Mapper loss.
            w_grad (float): Weight for Gradient (1st derivative) L1 loss.
            w_curvature (float): Weight for Curvature (2nd derivative) L1 loss.
        """
        super(SpectralShapeLoss, self).__init__()
        
        # Weights
        self.w_mse_eps = w_mse_eps
        self.w_mse_temp = w_mse_temp
        self.w_sam = w_sam
        self.w_grad = w_grad
        self.w_curve = w_curvature      # Weight for 2nd derivative (Curvature)
        
    def sam_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes 1 - Cosine Similarity (SAM Loss).

        Args:
            pred (torch.Tensor): Predicted tensor (Batch, N).
            target (torch.Tensor): Target tensor (Batch, N).

        Returns:
            torch.Tensor: Scalar mean SAM loss.
        """
        dot_product = (pred * target).sum(dim=1)
        norms = pred.norm(dim=1) * target.norm(dim=1)
        # Avoid div by zero
        sam_val = 1 - (dot_product / (norms + 1e-8))
        return sam_val.mean()

    def derivative_losses(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes L1 loss for 1st and 2nd derivatives (Slope and Curvature).

        Args:
            pred (torch.Tensor): Predicted emissivity (Batch, N).
            target (torch.Tensor): Target emissivity (Batch, N).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Gradient Loss, Curvature Loss).
        """
        # 1st Derivative: diff[i] = x[i+1] - x[i]
        diff1_pred = pred[:, 1:] - pred[:, :-1]
        diff1_target = target[:, 1:] - target[:, :-1]
        
        loss_grad = nn.L1Loss()(diff1_pred, diff1_target)
        
        # 2nd Derivative: diff2[i] = diff1[i+1] - diff1[i]
        diff2_pred = diff1_pred[:, 1:] - diff1_pred[:, :-1]
        diff2_target = diff1_target[:, 1:] - diff1_target[:, :-1]
        
        loss_curve = nn.L1Loss()(diff2_pred, diff2_target)
        
        return loss_grad, loss_curve

    def forward(self, 
                t_pred: torch.Tensor, 
                t_true: torch.Tensor, 
                eps_pred: torch.Tensor, 
                eps_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the total hybrid loss.

        Args:
            t_pred (torch.Tensor): Predicted Temperature (Batch, 1).
            t_true (torch.Tensor): True Temperature (Batch, 1).
            eps_pred (torch.Tensor): Predicted Emissivity (Batch, N_bands).
            eps_true (torch.Tensor): True Emissivity (Batch, N_bands).

        Returns:
            torch.Tensor: Total weighted loss value.
        """
        # 1. Standard MSE
        loss_mse_e = nn.MSELoss()(eps_pred, eps_true)
        loss_mse_t = nn.MSELoss()(t_pred, t_true)
        
        # 2. Shape Losses (Applied to Emissivity Only)
        loss_sam = self.sam_loss(eps_pred, eps_true)
        loss_grad, loss_curve = self.derivative_losses(eps_pred, eps_true)
        
        # 3. Combine
        # Using L1 for derivatives is more robust to sharp changes
        
        total_loss = (self.w_mse_eps * loss_mse_e) + \
                     (self.w_mse_temp * loss_mse_t) + \
                     (self.w_sam * loss_sam) + \
                     (self.w_grad * loss_grad) + \
                     (self.w_curve * loss_curve)
        
        return total_loss
