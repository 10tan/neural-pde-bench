import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any

class BranchNet(nn.Module):
    """Branch Network: encodes input function evaluated at m sensor points."""
    def __init__(self, m_sensors: int = 20, p_basis: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m_sensors, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, p_basis)
        )

    def forward(self, u_sensors: torch.Tensor) -> torch.Tensor:
        return self.net(u_sensors)

class TrunkNet(nn.Module):
    """Trunk Network: encodes query evaluation coordinates (x, t)."""
    def __init__(self, in_dim: int = 2, p_basis: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, p_basis)
        )

    def forward(self, y_coords: torch.Tensor) -> torch.Tensor:
        return self.net(y_coords)

class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) combining Branch and Trunk sub-networks.
    """
    def __init__(self, m_sensors: int = 20, p_basis: int = 40):
        super().__init__()
        self.branch = BranchNet(m_sensors, p_basis)
        self.trunk = TrunkNet(2, p_basis)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors: torch.Tensor, y_coords: torch.Tensor) -> torch.Tensor:
        # u_sensors: (batch, m_sensors)
        # y_coords: (batch, num_points, 2)
        b_out = self.branch(u_sensors) # (batch, p)
        t_out = self.trunk(y_coords)   # (batch, num_points, p)

        # Dot product over basis dimension p
        out = torch.einsum("bp,bnp->bn", b_out, t_out) + self.bias
        return out

class DeepONetSolver:
    """DeepONet Solver wrapper for benchmark evaluation."""

    @staticmethod
    def solve_deeponet_benchmark(nx: int = 50, nt: int = 50) -> Dict[str, Any]:
        start_time = time.perf_counter()

        m_sensors = 20
        model = DeepONet(m_sensors=m_sensors, p_basis=32)

        # Generate sample sensor evaluations of a(x)
        sensor_x = np.linspace(0, 1, m_sensors)
        sensor_vals = np.sin(np.pi * sensor_x)
        u_sensors_tensor = torch.tensor(sensor_vals, dtype=torch.float32).unsqueeze(0)

        # Query points
        gx = np.linspace(0, 1, nx)
        gt = np.linspace(0, 1, nt)
        GX, GT = np.meshgrid(gx, gt)
        y_coords = np.stack([GX.ravel(), GT.ravel()], axis=-1)
        y_coords_tensor = torch.tensor(y_coords, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(u_sensors_tensor, y_coords_tensor).numpy().reshape(nt, nx)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        u_exact = np.sin(np.pi * GX) * np.exp(-0.01 * (np.pi**2) * GT)
        l2_err = np.linalg.norm(pred - u_exact) / np.linalg.norm(u_exact)

        return {
            "x": gx.tolist(),
            "t": gt.tolist(),
            "u_pred": pred.tolist(),
            "u_exact": u_exact.tolist(),
            "l2_error": float(l2_err),
            "solve_time_ms": elapsed_ms,
            "architecture": "DeepONet (Branch-Trunk Operator Network)"
        }
