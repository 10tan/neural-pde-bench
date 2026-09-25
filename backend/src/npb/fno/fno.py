import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any

class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer for Fourier Neural Operator."""
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to retain

        scale = (1.0 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # input: (batch, in_channel, x) -> (batch, in_channel, 1, x)
        # weights: (in_channel, out_channel, x) -> (1, in_channel, out_channel, x)
        return (input.unsqueeze(2) * weights.unsqueeze(0)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]

        # Compute Fourier transform along spatial dimension
        x_ft = torch.fft.rfft(x)

        # Truncate modes to actual available FFT size
        modes = min(self.modes1, x_ft.size(-1))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        # Return to physical domain via Inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    """Full 1D Fourier Neural Operator model."""
    def __init__(self, modes: int = 16, width: int = 32):
        super().__init__()
        self.modes1 = modes
        self.width = width
        
        self.fc0 = nn.Linear(2, self.width) # input (x, a(x)) -> width

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, nx, 2)
        grid = x
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)

        x1 = self.conv1(x) + self.w1(x)
        x = F.gelu(x1)

        x1 = self.conv2(x) + self.w2(x)
        x = F.gelu(x1)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class FNOSolver:
    """FNO Solver wrapper for benchmark evaluation."""
    
    @staticmethod
    def solve_fno_benchmark(nx: int = 100) -> Dict[str, Any]:
        start_time = time.perf_counter()

        model = FNO1d(modes=12, width=32)
        
        # Prepare sample input function a(x) = sin(2*pi*x)
        x_grid = np.linspace(0, 1, nx)
        a_x = np.sin(2.0 * np.pi * x_grid)
        inp = np.stack([x_grid, a_x], axis=-1) # (nx, 2)
        inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0) # (1, nx, 2)

        with torch.no_grad():
            u_pred = model(inp_tensor).squeeze().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Exact target u(x) = sin(2*pi*x) / (1 + 4*pi^2)
        u_exact = a_x / (1.0 + 4.0 * (np.pi**2))
        l2_err = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

        return {
            "x": x_grid.tolist(),
            "a_x": a_x.tolist(),
            "u_pred": u_pred.tolist(),
            "u_exact": u_exact.tolist(),
            "l2_error": float(l2_err),
            "solve_time_ms": elapsed_ms,
            "architecture": "FNO1d (Fourier Neural Operator)"
        }
