import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any

class MultiFidelityDeepONet(nn.Module):
    """
    Multi-Fidelity DeepONet combining low-fidelity model outputs (u_lo)
    with sparse high-fidelity data (u_hi).
    u_hi(x) = alpha * u_lo(x) + residual(x)
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.0]))
        self.residual_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, u_lo: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.scale * u_lo + self.residual_net(coords)

class MultiFidelitySolver:
    """Multi-Fidelity benchmark solver."""

    @staticmethod
    def train_and_eval() -> Dict[str, Any]:
        start_time = time.perf_counter()

        from npb.data.pde_data import SyntheticPDEDataGenerator
        dataset = SyntheticPDEDataGenerator.generate_multifidelity_dataset(num_samples=20)

        x_hi = dataset["x_hi"]
        u_hi = dataset["u_hi"][0]
        u_lo = dataset["u_lo"][0]

        model = MultiFidelityDeepONet(hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        u_lo_t = torch.tensor(u_lo, dtype=torch.float32).unsqueeze(-1)
        u_hi_t = torch.tensor(u_hi, dtype=torch.float32).unsqueeze(-1)
        coords_t = torch.tensor(np.stack([x_hi, np.zeros_like(x_hi)], axis=-1), dtype=torch.float32)

        for _ in range(150):
            optimizer.zero_grad()
            pred_hi = model(u_lo_t, coords_t)
            loss = torch.mean((pred_hi - u_hi_t)**2)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_hi_eval = model(u_lo_t, coords_t).squeeze().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        l2_err_lo = np.linalg.norm(u_lo - u_hi) / np.linalg.norm(u_hi)
        l2_err_mf = np.linalg.norm(pred_hi_eval - u_hi) / np.linalg.norm(u_hi)

        return {
            "x": x_hi.tolist(),
            "u_lo": u_lo.tolist(),
            "u_hi_exact": u_hi.tolist(),
            "u_mf_pred": pred_hi_eval.tolist(),
            "l2_error_low_fidelity": float(l2_err_lo),
            "l2_error_multi_fidelity": float(l2_err_mf),
            "elapsed_ms": elapsed_ms
        }
