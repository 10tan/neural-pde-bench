import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, List

class SubdomainPINN(nn.Module):
    """Subdomain Neural Network for Extended PINN (XPINN)."""
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)

class XPINNTrainer:
    """
    Extended PINN (XPINN) trainer with domain decomposition
    and interface-continuity penalty loss.
    """
    def __init__(self, interface_x: float = 0.5):
        self.interface_x = interface_x
        self.subdomain1_net = SubdomainPINN(hidden_dim=32) # x in [0, 0.5]
        self.subdomain2_net = SubdomainPINN(hidden_dim=32) # x in [0.5, 1.0]

        self.opt1 = torch.optim.Adam(self.subdomain1_net.parameters(), lr=1e-3)
        self.opt2 = torch.optim.Adam(self.subdomain2_net.parameters(), lr=1e-3)

    def train_xpinn(self, epochs: int = 200, n_sub_pts: int = 150) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Collocation points Subdomain 1: x in [0, 0.5]
        pts_sub1 = torch.cat([
            torch.rand(n_sub_pts, 1) * self.interface_x,
            torch.rand(n_sub_pts, 1)
        ], dim=1)

        # Collocation points Subdomain 2: x in [0.5, 1.0]
        pts_sub2 = torch.cat([
            self.interface_x + torch.rand(n_sub_pts, 1) * (1.0 - self.interface_x),
            torch.rand(n_sub_pts, 1)
        ], dim=1)

        # Interface points: x = 0.5, t in [0, 1]
        t_int = torch.linspace(0, 1, 50).unsqueeze(1)
        pts_int = torch.cat([
            torch.full_like(t_int, self.interface_x),
            t_int
        ], dim=1)

        loss_history = []

        for epoch in range(epochs):
            self.opt1.zero_grad()
            self.opt2.zero_grad()

            # Subdomain 1 PDE Residual
            pts_sub1.requires_grad_(True)
            u1 = self.subdomain1_net(pts_sub1)
            grads1 = torch.autograd.grad(u1, pts_sub1, torch.ones_like(u1), create_graph=True)[0]
            u1_x, u1_t = grads1[:, 0:1], grads1[:, 1:2]
            u1_xx = torch.autograd.grad(u1_x, pts_sub1, torch.ones_like(u1_x), create_graph=True)[0][:, 0:1]
            res1 = u1_t - 0.01 * u1_xx

            # Subdomain 2 PDE Residual
            pts_sub2.requires_grad_(True)
            u2 = self.subdomain2_net(pts_sub2)
            grads2 = torch.autograd.grad(u2, pts_sub2, torch.ones_like(u2), create_graph=True)[0]
            u2_x, u2_t = grads2[:, 0:1], grads2[:, 1:2]
            u2_xx = torch.autograd.grad(u2_x, pts_sub2, torch.ones_like(u2_x), create_graph=True)[0][:, 0:1]
            res2 = u2_t - 0.01 * u2_xx

            loss_pde = torch.mean(res1**2) + torch.mean(res2**2)

            # Interface Continuity Loss: u1 = u2 and du1/dx = du2/dx
            pts_int.requires_grad_(True)
            u1_int = self.subdomain1_net(pts_int)
            u2_int = self.subdomain2_net(pts_int)

            loss_interface_val = torch.mean((u1_int - u2_int)**2)

            grads1_int = torch.autograd.grad(u1_int, pts_int, torch.ones_like(u1_int), create_graph=True)[0][:, 0:1]
            grads2_int = torch.autograd.grad(u2_int, pts_int, torch.ones_like(u2_int), create_graph=True)[0][:, 0:1]
            loss_interface_flux = torch.mean((grads1_int - grads2_int)**2)

            total_loss = loss_pde + 10.0 * loss_interface_val + 1.0 * loss_interface_flux

            total_loss.backward()
            self.opt1.step()
            self.opt2.step()

            loss_history.append({
                "epoch": epoch,
                "total_loss": float(total_loss.item()),
                "pde_loss": float(loss_pde.item()),
                "interface_val_loss": float(loss_interface_val.item()),
                "interface_flux_loss": float(loss_interface_flux.item())
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Construct full solution from subdomains
        gx = np.linspace(0, 1, 60)
        gt = np.linspace(0, 1, 60)
        GX, GT = np.meshgrid(gx, gt)

        u_pred_full = np.zeros((60, 60))
        with torch.no_grad():
            for i in range(60):
                for j in range(60):
                    px = GX[i, j]
                    pt = GT[i, j]
                    inp = torch.tensor([[px, pt]], dtype=torch.float32)
                    if px <= self.interface_x:
                        u_pred_full[i, j] = float(self.subdomain1_net(inp).item())
                    else:
                        u_pred_full[i, j] = float(self.subdomain2_net(inp).item())

        return {
            "loss_history": loss_history,
            "interface_x": self.interface_x,
            "u_pred": u_pred_full.tolist(),
            "train_time_ms": elapsed_ms
        }
