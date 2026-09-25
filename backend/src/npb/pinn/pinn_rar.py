import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Any

class PINN(nn.Module):
    """
    Standard Physics-Informed Neural Network for 1D/2D PDEs.
    """
    def __init__(self, in_dim: int = 2, hidden_dim: int = 32, num_layers: int = 4, out_dim: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PINNTrainerWithRAR:
    """
    PINN Trainer featuring Residual-based Adaptive Refinement (RAR).
    Enables dynamic allocation of collocation points in regions with high PDE residual.
    """
    def __init__(self, pde_type: str = "heat_1d", alpha: float = 0.01):
        self.pde_type = pde_type
        self.alpha = alpha
        self.model = PINN(in_dim=2, hidden_dim=32, num_layers=4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def pde_residual(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Computes PDE residual via Automatic Differentiation:
        Heat 1D: u_t - alpha * u_xx - f(x,t) = 0
        """
        xt.requires_grad_(True)
        u = self.model(xt)

        # Gradients
        grads = torch.autograd.grad(
            u, xt, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]

        x = xt[:, 0:1]
        t = xt[:, 1:2]
        omega = 2.0
        # Source term f(x,t) for MMS
        f = -omega * torch.sin(np.pi * x) * torch.sin(omega * t) - self.alpha * (np.pi**2) * torch.sin(np.pi * x) * torch.cos(omega * t)

        res = u_t - self.alpha * u_xx - f
        return res

    def train_with_rar(
        self, 
        n_initial_collocation: int = 200, 
        rar_iterations: int = 3, 
        epochs_per_iter: int = 150, 
        points_per_rar: int = 20
    ) -> Dict[str, Any]:
        """
        Runs PINN training with RAR collocation point adaptive sampling.
        """
        start_time = time.perf_counter()

        # Initial collocation points in [0,1] x [0,1]
        colloc_pts = torch.rand(n_initial_collocation, 2, dtype=torch.float32)
        
        # Initial condition (t=0): u(x,0) = sin(pi * x)
        x_ic = torch.linspace(0, 1, 50).unsqueeze(1)
        t_ic = torch.zeros_like(x_ic)
        ic_pts = torch.cat([x_ic, t_ic], dim=1)
        ic_vals = torch.sin(np.pi * x_ic)

        # Boundary condition (x=0, x=1): u(0,t) = 0, u(1,t) = 0
        t_bc = torch.linspace(0, 1, 50).unsqueeze(1)
        bc_left = torch.cat([torch.zeros_like(t_bc), t_bc], dim=1)
        bc_right = torch.cat([torch.ones_like(t_bc), t_bc], dim=1)

        loss_history = []
        rar_added_points = []

        for rar_step in range(rar_iterations + 1):
            for epoch in range(epochs_per_iter):
                self.optimizer.zero_grad()

                # PDE loss
                res = self.pde_residual(colloc_pts)
                loss_pde = torch.mean(res**2)

                # IC loss
                u_ic_pred = self.model(ic_pts)
                loss_ic = torch.mean((u_ic_pred - ic_vals)**2)

                # BC loss
                u_bc_l = self.model(bc_left)
                u_bc_r = self.model(bc_right)
                loss_bc = torch.mean(u_bc_l**2) + torch.mean(u_bc_r**2)

                total_loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
                total_loss.backward()
                self.optimizer.step()

                loss_history.append({
                    "step": len(loss_history),
                    "total_loss": float(total_loss.item()),
                    "loss_pde": float(loss_pde.item()),
                    "loss_ic": float(loss_ic.item()),
                    "loss_bc": float(loss_bc.item()),
                    "num_colloc": len(colloc_pts)
                })

            if rar_step < rar_iterations:
                # Candidate points pool
                candidates = torch.rand(1000, 2, dtype=torch.float32)
                res_cand = torch.abs(self.pde_residual(candidates)).squeeze().detach().numpy()
                
                # Pick top points with maximum residual
                top_indices = np.argsort(res_cand)[-points_per_rar:]
                new_points = candidates[top_indices]

                rar_added_points.append(new_points.detach().numpy().tolist())
                colloc_pts = torch.cat([colloc_pts, new_points.detach()], dim=0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Evaluate final predictions on a dense grid
        gx = np.linspace(0, 1, 50)
        gt = np.linspace(0, 1, 50)
        GX, GT = np.meshgrid(gx, gt)
        grid_pts = torch.tensor(np.column_stack([GX.ravel(), GT.ravel()]), dtype=torch.float32)

        with torch.no_grad():
            u_pred = self.model(grid_pts).numpy().reshape(50, 50)

        u_exact = np.sin(np.pi * GX) * np.cos(2.0 * GT)
        l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

        return {
            "loss_history": loss_history,
            "rar_added_points": rar_added_points,
            "collocation_points": colloc_pts.detach().numpy().tolist(),
            "u_pred": u_pred.tolist(),
            "u_exact": u_exact.tolist(),
            "l2_error": float(l2_error),
            "train_time_ms": elapsed_ms
        }
