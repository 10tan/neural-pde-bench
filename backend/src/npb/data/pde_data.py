import numpy as np
import torch
from typing import Dict, Tuple, Any

class SyntheticPDEDataGenerator:
    """
    Generates synthetic datasets using Method of Manufactured Solutions (MMS),
    analytical functions, and numerical solver ground truths for SciML benchmarks.
    """

    @staticmethod
    def generate_heat_1d_mms(
        nx: int = 100, 
        nt: int = 50, 
        alpha: float = 0.01, 
        omega: float = 2.0, 
        noise_level: float = 0.0
    ) -> Dict[str, Any]:
        """
        1D Transient Heat Equation with MMS:
        u_t - alpha * u_xx = f(x,t)
        Exact solution: u(x,t) = sin(pi * x) * cos(omega * t)
        """
        x = np.linspace(0, 1, nx)
        t = np.linspace(0, 1, nt)
        X, T = np.meshgrid(x, t)

        u_exact = np.sin(np.pi * X) * np.cos(omega * T)
        
        # Source term f(x,t) derived from exact solution
        u_t = -omega * np.sin(np.pi * X) * np.sin(omega * T)
        u_xx = -(np.pi**2) * np.sin(np.pi * X) * np.cos(omega * T)
        f = u_t - alpha * u_xx

        # Add Gaussian noise if specified
        if noise_level > 0.0:
            noise = np.random.normal(0, noise_level * np.std(u_exact), u_exact.shape)
            u_observed = u_exact + noise
        else:
            u_observed = u_exact.copy()

        return {
            "x": x,
            "t": t,
            "X": X,
            "T": T,
            "u_exact": u_exact,
            "u_observed": u_observed,
            "f": f,
            "alpha": alpha,
            "omega": omega,
            "pde": "heat_1d_mms"
        }

    @staticmethod
    def generate_poisson_2d(
        nx: int = 50, 
        ny: int = 50, 
        noise_level: float = 0.0
    ) -> Dict[str, Any]:
        """
        2D Poisson Equation: - (u_xx + u_yy) = f(x,y)
        Exact solution: u(x,y) = sin(pi * x) * sin(pi * y)
        Source term: f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)
        """
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        f = 2.0 * (np.pi**2) * np.sin(np.pi * X) * np.sin(np.pi * Y)

        if noise_level > 0.0:
            noise = np.random.normal(0, noise_level * np.std(u_exact), u_exact.shape)
            u_observed = u_exact + noise
        else:
            u_observed = u_exact.copy()

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "u_exact": u_exact,
            "u_observed": u_observed,
            "f": f,
            "pde": "poisson_2d"
        }

    @staticmethod
    def generate_inverse_field_data(
        nx: int = 100, 
        parameter_type: str = "sinusoidal", 
        noise_std: float = 0.02
    ) -> Dict[str, Any]:
        """
        Generates synthetic 1D spatially-varying parameter field E(x) and corresponding displacement u(x).
        PDE: - d/dx ( E(x) * du/dx ) = f(x)
        """
        x = np.linspace(0, 1, nx)
        dx = x[1] - x[0]

        if parameter_type == "sinusoidal":
            E_true = 1.0 + 0.5 * np.sin(2.0 * np.pi * x)
        elif parameter_type == "step":
            E_true = np.where(x < 0.5, 1.0, 2.5)
        else:
            E_true = 1.0 + 0.3 * np.cos(3.0 * np.pi * x)

        f = np.ones_like(x) # constant load

        # Solve numerically for ground truth u(x) using finite differences
        # Boundary conditions: u(0) = 0, u(1) = 0
        N = nx - 2
        A = np.zeros((N, N))
        b = f[1:-1] * (dx**2)

        for i in range(N):
            idx = i + 1
            E_plus = 0.5 * (E_true[idx] + E_true[idx + 1])
            E_minus = 0.5 * (E_true[idx] + E_true[idx - 1])

            A[i, i] = E_plus + E_minus
            if i > 0:
                A[i, i - 1] = -E_minus
            if i < N - 1:
                A[i, i + 1] = -E_plus

        u_internal = np.linalg.solve(A, b)
        u_true = np.zeros(nx)
        u_true[1:-1] = u_internal

        # Noisy measurements
        u_obs = u_true + np.random.normal(0, noise_std, nx)

        return {
            "x": x,
            "E_true": E_true,
            "u_true": u_true,
            "u_obs": u_obs,
            "f": f,
            "noise_std": noise_std
        }

    @staticmethod
    def generate_multifidelity_dataset(
        num_samples: int = 100, 
        nx_hi: int = 100, 
        nx_lo: int = 20
    ) -> Dict[str, Any]:
        """
        Generates paired low-fidelity (coarse grid) and high-fidelity (fine grid) data
        for Multi-Fidelity DeepONet testing.
        u_hi(x) = sin(pi * x * k) + 0.1 * cos(3 * pi * x * k)
        u_lo(x) = u_hi(x) * 0.8 + 0.1 * sin(2 * pi * x)
        """
        x_hi = np.linspace(0, 1, nx_hi)
        x_lo = np.linspace(0, 1, nx_lo)

        k_values = np.random.uniform(1.0, 3.0, num_samples)
        
        inputs_branch = [] # branch inputs (evaluated on coarse grid)
        u_hi_list = []
        u_lo_list = []

        for k in k_values:
            inp = np.sin(np.pi * x_lo * k)
            u_h = np.sin(np.pi * x_hi * k) + 0.1 * np.cos(3.0 * np.pi * x_hi * k)
            u_l = u_h * 0.8 + 0.1 * np.sin(2.0 * np.pi * x_hi)

            inputs_branch.append(inp)
            u_hi_list.append(u_h)
            u_lo_list.append(u_l)

        return {
            "x_hi": x_hi,
            "x_lo": x_lo,
            "branch_inputs": np.array(inputs_branch),
            "u_hi": np.array(u_hi_list),
            "u_lo": np.array(u_lo_list),
            "k_values": k_values
        }

    @staticmethod
    def generate_domain_mesh(domain_type: str = "rectangle", num_nodes: int = 200) -> Dict[str, Any]:
        """
        Generates 2D node coordinates and adjacency graph edges for GNO testing.
        """
        if domain_type == "l_shape":
            # Points in L-shape domain: [0,1]x[0,1] minus [0.5,1]x[0.5,1]
            pts = []
            while len(pts) < num_nodes:
                p = np.random.uniform(0, 1, 2)
                if not (p[0] >= 0.5 and p[1] >= 0.5):
                    pts.append(p)
            nodes = np.array(pts)
        elif domain_type == "circle_hole":
            # [0,1]x[0,1] with hole at center (0.5, 0.5) r=0.25
            pts = []
            while len(pts) < num_nodes:
                p = np.random.uniform(0, 1, 2)
                if np.linalg.norm(p - 0.5) >= 0.25:
                    pts.append(p)
            nodes = np.array(pts)
        else:
            # Standard rectangle
            nodes = np.random.uniform(0, 1, (num_nodes, 2))

        # Build k-NN adjacency graph (k=6)
        from scipy.spatial import cKDTree
        tree = cKDTree(nodes)
        _, indices = tree.query(nodes, k=7) # includes self
        edges = []
        for i in range(len(nodes)):
            for j in indices[i][1:]:
                edges.append([i, j])

        # Exact solution u(x,y) on domain
        u_nodes = np.sin(np.pi * nodes[:, 0]) * np.cos(np.pi * nodes[:, 1])

        return {
            "nodes": nodes,
            "edges": np.array(edges),
            "u_nodes": u_nodes,
            "domain_type": domain_type
        }
