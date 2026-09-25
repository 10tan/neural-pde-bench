import numpy as np
import time
from typing import Dict, Any

class ClassicalPDESolvers:
    """
    High-performance classical numerical PDE solvers (FDM/FEM)
    used for benchmark ground truths, accuracy evaluations, and speed comparisons.
    """

    @staticmethod
    def solve_heat_1d_fdm(
        nx: int = 100, 
        nt: int = 100, 
        alpha: float = 0.01, 
        t_max: float = 1.0
    ) -> Dict[str, Any]:
        """
        Solves 1D Heat Equation u_t = alpha * u_xx using Crank-Nicolson FDM scheme.
        """
        start_time = time.perf_counter()

        x = np.linspace(0, 1, nx)
        t = np.linspace(0, t_max, nt)
        dx = x[1] - x[0]
        dt = t[1] - t[0]

        r = alpha * dt / (2.0 * dx**2)

        # Initial condition: u(x,0) = sin(pi * x)
        u = np.sin(np.pi * x)
        u_history = [u.copy()]

        # Tridiagonal system matrix for Crank-Nicolson
        N = nx - 2
        A = np.zeros((N, N))
        B = np.zeros((N, N))

        for i in range(N):
            A[i, i] = 1.0 + 2.0 * r
            B[i, i] = 1.0 - 2.0 * r
            if i > 0:
                A[i, i - 1] = -r
                B[i, i - 1] = r
            if i < N - 1:
                A[i, i + 1] = -r
                B[i, i + 1] = r

        u_curr = u[1:-1]
        for n in range(1, nt):
            rhs = B @ u_curr
            u_next = np.linalg.solve(A, rhs)
            u_curr = u_next

            u_full = np.zeros(nx)
            u_full[1:-1] = u_curr
            u_history.append(u_full)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Exact analytical solution: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)
        X, T = np.meshgrid(x, t)
        u_exact = np.exp(-alpha * (np.pi**2) * T) * np.sin(np.pi * X)
        u_fem_grid = np.array(u_history)

        l2_error = np.linalg.norm(u_fem_grid - u_exact) / np.linalg.norm(u_exact)

        return {
            "x": x,
            "t": t,
            "u_solution": u_fem_grid,
            "u_exact": u_exact,
            "l2_error": float(l2_error),
            "solve_time_ms": elapsed_ms,
            "method": "FDM (Crank-Nicolson)"
        }

    @staticmethod
    def solve_poisson_2d_fdm(nx: int = 50, ny: int = 50) -> Dict[str, Any]:
        """
        Solves 2D Poisson Equation - (u_xx + u_yy) = 2 * pi^2 * sin(pi*x) * sin(pi*y)
        using 5-point stencil finite difference solver.
        """
        start_time = time.perf_counter()

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        X, Y = np.meshgrid(x, y)
        f = 2.0 * (np.pi**2) * np.sin(np.pi * X) * np.sin(np.pi * Y)

        N_internal = (nx - 2) * (ny - 2)
        A = np.zeros((N_internal, N_internal))
        b = np.zeros(N_internal)

        def get_idx(i, j):
            return (i - 1) * (ny - 2) + (j - 1)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                idx = get_idx(i, j)
                b[idx] = f[j, i] # Note meshgrid indexing [y, x]

                A[idx, idx] = 2.0 / (dx**2) + 2.0 / (dy**2)

                if i > 1:
                    A[idx, get_idx(i - 1, j)] = -1.0 / (dx**2)
                if i < nx - 2:
                    A[idx, get_idx(i + 1, j)] = -1.0 / (dx**2)
                if j > 1:
                    A[idx, get_idx(i, j - 1)] = -1.0 / (dy**2)
                if j < ny - 2:
                    A[idx, get_idx(i, j + 1)] = -1.0 / (dy**2)

        u_vec = np.linalg.solve(A, b)

        u_sol = np.zeros((ny, nx))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_sol[j, i] = u_vec[get_idx(i, j)]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        l2_error = np.linalg.norm(u_sol - u_exact) / np.linalg.norm(u_exact)

        return {
            "x": x,
            "y": y,
            "u_solution": u_sol,
            "u_exact": u_exact,
            "l2_error": float(l2_error),
            "solve_time_ms": elapsed_ms,
            "method": "5-Point Stencil FDM"
        }
