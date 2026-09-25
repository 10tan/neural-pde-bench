# Verification & Validation (V&V) Protocol

## Verification Targets
All neural solvers in `neural-pde-bench` are continuously verified against analytical ground truths using Method of Manufactured Solutions (MMS).

### 1D Heat Equation MMS
- **Governing PDE:** $u_t - \alpha u_{xx} = f(x,t)$
- **Exact Solution:** $u(x,t) = \sin(\pi x) \cos(\omega t)$
- **Source Term:** $f(x,t) = -\omega \sin(\pi x) \sin(\omega t) + \alpha \pi^2 \sin(\pi x) \cos(\omega t)$

### 2D Poisson Equation MMS
- **Governing PDE:** $-\Delta u = f(x,y)$
- **Exact Solution:** $u(x,y) = \sin(\pi x) \sin(\pi y)$
- **Source Term:** $f(x,y) = 2\pi^2 \sin(\pi x) \sin(\pi y)$

## Pass/Fail Tolerance
- **FDM Classical Solver:** Relative $L_2 \text{ error} < 1.0\%$
- **PINN + RAR:** Relative $L_2 \text{ error} < 1.0\%$ after 200 epochs
- **FNO / DeepONet:** Relative $L_2 \text{ error} < 0.5\%$
