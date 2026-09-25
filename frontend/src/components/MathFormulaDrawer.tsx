import React from "react";
import { BookOpen, X, CheckCircle } from "lucide-react";

interface MathFormulaDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MathFormulaDrawer: React.FC<MathFormulaDrawerProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const formulas = [
    {
      title: "1D Transient Heat Equation MMS",
      pde: "u_t - α * u_xx = f(x,t)",
      exact: "u(x,t) = sin(πx) * cos(ωt)",
      source: "f(x,t) = -ω sin(πx) sin(ωt) + α π² sin(πx) cos(ωt)",
      description: "Automatic differentiation calculates loss: L_PDE = 1/N Σ |u_t - α u_xx - f(x,t)|²",
    },
    {
      title: "2D Poisson Equation MMS",
      pde: "- (u_xx + u_yy) = f(x,y)",
      exact: "u(x,y) = sin(πx) * sin(πy)",
      source: "f(x,y) = 2π² sin(πx) sin(πy)",
      description: "5-point finite difference stencil: (4u_{i,j} - u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1})/h² = f_{i,j}",
    },
    {
      title: "Fourier Neural Operator (FNO 1D)",
      pde: "(K(a) v)(x) = F⁻¹ ( R_θ · F(v) )(x)",
      exact: "Truncated modes k ≤ k_max with complex parameter matrix R_θ",
      source: "rfft(x) → elementwise spectral multiplication (R_θ) → irfft(x)",
      description: "Maps continuous input function spaces to PDE solution operators with grid resolution invariance.",
    },
    {
      title: "Deep Operator Network (DeepONet)",
      pde: "G(a)(x,t) = Σ (b_k(a(x_1), ..., a(x_m)) * t_k(x,t)) + b_0",
      exact: "Branch b_k: MLP over m sensor points; Trunk t_k: MLP over query coords (x,t)",
      source: "Inner product between branch and trunk network latent space (dimension p = 32)",
      description: "Learns non-linear continuous operators mapping function evaluations at sensor locations to solution fields.",
    },
    {
      title: "XPINN Subdomain Interface Penalty Loss",
      pde: "L_total = L_pde1 + L_pde2 + 10.0 * L_val_int + 1.0 * L_flux_int",
      exact: "L_val_int = ||u₁(x_int, t) - u₂(x_int, t)||², L_flux_int = ||∇u₁(x_int, t) - ∇u₂(x_int, t)||²",
      source: "Subdomain 1: x ∈ [0, 0.5]; Subdomain 2: x ∈ [0.5, 1.0]",
      description: "Ensures value C⁰ and flux C¹ continuity across domain decomposition interface boundary x = 0.5.",
    },
    {
      title: "Score-Based Diffusion Inverse Sampler",
      pde: "dE_t = g(t) dW_t + g(t)² ∇_E log p_t(E | y) dt",
      exact: "Reverse SDE sampling process for inverse parameter identification E(x)",
      source: "Generates 95% (±2σ) posterior uncertainty bounds around E_true(x)",
      description: "Benchmarked directly against Hamiltonian Monte Carlo (HMC) sampling chains for inverse problem UQ.",
    },
  ];

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        right: 0,
        width: "480px",
        height: "100vh",
        backgroundColor: "var(--bg-card)",
        boxShadow: "-4px 0 24px rgba(0,0,0,0.12)",
        zIndex: 100,
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
        borderLeft: "1px solid var(--border-color)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <BookOpen size={20} color="var(--accent-blue)" />
          <h2 style={{ fontSize: "1.05rem", fontWeight: "700", color: "var(--accent-blue)" }}>
            Mathematical Specs & Governing Equations
          </h2>
        </div>
        <button className="btn-secondary" style={{ padding: "4px" }} onClick={onClose}>
          <X size={16} />
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {formulas.map((item, idx) => (
          <div key={idx} style={{ padding: "14px", backgroundColor: "var(--bg-main)", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
            <div style={{ fontWeight: "700", fontSize: "0.85rem", color: "var(--accent-blue)", marginBottom: "6px", display: "flex", alignItems: "center", gap: "6px" }}>
              <CheckCircle size={14} color="var(--accent-green)" />
              <span>{item.title}</span>
            </div>

            <div className="font-mono" style={{ fontSize: "0.775rem", backgroundColor: "#FFFFFF", padding: "8px", borderRadius: "4px", border: "1px solid var(--border-color)", marginBottom: "8px", color: "#1A1A1A" }}>
              {item.pde}
            </div>

            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
              <strong>Exact Formulation:</strong> {item.exact}
            </div>

            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
              <strong>Source / Stencil:</strong> {item.source}
            </div>

            <div style={{ fontSize: "0.75rem", color: "var(--text-primary)", fontStyle: "italic", marginTop: "4px" }}>
              {item.description}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
