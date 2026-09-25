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
      title: "1D Heat Equation MMS",
      pde: "u_t - α * u_xx = f(x,t)",
      exact: "u(x,t) = sin(πx) * cos(ωt)",
      source: "f(x,t) = -ω sin(πx) sin(ωt) + α π² sin(πx) cos(ωt)",
    },
    {
      title: "Fourier Neural Operator (FNO)",
      pde: "(K(a) v)(x) = F⁻¹ ( R_θ · F(v) )(x)",
      exact: "Truncated modes k_max with complex parameter weights R_θ",
      source: "Maps initial input function a(x) to solution space u(x)",
    },
    {
      title: "DeepONet (Branch-Trunk Operator)",
      pde: "G(u)(y) = Σ (b_k(u) * t_k(y)) + b_0",
      exact: "b_k: Branch network over m sensor points; t_k: Trunk network over query (x,t)",
      source: "Nonlinear continuous operator mapping continuous input spaces",
    },
    {
      title: "XPINN Interface Penalty Loss",
      pde: "L_total = L_pde1 + L_pde2 + 10.0 * L_val_int + 1.0 * L_flux_int",
      exact: "L_val_int = ||u₁(x_int) - u₂(x_int)||², L_flux_int = ||∇u₁(x_int) - ∇u₂(x_int)||²",
      source: "Ensures C¹ continuity across domain decomposition boundary x = 0.5",
    },
  ];

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        right: 0,
        width: "420px",
        height: "100vh",
        backgroundColor: "var(--bg-card)",
        boxShadow: "-4px 0 20px rgba(0,0,0,0.1)",
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
          <BookOpen size={18} color="var(--accent-blue)" />
          <h2 style={{ fontSize: "1rem", fontWeight: "700", color: "var(--accent-blue)" }}>
            Mathematical Formulations
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
            <div className="font-mono" style={{ fontSize: "0.775rem", backgroundColor: "#FFFFFF", padding: "8px", borderRadius: "4px", border: "1px solid var(--border-color)", marginBottom: "6px" }}>
              {item.pde}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "2px" }}>
              <strong>Exact Formulation:</strong> {item.exact}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
              <strong>Context / Source:</strong> {item.source}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
