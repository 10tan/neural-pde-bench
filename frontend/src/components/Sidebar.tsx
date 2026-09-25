import React from "react";
import { Layers, Network, Activity, Cpu, Compass, Sliders } from "lucide-react";

interface SidebarProps {
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ selectedModel, onSelectModel }) => {
  const categories = [
    {
      title: "Classical Numerical Baseline",
      icon: Layers,
      items: [
        { id: "fdm_heat", name: "1D FDM Crank-Nicolson" },
        { id: "fdm_poisson", name: "2D Poisson 5-Point Stencil" },
      ],
    },
    {
      title: "Physics-Informed Networks",
      icon: Network,
      items: [
        { id: "pinn_rar", name: "PINN + RAR Adaptive Sampling" },
        { id: "xpinn", name: "XPINN Domain Decomposition" },
      ],
    },
    {
      title: "Bayesian PINN & UQ",
      icon: Activity,
      items: [
        { id: "bayes_vi", name: "Variational Inference (MC Dropout)" },
        { id: "bayes_hmc", name: "Hamiltonian Monte Carlo (HMC)" },
      ],
    },
    {
      title: "Neural Operators",
      icon: Cpu,
      items: [
        { id: "fno_1d", name: "Fourier Neural Operator (FNO 1D)" },
        { id: "deeponet", name: "DeepONet (Branch-Trunk)" },
        { id: "gno_mesh", name: "Geometry Graph Operator (GNO)" },
      ],
    },
    {
      title: "Inverse & Generative UQ",
      icon: Compass,
      items: [
        { id: "diffusion_inverse", name: "Diffusion Posterior Sampler" },
        { id: "multifidelity", name: "Multi-Fidelity DeepONet" },
      ],
    },
    {
      title: "Experimental Design & Pretraining",
      icon: Sliders,
      items: [
        { id: "boed_sensors", name: "BOED Active Sensor Placement" },
        { id: "pretrain_backbone", name: "Pretrain Multi-PDE Backbone" },
      ],
    },
  ];

  return (
    <aside className="card" style={{ width: "280px", padding: "16px", height: "calc(100vh - 100px)", overflowY: "auto" }}>
      <h2 style={{ fontSize: "0.85rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-secondary)", marginBottom: "16px" }}>
        Model & Paradigm Selector
      </h2>

      {categories.map((cat, idx) => {
        const Icon = cat.icon;
        return (
          <div key={idx} style={{ marginBottom: "20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.85rem", fontWeight: "600", color: "var(--accent-blue)", marginBottom: "8px" }}>
              <Icon size={16} />
              <span>{cat.title}</span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              {cat.items.map((item) => {
                const isSelected = selectedModel === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onSelectModel(item.id)}
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      padding: "8px 12px",
                      borderRadius: "4px",
                      fontSize: "0.825rem",
                      border: "none",
                      backgroundColor: isSelected ? "var(--accent-blue-light)" : "transparent",
                      color: isSelected ? "var(--accent-blue)" : "var(--text-primary)",
                      fontWeight: isSelected ? "600" : "400",
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                    }}
                  >
                    {item.name}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </aside>
  );
};
