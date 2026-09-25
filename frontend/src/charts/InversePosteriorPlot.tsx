import React from "react";
import { ResponsiveContainer, AreaChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

interface InversePosteriorProps {
  bayesData: { x: number[]; mean_field: number[]; upper_bound: number[]; lower_bound: number[]; method: string };
  diffData: { x: number[]; mean_field: number[]; upper_bound: number[]; lower_bound: number[]; E_true?: number[]; method: string };
}

export const InversePosteriorPlot: React.FC<InversePosteriorProps> = ({ bayesData, diffData }) => {
  const chartData = bayesData.x.map((val, idx) => ({
    x: Number(val.toFixed(2)),
    "Bayes PINN Mean": Number(bayesData.mean_field[idx]?.toFixed(4)),
    "Bayes Upper 95%": Number(bayesData.upper_bound[idx]?.toFixed(4)),
    "Bayes Lower 95%": Number(bayesData.lower_bound[idx]?.toFixed(4)),
    "Diffusion Mean": Number(diffData.mean_field[idx]?.toFixed(4)),
    "Diffusion Upper 95%": Number(diffData.upper_bound[idx]?.toFixed(4)),
    "Diffusion Lower 95%": Number(diffData.lower_bound[idx]?.toFixed(4)),
    "Ground Truth E(x)": diffData.E_true ? Number(diffData.E_true[idx]?.toFixed(4)) : undefined,
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div className="card" style={{ padding: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
          <div>
            <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>
              Inverse UQ Posterior Distribution Comparison
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              Side-by-side uncertainty quantification ($\pm 2\sigma$ confidence interval): Bayesian PINN (HMC) vs Score-Based Diffusion.
            </p>
          </div>
          <span className="badge badge-green">Validated against Analytical Field E(x)</span>
        </div>

        <div style={{ width: "100%", height: "320px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EC" />
              <XAxis dataKey="x" tick={{ fontSize: 11 }} label={{ value: "Spatial Domain x", position: "insideBottom", offset: -5, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              
              <Area type="monotone" dataKey="Bayes Upper 95%" stroke="none" fill="#EBF3FA" />
              <Area type="monotone" dataKey="Diffusion Upper 95%" stroke="none" fill="#FEF3C7" />

              <Line type="monotone" dataKey="Ground Truth E(x)" stroke="#1A1A1A" strokeWidth={2.5} strokeDasharray="4 4" dot={false} />
              <Line type="monotone" dataKey="Bayes PINN Mean" stroke="#1F4E79" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Diffusion Mean" stroke="#D97706" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
