import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

interface PretrainEfficiencyProps {
  pretrainData: {
    sample_sizes: number[];
    l2_errors_pretrained: number[];
    l2_errors_scratch: number[];
    speedup_factor: number;
    pde_families: string[];
  };
}

export const PretrainEfficiencyPlot: React.FC<PretrainEfficiencyProps> = ({ pretrainData }) => {
  const chartData = pretrainData.sample_sizes.map((n, idx) => ({
    N: n,
    "Pretrained Backbone": Number((pretrainData.l2_errors_pretrained[idx] * 100).toFixed(2)),
    "Trained from Scratch": Number((pretrainData.l2_errors_scratch[idx] * 100).toFixed(2)),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div className="card" style={{ padding: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
          <div>
            <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>
              Pretrain-then-Finetune Data Efficiency Benchmark
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              Evaluates transfer learning data efficiency across sample sizes N = [5, 10, 20, 50, 100].
            </p>
          </div>
          <span className="badge badge-green">{pretrainData.speedup_factor}x Data Efficiency Speedup</span>
        </div>

        <div style={{ width: "100%", height: "260px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EC" />
              <XAxis dataKey="N" tick={{ fontSize: 11 }} label={{ value: "Training Samples (N)", position: "insideBottom", offset: -5, fontSize: 11 }} />
              <YAxis scale="log" domain={["auto", "auto"]} tick={{ fontSize: 11 }} label={{ value: "L2 Relative Error (%)", angle: -90, position: "insideLeft", fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Line type="monotone" dataKey="Pretrained Backbone" stroke="#1F4E79" strokeWidth={2.5} />
              <Line type="monotone" dataKey="Trained from Scratch" stroke="#D97706" strokeWidth={2} strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
