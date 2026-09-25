import React from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

interface BenchmarkChartProps {
  data: Array<{ model: string; l2_error: number; inference_time_ms: number; train_time_sec: number }>;
}

export const BenchmarkChart: React.FC<BenchmarkChartProps> = ({ data }) => {
  const chartData = data.map((d) => ({
    model: d.model,
    "L2 Error (%)": Number((d.l2_error * 100).toFixed(3)),
    "Inference Latency (ms)": Number(d.inference_time_ms.toFixed(2)),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div className="card" style={{ padding: "16px" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: "700", marginBottom: "4px", color: "var(--accent-blue)" }}>
          Paradigm Speed vs Accuracy Benchmark Studio
        </h3>
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "16px" }}>
          Direct accuracy (L2 Relative Error) and execution latency comparison across FDM, PINN+RAR, FNO, DeepONet, and GNO.
        </p>

        <div style={{ width: "100%", height: "300px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EC" />
              <XAxis dataKey="model" tick={{ fontSize: 11, fontWeight: 600 }} />
              <YAxis yAxisId="left" orientation="left" stroke="#1F4E79" label={{ value: "L2 Error (%)", angle: -90, position: "insideLeft", fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" stroke="#D97706" label={{ value: "Latency (ms)", angle: 90, position: "insideRight", fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar yAxisId="left" dataKey="L2 Error (%)" fill="#1F4E79" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="right" dataKey="Inference Latency (ms)" fill="#D97706" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ padding: "16px" }}>
        <h4 style={{ fontSize: "0.85rem", fontWeight: "700", marginBottom: "8px", textTransform: "uppercase" }}>
          Detailed Model Telemetry Matrix
        </h4>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.825rem" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid var(--border-color)", textAlign: "left" }}>
              <th style={{ padding: "8px" }}>Model Architecture</th>
              <th style={{ padding: "8px" }}>L2 Relative Error</th>
              <th style={{ padding: "8px" }}>Inference Latency</th>
              <th style={{ padding: "8px" }}>Pretrain / Setup Cost</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid var(--border-color)" }}>
                <td style={{ padding: "8px", fontWeight: "600", color: "var(--accent-blue)" }}>{row.model}</td>
                <td style={{ padding: "8px" }} className="font-mono">{(row.l2_error * 100).toFixed(3)}%</td>
                <td style={{ padding: "8px" }} className="font-mono">{row.inference_time_ms.toFixed(2)} ms</td>
                <td style={{ padding: "8px" }} className="font-mono">{row.train_time_sec > 0 ? `${row.train_time_sec.toFixed(2)}s` : "Zero (Direct Solve)"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
