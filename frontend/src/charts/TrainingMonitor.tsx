import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

interface TrainingMonitorProps {
  lossHistory: Array<{ step: number; total_loss: number; loss_pde: number; loss_ic: number; loss_bc: number }>;
  l2Error: number;
  trainTimeMs: number;
}

export const TrainingMonitor: React.FC<TrainingMonitorProps> = ({ lossHistory, l2Error, trainTimeMs }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>Final L2 Error</div>
          <div style={{ fontSize: "1.4rem", fontWeight: "700", color: "var(--accent-blue)" }} className="font-mono">
            {(l2Error * 100).toFixed(3)}%
          </div>
        </div>
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>Training Duration</div>
          <div style={{ fontSize: "1.4rem", fontWeight: "700", color: "var(--accent-green)" }} className="font-mono">
            {trainTimeMs.toFixed(1)} ms
          </div>
        </div>
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>Final Total Loss</div>
          <div style={{ fontSize: "1.4rem", fontWeight: "700", color: "var(--accent-amber)" }} className="font-mono">
            {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].total_loss.toExponential(3) : "N/A"}
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: "16px" }}>
        <h3 style={{ fontSize: "0.9rem", fontWeight: "700", marginBottom: "12px", color: "var(--accent-blue)" }}>
          PINN Physics Loss Convergence (PDE, IC, BC)
        </h3>
        <div style={{ width: "100%", height: "260px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lossHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EC" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} />
              <YAxis scale="log" domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value: any) => Number(value).toExponential(3)} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Line type="monotone" dataKey="total_loss" name="Total Loss" stroke="#1F4E79" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="loss_pde" name="PDE Residual" stroke="#D97706" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="loss_ic" name="IC Loss" stroke="#2E7D32" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="loss_bc" name="BC Loss" stroke="#6D28D9" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
