import React from "react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

interface SensorPlacementProps {
  boedData: {
    x_grid: number[];
    prior_variance: number[];
    posterior_variance: number[];
    existing_sensors: number[];
    recommended_sensors: number[];
    expected_information_gain: number[];
  };
}

export const SensorPlacementMap: React.FC<SensorPlacementProps> = ({ boedData }) => {
  const chartData = boedData.x_grid.map((x, idx) => ({
    x: Number(x.toFixed(2)),
    "Prior Uncertainty": Number(boedData.prior_variance[idx]?.toFixed(4)),
    "Posterior Uncertainty": Number(boedData.posterior_variance[idx]?.toFixed(4)),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div className="card" style={{ padding: "16px" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: "700", marginBottom: "4px", color: "var(--accent-blue)" }}>
          BOED Active Sensor Placement Optimization
        </h3>
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "16px" }}>
          Maximizes Expected Information Gain (EIG) to minimize parameter uncertainty field.
        </p>

        <div style={{ display: "flex", gap: "12px", marginBottom: "16px" }}>
          <div style={{ flex: 1, padding: "12px", backgroundColor: "var(--bg-main)", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>Existing Sensors (x)</div>
            <div style={{ fontSize: "1rem", fontWeight: "700", color: "var(--text-primary)" }} className="font-mono">
              {boedData.existing_sensors.join(", ")}
            </div>
          </div>
          <div style={{ flex: 1, padding: "12px", backgroundColor: "var(--accent-blue-light)", borderRadius: "6px", border: "1px solid rgba(31,78,121,0.2)" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--accent-blue)", fontWeight: "600" }}>BOED Recommended Placements</div>
            <div style={{ fontSize: "1rem", fontWeight: "700", color: "var(--accent-blue)" }} className="font-mono">
              {boedData.recommended_sensors.join(", ")}
            </div>
          </div>
        </div>

        <div style={{ width: "100%", height: "260px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EC" />
              <XAxis dataKey="x" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Area type="monotone" dataKey="Prior Uncertainty" stroke="#D97706" fill="#FEF3C7" strokeWidth={2} />
              <Area type="monotone" dataKey="Posterior Uncertainty" stroke="#2E7D32" fill="#E8F5E9" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
