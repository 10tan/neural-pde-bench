import React from "react";
import { Cpu, Activity, Play, BookOpen, Download } from "lucide-react";

interface HeaderProps {
  deviceStatus: string;
  onRunBenchmark: () => void;
  onOpenMath: () => void;
  onExportReport: () => void;
  isBenchmarking: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  deviceStatus,
  onRunBenchmark,
  onOpenMath,
  onExportReport,
  isBenchmarking,
}) => {
  return (
    <header className="card" style={{ padding: "12px 24px", marginBottom: "16px", borderRadius: "0 0 6px 6px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <h1 style={{ fontSize: "1.25rem", fontWeight: "700", color: "var(--accent-blue)" }}>
                Neural PDE Bench
              </h1>
              <span className="badge badge-blue">v0.2.0 Extended</span>
            </div>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              Physics-Informed & Operator-Learning Framework for Forward/Inverse PDEs with UQ
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }} className="badge badge-green">
            <Activity size={14} />
            <span>API Online</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "6px" }} className="badge badge-blue">
            <Cpu size={14} />
            <span style={{ textTransform: "uppercase" }}>{deviceStatus}</span>
          </div>

          <button className="btn-secondary" style={{ fontSize: "0.8rem" }} onClick={onOpenMath}>
            <BookOpen size={14} /> Math Specs
          </button>

          <button className="btn-secondary" style={{ fontSize: "0.8rem" }} onClick={onExportReport}>
            <Download size={14} /> Audit Report
          </button>

          <button className="btn-primary" onClick={onRunBenchmark} disabled={isBenchmarking}>
            <Play size={16} />
            {isBenchmarking ? "Benchmarking..." : "Run Benchmark Suite"}
          </button>
        </div>
      </div>
    </header>
  );
};
