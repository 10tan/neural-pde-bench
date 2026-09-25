import React from "react";
import { Sliders, RefreshCw, ShieldAlert } from "lucide-react";

interface ParameterInspectorProps {
  nx: number;
  setNx: (val: number) => void;
  nt: number;
  setNt: (val: number) => void;
  rarIter: number;
  setRarIter: (val: number) => void;
  noiseLevel: number;
  setNoiseLevel: (val: number) => void;
  onExecute: () => void;
  isExecuting: boolean;
}

export const ParameterInspector: React.FC<ParameterInspectorProps> = ({
  nx,
  setNx,
  nt,
  setNt,
  rarIter,
  setRarIter,
  noiseLevel,
  setNoiseLevel,
  onExecute,
  isExecuting,
}) => {
  return (
    <aside className="card" style={{ width: "300px", padding: "16px", height: "calc(100vh - 100px)", overflowY: "auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
        <Sliders size={18} color="var(--accent-blue)" />
        <h2 style={{ fontSize: "0.9rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-primary)" }}>
          Parameter Inspector
        </h2>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div>
          <label style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: "600", marginBottom: "4px" }}>
            <span>Spatial Grid Points (N_x):</span>
            <span className="font-mono">{nx}</span>
          </label>
          <input
            type="range"
            min="20"
            max="200"
            step="10"
            value={nx}
            onChange={(e) => setNx(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent-blue)" }}
          />
        </div>

        <div>
          <label style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: "600", marginBottom: "4px" }}>
            <span>Time Steps (N_t):</span>
            <span className="font-mono">{nt}</span>
          </label>
          <input
            type="range"
            min="10"
            max="100"
            step="10"
            value={nt}
            onChange={(e) => setNt(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent-blue)" }}
          />
        </div>

        <div>
          <label style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: "600", marginBottom: "4px" }}>
            <span>RAR Iterations:</span>
            <span className="font-mono">{rarIter}</span>
          </label>
          <input
            type="range"
            min="1"
            max="5"
            step="1"
            value={rarIter}
            onChange={(e) => setRarIter(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent-blue)" }}
          />
        </div>

        <div>
          <label style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: "600", marginBottom: "4px" }}>
            <span>Observation Noise Std:</span>
            <span className="font-mono">{(noiseLevel * 100).toFixed(1)}%</span>
          </label>
          <input
            type="range"
            min="0.0"
            max="0.1"
            step="0.01"
            value={noiseLevel}
            onChange={(e) => setNoiseLevel(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent-blue)" }}
          />
        </div>

        <button className="btn-primary" style={{ width: "100%", justifyContent: "center", marginTop: "12px" }} onClick={onExecute} disabled={isExecuting}>
          <RefreshCw size={16} className={isExecuting ? "animate-spin" : ""} />
          {isExecuting ? "Executing Simulation..." : "Re-run Selected Model"}
        </button>

        <div style={{ marginTop: "20px", padding: "12px", backgroundColor: "var(--accent-blue-light)", borderRadius: "6px", fontSize: "0.775rem", color: "var(--accent-blue)" }}>
          <div style={{ fontWeight: "600", marginBottom: "4px", display: "flex", alignItems: "center", gap: "4px" }}>
            <ShieldAlert size={14} />
            <span>Verification Specs</span>
          </div>
          All solvers enforcement L2 accuracy relative to analytical ground truth MMS functions. Failure thresholds active.
        </div>
      </div>
    </aside>
  );
};
