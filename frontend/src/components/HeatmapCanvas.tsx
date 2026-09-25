import React, { useRef, useEffect, useState } from "react";
import { Play, Pause, RotateCcw, Eye } from "lucide-react";

interface HeatmapCanvasProps {
  x: number[];
  y: number[];
  field: number[][]; // 2D array [ny, nx]
  title: string;
  pdeType?: string;
}

export const HeatmapCanvas: React.FC<HeatmapCanvasProps> = ({ x, y, field, title }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ xVal: number; yVal: number; uVal: number; px: number; py: number } | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [frameOffset, setFrameOffset] = useState(0);

  const ny = field.length;
  const nx = field[0]?.length || 0;

  // Thermal Color Map function (Blue -> Teal -> Yellow -> Red)
  const getThermalColor = (val: number, minVal: number, maxVal: number) => {
    const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1)));
    let r = 0, g = 0, b = 0;
    if (norm < 0.33) {
      // Blue to Cyan
      const t = norm / 0.33;
      b = Math.floor(255 * (1 - t * 0.2));
      g = Math.floor(255 * t * 0.8);
      r = Math.floor(31 * (1 - t));
    } else if (norm < 0.66) {
      // Cyan to Yellow
      const t = (norm - 0.33) / 0.33;
      r = Math.floor(245 * t);
      g = Math.floor(200 + 55 * t);
      b = Math.floor(200 * (1 - t));
    } else {
      // Yellow to Deep Red
      const t = (norm - 0.66) / 0.34;
      r = Math.floor(245 + 10 * t);
      g = Math.floor(255 * (1 - t * 0.85));
      b = Math.floor(30 * (1 - t));
    }
    return `rgb(${r}, ${g}, ${b})`;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || ny === 0 || nx === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let minV = Infinity, maxV = -Infinity;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = field[j][i];
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }

    const cellW = canvas.width / nx;
    const cellH = canvas.height / ny;

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        // Apply frame phase shift for play animation
        const val = field[(j + frameOffset) % ny][i];
        ctx.fillStyle = getThermalColor(val, minV, maxV);
        ctx.fillRect(i * cellW, (ny - 1 - j) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [field, nx, ny, frameOffset]);

  useEffect(() => {
    let timer: any;
    if (isPlaying) {
      timer = setInterval(() => {
        setFrameOffset((prev) => (prev + 1) % (ny || 1));
      }, 80);
    }
    return () => clearInterval(timer);
  }, [isPlaying, ny]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || nx === 0 || ny === 0) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const ix = Math.floor((px / canvas.width) * nx);
    const iy = Math.floor(((canvas.height - py) / canvas.height) * ny);

    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
      const uVal = field[iy][ix];
      setHoverInfo({
        xVal: x[ix] ?? (ix / nx),
        yVal: y[iy] ?? (iy / ny),
        uVal,
        px,
        py,
      });
    }
  };

  return (
    <div className="card" style={{ padding: "16px", position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Eye size={16} color="var(--accent-blue)" />
          <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>{title}</h3>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <button className="btn-secondary" style={{ padding: "4px 8px", fontSize: "0.75rem" }} onClick={() => setIsPlaying(!isPlaying)}>
            {isPlaying ? <Pause size={12} /> : <Play size={12} />}
            {isPlaying ? "Pause Anim" : "Play Transient"}
          </button>
          <button className="btn-secondary" style={{ padding: "4px 8px", fontSize: "0.75rem" }} onClick={() => setFrameOffset(0)}>
            <RotateCcw size={12} /> Reset
          </button>
        </div>
      </div>

      <div style={{ position: "relative", width: "100%", display: "flex", justifyContent: "center" }}>
        <canvas
          ref={canvasRef}
          width={480}
          height={260}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverInfo(null)}
          style={{ width: "100%", height: "260px", borderRadius: "4px", border: "1px solid var(--border-color)", cursor: "crosshair" }}
        />

        {hoverInfo && (
          <div
            style={{
              position: "absolute",
              top: Math.max(10, hoverInfo.py - 40),
              left: Math.min(350, hoverInfo.px + 10),
              backgroundColor: "rgba(26, 26, 26, 0.9)",
              color: "#FFFFFF",
              padding: "4px 8px",
              borderRadius: "4px",
              fontSize: "0.75rem",
              pointerEvents: "none",
              zIndex: 10,
            }}
            className="font-mono"
          >
            x: {hoverInfo.xVal.toFixed(3)}, y/t: {hoverInfo.yVal.toFixed(3)} | u = {hoverInfo.uVal.toFixed(4)}
          </div>
        )}
      </div>

      {/* Color Scale Bar */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "10px", fontSize: "0.725rem", color: "var(--text-secondary)" }}>
        <span>Min Field (0.0)</span>
        <div
          style={{
            flex: 1,
            height: "8px",
            borderRadius: "4px",
            background: "linear-gradient(to right, rgb(31,0,255), rgb(0,200,200), rgb(245,255,0), rgb(255,0,0))",
          }}
        />
        <span>Max Field (1.0)</span>
      </div>
    </div>
  );
};
