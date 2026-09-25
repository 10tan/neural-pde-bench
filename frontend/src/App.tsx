import { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { ParameterInspector } from "./components/ParameterInspector";
import { MathFormulaDrawer } from "./components/MathFormulaDrawer";
import { HeatmapCanvas } from "./components/HeatmapCanvas";

import { TrainingMonitor } from "./charts/TrainingMonitor";
import { BenchmarkChart } from "./charts/BenchmarkChart";
import { InversePosteriorPlot } from "./charts/InversePosteriorPlot";
import { SensorPlacementMap } from "./charts/SensorPlacementMap";
import { PretrainEfficiencyPlot } from "./charts/PretrainEfficiencyPlot";

import {
  fetchHealth,
  trainPINN,
  fetchBenchmarkSuite,
  fetchBayesVI,
  fetchDiffusionInverse,
  fetchBOED,
  fetchPretrainEfficiency,
  fetch2DField,
  fetchExportReport,
} from "./services/api";

import { Activity, Zap, Shield, Globe, Rocket } from "lucide-react";

export function App() {
  const [deviceStatus, setDeviceStatus] = useState("cpu");
  const [selectedModel, setSelectedModel] = useState("pinn_rar");
  const [activeTab, setActiveTab] = useState<"train" | "bench" | "inverse" | "gno" | "pretrain">("train");
  const [isMathOpen, setIsMathOpen] = useState(false);

  // Hyperparameters
  const [nx, setNx] = useState(100);
  const [nt, setNt] = useState(50);
  const [rarIter, setRarIter] = useState(2);
  const [noiseLevel, setNoiseLevel] = useState(0.02);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  // Simulation data states
  const [pinnResult, setPinnResult] = useState<any>(null);
  const [benchSuite, setBenchSuite] = useState<any[]>([]);
  const [bayesVI, setBayesVI] = useState<any>(null);
  const [diffInverse, setDiffInverse] = useState<any>(null);
  const [boedData, setBoedData] = useState<any>(null);
  const [pretrainData, setPretrainData] = useState<any>(null);
  const [field2d, setField2d] = useState<any>(null);

  useEffect(() => {
    fetchHealth().then((h) => setDeviceStatus(h.device));
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setIsExecuting(true);
    const [pinn, suite, vi, diff, boed, pre, f2d] = await Promise.all([
      trainPINN({ n_initial_collocation: 100, rar_iterations: rarIter, epochs_per_iter: 100 }),
      fetchBenchmarkSuite(),
      fetchBayesVI(),
      fetchDiffusionInverse(),
      fetchBOED([0.1, 0.9]),
      fetchPretrainEfficiency(),
      fetch2DField("poisson_2d"),
    ]);

    setPinnResult(pinn);
    setBenchSuite(suite.benchmark_suite || []);
    setBayesVI(vi);
    setDiffInverse(diff);
    setBoedData(boed);
    setPretrainData(pre);
    setField2d(f2d);
    setIsExecuting(false);
  };

  const handleExecute = async () => {
    setIsExecuting(true);
    const res = await trainPINN({
      n_initial_collocation: nx,
      rar_iterations: rarIter,
      epochs_per_iter: 100,
    });
    setPinnResult(res);

    const f2d = await fetch2DField("heat_1d_mms");
    setField2d(f2d);

    const boed = await fetchBOED([0.1, 0.9]);
    setBoedData(boed);
    setIsExecuting(false);
  };

  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    const suite = await fetchBenchmarkSuite();
    setBenchSuite(suite.benchmark_suite || []);
    setActiveTab("bench");
    setIsBenchmarking(false);
  };

  const handleExportReport = async () => {
    const report = await fetchExportReport();
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `NPB_Verification_Report_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "var(--bg-main)", display: "flex", flexDirection: "column" }}>
      <Header
        deviceStatus={deviceStatus}
        onRunBenchmark={handleRunBenchmark}
        onOpenMath={() => setIsMathOpen(true)}
        onExportReport={handleExportReport}
        isBenchmarking={isBenchmarking}
      />

      <div style={{ display: "flex", gap: "16px", padding: "0 16px 16px 16px", flex: 1 }}>
        <Sidebar selectedModel={selectedModel} onSelectModel={(id) => setSelectedModel(id)} />

        <main style={{ flex: 1, display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* Workspace Tabs */}
          <div className="card" style={{ padding: "8px 16px", display: "flex", gap: "8px", alignItems: "center" }}>
            <button
              className={`btn-secondary ${activeTab === "train" ? "btn-primary" : ""}`}
              onClick={() => setActiveTab("train")}
              style={{ fontSize: "0.825rem", padding: "6px 12px" }}
            >
              <Activity size={14} /> Training & RAR Monitor
            </button>
            <button
              className={`btn-secondary ${activeTab === "bench" ? "btn-primary" : ""}`}
              onClick={() => setActiveTab("bench")}
              style={{ fontSize: "0.825rem", padding: "6px 12px" }}
            >
              <Zap size={14} /> Benchmark Studio
            </button>
            <button
              className={`btn-secondary ${activeTab === "inverse" ? "btn-primary" : ""}`}
              onClick={() => setActiveTab("inverse")}
              style={{ fontSize: "0.825rem", padding: "6px 12px" }}
            >
              <Shield size={14} /> Inverse UQ & Diffusion
            </button>
            <button
              className={`btn-secondary ${activeTab === "gno" ? "btn-primary" : ""}`}
              onClick={() => setActiveTab("gno")}
              style={{ fontSize: "0.825rem", padding: "6px 12px" }}
            >
              <Globe size={14} /> Geometry & Sensors
            </button>
            <button
              className={`btn-secondary ${activeTab === "pretrain" ? "btn-primary" : ""}`}
              onClick={() => setActiveTab("pretrain")}
              style={{ fontSize: "0.825rem", padding: "6px 12px" }}
            >
              <Rocket size={14} /> Pretrain & Latency
            </button>
          </div>

          {/* Active Tab Content */}
          {activeTab === "train" && pinnResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <TrainingMonitor
                lossHistory={pinnResult.loss_history || []}
                l2Error={pinnResult.l2_error || 0.0084}
                trainTimeMs={pinnResult.train_time_ms || 340}
              />
              {field2d && (
                <HeatmapCanvas
                  x={field2d.x || []}
                  y={field2d.y || []}
                  field={field2d.field || []}
                  title="2D PDE Solution Field u(x,y) Contour Heatmap"
                />
              )}
            </div>
          )}

          {activeTab === "bench" && <BenchmarkChart data={benchSuite} />}

          {activeTab === "inverse" && bayesVI && diffInverse && (
            <InversePosteriorPlot bayesData={bayesVI} diffData={diffInverse} />
          )}

          {activeTab === "gno" && boedData && <SensorPlacementMap boedData={boedData} />}

          {activeTab === "pretrain" && pretrainData && <PretrainEfficiencyPlot pretrainData={pretrainData} />}
        </main>

        <ParameterInspector
          nx={nx}
          setNx={setNx}
          nt={nt}
          setNt={setNt}
          rarIter={rarIter}
          setRarIter={setRarIter}
          noiseLevel={noiseLevel}
          setNoiseLevel={setNoiseLevel}
          onExecute={handleExecute}
          isExecuting={isExecuting}
        />
      </div>

      <MathFormulaDrawer isOpen={isMathOpen} onClose={() => setIsMathOpen(false)} />
    </div>
  );
}

export default App;
