import React, { useState } from "react";
import { BookOpen, Code, Cpu, ShieldCheck, Copy, Check, Network, Activity, Sliders, Terminal } from "lucide-react";

export const DocumentationTab: React.FC = () => {
  const [copiedSection, setCopiedSection] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<"all" | "pinn" | "operator" | "bayes" | "boed">("all");

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(label);
    setTimeout(() => setCopiedSection(null), 2000);
  };

  const codeSnippets = {
    pinnLoss: `# PyTorch Automatic Differentiation Loss
def pde_residual(self, xt: torch.Tensor) -> torch.Tensor:
    xt.requires_grad_(True)
    u = self.model(xt)
    grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    res = u_t - self.alpha * u_xx - f(xt[:,0:1], xt[:,1:2])
    return res`,
    fnoConv: `# Fourier Spectral Convolution Layer
class SpectralConv1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x) # Physical to Fourier domain
        modes = min(self.modes1, x_ft.size(-1))
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        out_ft[:, :, :modes] = (x_ft[:, :, :modes].unsqueeze(2) * self.weights1[:, :, :modes].unsqueeze(0)).sum(dim=1)
        return torch.fft.irfft(out_ft, n=x.size(-1)) # Back to Physical domain`,
    deeponet: `# DeepONet Inner Product
def forward(self, u_sensors: torch.Tensor, y_coords: torch.Tensor) -> torch.Tensor:
    b_out = self.branch(u_sensors) # (batch, p_basis)
    t_out = self.trunk(y_coords)   # (batch, num_points, p_basis)
    return torch.einsum("bp,bnp->bn", b_out, t_out) + self.bias`,
    backendStart: "PYTHONPATH=src python3 -m uvicorn npb.api.main:app --reload --port 8001",
    frontendStart: "cd frontend && npm install && npm run dev",
    pytestRun: "PYTHONPATH=src python3 -m pytest tests/",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Top Banner Card */}
      <div className="card" style={{ padding: "20px", background: "linear-gradient(135deg, #FFFFFF 0%, #F4F8FA 100%)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
              <BookOpen size={24} color="var(--accent-blue)" />
              <h2 style={{ fontSize: "1.2rem", fontWeight: "700", color: "var(--accent-blue)" }}>
                Neural PDE Bench — Comprehensive Docs & Specs Studio
              </h2>
            </div>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", maxWidth: "800px" }}>
              In-depth theoretical derivations, exact loss residual formulations, operator learning architectures (FNO, DeepONet, GNO), Bayesian UQ pipelines, and V&V compliance standards.
            </p>
          </div>
          <span className="badge badge-green" style={{ fontSize: "0.8rem", padding: "4px 10px" }}>
            <ShieldCheck size={14} /> V&V Compliance Verified
          </span>
        </div>

        {/* Category Filter Pills */}
        <div style={{ display: "flex", gap: "8px", marginTop: "16px" }}>
          <button className={`btn-secondary ${activeCategory === "all" ? "btn-primary" : ""}`} onClick={() => setActiveCategory("all")} style={{ fontSize: "0.775rem", padding: "4px 10px" }}>
            All Specifications
          </button>
          <button className={`btn-secondary ${activeCategory === "pinn" ? "btn-primary" : ""}`} onClick={() => setActiveCategory("pinn")} style={{ fontSize: "0.775rem", padding: "4px 10px" }}>
            <Network size={12} /> PINN & RAR
          </button>
          <button className={`btn-secondary ${activeCategory === "operator" ? "btn-primary" : ""}`} onClick={() => setActiveCategory("operator")} style={{ fontSize: "0.775rem", padding: "4px 10px" }}>
            <Cpu size={12} /> FNO / DeepONet / GNO
          </button>
          <button className={`btn-secondary ${activeCategory === "bayes" ? "btn-primary" : ""}`} onClick={() => setActiveCategory("bayes")} style={{ fontSize: "0.775rem", padding: "4px 10px" }}>
            <Activity size={12} /> Bayesian UQ & Diffusion
          </button>
          <button className={`btn-secondary ${activeCategory === "boed" ? "btn-primary" : ""}`} onClick={() => setActiveCategory("boed")} style={{ fontSize: "0.775rem", padding: "4px 10px" }}>
            <Sliders size={12} /> BOED & Pretrain
          </button>
        </div>
      </div>

      {/* Main Grid Section */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: "16px" }}>
        {/* Left Column: Mathematical Formulations & Architecture Specifications */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {(activeCategory === "all" || activeCategory === "pinn") && (
            <div className="card" style={{ padding: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                <Network size={18} color="var(--accent-blue)" />
                <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>
                  1. Physics-Informed Neural Networks (PINN + RAR)
                </h3>
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "10px" }}>
                PINNs parameterize solutions $u_\theta(x,t)$ using neural networks constrained by partial differential operators via automatic differentiation.
              </p>

              <div style={{ padding: "10px", backgroundColor: "var(--bg-main)", borderRadius: "6px", border: "1px solid var(--border-color)", marginBottom: "10px" }}>
                <div style={{ fontSize: "0.75rem", fontWeight: "600", color: "var(--text-primary)", marginBottom: "4px" }}>
                  Governing PDE & Residual Loss:
                </div>
                <div className="font-mono" style={{ fontSize: "0.75rem", color: "var(--accent-blue)" }}>
                  L_total(θ) = L_PDE(θ) + 10 * L_IC(θ) + 10 * L_BC(θ)
                </div>
                <div className="font-mono" style={{ fontSize: "0.725rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                  L_PDE(θ) = 1/N * Σ | ∂u_θ/∂t - α ∂²u_θ/∂x² - f(x,t) |²
                </div>
              </div>

              <div style={{ fontSize: "0.775rem", color: "var(--text-secondary)" }}>
                <strong>Residual-Based Adaptive Refinement (RAR):</strong> At each refinement iteration, candidate points $(x_k, t_k)$ are evaluated across domain candidate space. The top $K$ points with highest residual $|R(x,t)|$ are dynamically added to the collocation set, focusing capacity on steep gradient fronts.
              </div>
            </div>
          )}

          {(activeCategory === "all" || activeCategory === "operator") && (
            <div className="card" style={{ padding: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                <Cpu size={18} color="var(--accent-blue)" />
                <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>
                  2. Operator Learning Suite (FNO, DeepONet, GNO)
                </h3>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.8rem" }}>
                <div style={{ padding: "10px", backgroundColor: "var(--bg-main)", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
                  <div style={{ fontWeight: "700", color: "var(--accent-blue)", marginBottom: "4px" }}>
                    Fourier Neural Operator (FNO 1D)
                  </div>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.775rem", marginBottom: "6px" }}>
                    Fourier Neural Operators map continuous functions to functions. The kernel integral is computed via Fast Fourier Transform (FFT):
                  </p>
                  <code className="font-mono" style={{ fontSize: "0.75rem", color: "#1A1A1A", display: "block" }}>
                    (K(a) v)(x) = F⁻¹ ( R_θ · F(v) )(x)
                  </code>
                </div>

                <div style={{ padding: "10px", backgroundColor: "var(--bg-main)", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
                  <div style={{ fontWeight: "700", color: "var(--accent-blue)", marginBottom: "4px" }}>
                    Deep Operator Network (DeepONet)
                  </div>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.775rem", marginBottom: "6px" }}>
                    Dual Branch (sensor evaluations) and Trunk (query coordinates) inner product:
                  </p>
                  <code className="font-mono" style={{ fontSize: "0.75rem", color: "#1A1A1A", display: "block" }}>
                    G(a)(x,t) = Σ b_k(a(x_1), ..., a(x_m)) * t_k(x,t) + b_0
                  </code>
                </div>

                <div style={{ padding: "10px", backgroundColor: "var(--bg-main)", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
                  <div style={{ fontWeight: "700", color: "var(--accent-blue)", marginBottom: "4px" }}>
                    Geometry Graph Neural Operator (GNO)
                  </div>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.775rem" }}>
                    Spatial Graph Message Passing over arbitrary domain geometries (L-shaped domains, circular hole cutouts).
                  </p>
                </div>
              </div>
            </div>
          )}

          {(activeCategory === "all" || activeCategory === "bayes") && (
            <div className="card" style={{ padding: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                <Activity size={18} color="var(--accent-blue)" />
                <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>
                  3. Bayesian PINN UQ & Score-Based Diffusion Samplers
                </h3>
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "10px" }}>
                Quantifies predictive mean $\mu(x)$ and 95% ($\pm 2\sigma$) uncertainty bounds for inverse parameter identification $E(x)$.
              </p>
              <ul style={{ fontSize: "0.775rem", color: "var(--text-secondary)", paddingLeft: "18px", lineHeight: "1.6" }}>
                <li><strong>Variational Inference (MC Dropout):</strong> Performs stochastic Monte Carlo forward passes to quantify predictive variance.</li>
                <li><strong>Hamiltonian Monte Carlo (HMC):</strong> Markov Chain Monte Carlo parameter posterior sampling.</li>
                <li><strong>Diffusion Inverse Sampler:</strong> Score-based generative model $dE_t = g(t) dW_t + g(t)^2 \nabla_E \log p_t(E|y) dt$ for rapid inverse UQ sampling.</li>
              </ul>
            </div>
          )}
        </div>

        {/* Right Column: Interactive PyTorch Snippets & Telemetry Matrix */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* PyTorch Code Implementation Viewer */}
          <div className="card" style={{ padding: "16px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
              <Terminal size={18} color="var(--accent-blue)" />
              <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>
                Core PyTorch Implementation Snippets
              </h3>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", fontWeight: "600", marginBottom: "4px" }}>
                  <span>PINN Automatic Differentiation Loss</span>
                  <button className="btn-secondary" style={{ padding: "2px 6px", fontSize: "0.7rem" }} onClick={() => copyToClipboard(codeSnippets.pinnLoss, "pinnLoss")}>
                    {copiedSection === "pinnLoss" ? <Check size={12} color="green" /> : <Copy size={12} />} Copy
                  </button>
                </div>
                <pre className="font-mono" style={{ fontSize: "0.725rem", backgroundColor: "#1A1A1A", color: "#61DAFB", padding: "8px", borderRadius: "4px", overflowX: "auto", maxHeight: "140px" }}>
                  {codeSnippets.pinnLoss}
                </pre>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", fontWeight: "600", marginBottom: "4px" }}>
                  <span>FNO Spectral Convolution Layer</span>
                  <button className="btn-secondary" style={{ padding: "2px 6px", fontSize: "0.7rem" }} onClick={() => copyToClipboard(codeSnippets.fnoConv, "fnoConv")}>
                    {copiedSection === "fnoConv" ? <Check size={12} color="green" /> : <Copy size={12} />} Copy
                  </button>
                </div>
                <pre className="font-mono" style={{ fontSize: "0.725rem", backgroundColor: "#1A1A1A", color: "#A7F3D0", padding: "8px", borderRadius: "4px", overflowX: "auto", maxHeight: "140px" }}>
                  {codeSnippets.fnoConv}
                </pre>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", fontWeight: "600", marginBottom: "4px" }}>
                  <span>DeepONet Branch-Trunk Inner Product</span>
                  <button className="btn-secondary" style={{ padding: "2px 6px", fontSize: "0.7rem" }} onClick={() => copyToClipboard(codeSnippets.deeponet, "deeponet")}>
                    {copiedSection === "deeponet" ? <Check size={12} color="green" /> : <Copy size={12} />} Copy
                  </button>
                </div>
                <pre className="font-mono" style={{ fontSize: "0.725rem", backgroundColor: "#1A1A1A", color: "#FDE047", padding: "8px", borderRadius: "4px", overflowX: "auto", maxHeight: "100px" }}>
                  {codeSnippets.deeponet}
                </pre>
              </div>
            </div>
          </div>

          {/* Quickstart Terminal Commands */}
          <div className="card" style={{ padding: "16px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
              <Code size={18} color="var(--accent-blue)" />
              <h3 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--accent-blue)" }}>
                CLI Commands
              </h3>
            </div>
            <div style={{ fontSize: "0.775rem", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "8px" }}>
              <div>
                <strong>Backend FastAPI:</strong>
                <code className="font-mono" style={{ display: "block", backgroundColor: "var(--bg-main)", padding: "4px 8px", borderRadius: "4px", marginTop: "2px" }}>
                  PYTHONPATH=src python3 -m uvicorn npb.api.main:app --port 8001
                </code>
              </div>
              <div>
                <strong>Pytest Suite:</strong>
                <code className="font-mono" style={{ display: "block", backgroundColor: "var(--bg-main)", padding: "4px 8px", borderRadius: "4px", marginTop: "2px" }}>
                  PYTHONPATH=src python3 -m pytest tests/
                </code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
