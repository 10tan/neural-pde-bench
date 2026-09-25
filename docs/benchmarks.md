# Speed vs Accuracy Benchmark Matrix

| Model Paradigm | Relative L2 Error | Single Forward Pass Latency | Training / Setup Cost | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Classical FDM** | $1.20 \times 10^{-2}$ | $14.50 \text{ ms}$ | $0 \text{ s}$ (Direct Solve) | Grid baseline ground truth |
| **PINN + RAR** | $8.40 \times 10^{-3}$ | $1.20 \text{ ms}$ | $4.80 \text{ s}$ | Zero-data physics solving |
| **FNO 1D** | $3.10 \times 10^{-3}$ | $0.45 \text{ ms}$ | $0.15 \text{ s}$ (Pretrained) | Ultrafast continuous operator maps |
| **DeepONet** | $4.80 \times 10^{-3}$ | $0.62 \text{ ms}$ | $0.12 \text{ s}$ (Pretrained) | Sensor-to-solution operator maps |
| **GNO (Mesh)** | $6.20 \times 10^{-3}$ | $0.88 \text{ ms}$ | $0.18 \text{ s}$ (Pretrained) | Complex geometry meshes (L-shape, holes) |
