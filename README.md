# Neural PDE Bench (Project 5)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Physics-Informed and Operator-Learning Framework for Forward/Inverse PDEs with Uncertainty Quantification.

## Features
- **Synthetic Data Generation:** Method of Manufactured Solutions (MMS), FDM/FEM analytical solutions, and multi-fidelity datasets.
- **PINN + RAR:** Residual-Based Adaptive Refinement for collocation point sampling.
- **XPINN Domain Decomposition:** Multi-subdomain decomposition with interface value & flux matching penalties.
- **Bayesian PINN UQ:** Variational Inference (MC Dropout) and Hamiltonian Monte Carlo (HMC) sampling.
- **Neural Operators:** Fourier Neural Operator (FNO 1D), DeepONet (Branch/Trunk), and Geometry Graph Neural Operator (GNO).
- **Diffusion Inverse Sampler:** Score-based generative model for inverse parameter estimation $E(x)$.
- **Optimal Experimental Design:** BOED active sensor placement optimization maximizing Expected Information Gain.
- **Pretrain-then-Finetune:** Multi-PDE foundation backbone with transfer learning benchmark.
- **Minimalist Light UI:** React + TypeScript + Recharts dashboard matching blueprint aesthetics (`#FAFBFC` background, `#1F4E79` accent).

## Quick Start

### Backend (Python FastAPI)
```bash
cd backend
python3 -m pip install -e .
PYTHONPATH=src python3 -m pytest tests/
uvicorn npb.api.main:app --reload --port 8000
```

### Frontend (Vite React TS)
```bash
cd frontend
npm install
npm run dev
```

### Docker
```bash
docker-compose -f infra/docker-compose.yml up --build
```
