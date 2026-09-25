import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any

class GraphConvLayer(nn.Module):
    """Spatial Graph Convolution Layer for Graph Neural Operator (GNO)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neighbor = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (num_nodes, in_dim)
        # edge_index: (num_edges, 2)
        h_self = self.fc_self(x)
        h_neigh = self.fc_neighbor(x)

        # Aggregate neighbor features
        src, dst = edge_index[:, 0], edge_index[:, 1]
        agg = torch.zeros_like(h_self)
        agg.index_add_(0, dst, h_neigh[src])

        return torch.tanh(h_self + agg / 6.0)

class GeometryGraphNeuralOperator(nn.Module):
    """Graph Neural Operator (GNO) capable of predicting PDE solutions on arbitrary geometry meshes."""
    def __init__(self, in_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.gconv1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gconv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, node_coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.encoder(node_coords))
        h = self.gconv1(h, edge_index)
        h = self.gconv2(h, edge_index)
        out = self.decoder(h)
        return out

class GNOSolver:
    """GNO Solver wrapper for geometry-aware PDE benchmarking."""

    @staticmethod
    def solve_gno_on_domain(domain_type: str = "l_shape") -> Dict[str, Any]:
        start_time = time.perf_counter()

        from npb.data.pde_data import SyntheticPDEDataGenerator
        mesh_data = SyntheticPDEDataGenerator.generate_domain_mesh(domain_type=domain_type, num_nodes=200)

        nodes = torch.tensor(mesh_data["nodes"], dtype=torch.float32)
        edges = torch.tensor(mesh_data["edges"], dtype=torch.long)
        u_exact = mesh_data["u_nodes"]

        model = GeometryGraphNeuralOperator(in_dim=2, hidden_dim=32)
        with torch.no_grad():
            u_pred = model(nodes, edges).squeeze().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        l2_err = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

        return {
            "domain_type": domain_type,
            "nodes": mesh_data["nodes"].tolist(),
            "edges": mesh_data["edges"].tolist(),
            "u_pred": u_pred.tolist(),
            "u_exact": u_exact.tolist(),
            "l2_error": float(l2_err),
            "solve_time_ms": elapsed_ms,
            "architecture": "GNO (Geometry-aware Graph Neural Operator)"
        }
