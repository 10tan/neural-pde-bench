import pytest
from fastapi.testclient import TestClient

from npb.data.pde_data import SyntheticPDEDataGenerator
from npb.solvers.fem_fdm import ClassicalPDESolvers
from npb.pinn.pinn_rar import PINNTrainerWithRAR
from npb.xpinn.xpinn import XPINNTrainer
from npb.bayes_pinn.bayes_pinn import BayesianPINNSolver
from npb.fno.fno import FNOSolver
from npb.deeponet.deeponet import DeepONetSolver
from npb.gno.gno import GNOSolver
from npb.diffusion.diffusion_inverse import DiffusionPosteriorSampler
from npb.multifidelity.multifidelity import MultiFidelitySolver
from npb.boed.boed import BayesianOptimalDesign
from npb.pretrain.pretrain import PretrainBenchmark
from npb.api.main import app

client = TestClient(app)

def test_data_generators():
    heat = SyntheticPDEDataGenerator.generate_heat_1d_mms(nx=20, nt=10)
    assert len(heat["x"]) == 20
    assert heat["u_exact"].shape == (10, 20)

    poisson = SyntheticPDEDataGenerator.generate_poisson_2d(nx=10, ny=10)
    assert poisson["u_exact"].shape == (10, 10)

def test_classical_solvers():
    fdm = ClassicalPDESolvers.solve_heat_1d_fdm(nx=20, nt=20)
    assert fdm["l2_error"] < 0.1

def test_pinn_rar():
    trainer = PINNTrainerWithRAR()
    res = trainer.train_with_rar(n_initial_collocation=30, rar_iterations=1, epochs_per_iter=10)
    assert "l2_error" in res
    assert len(res["loss_history"]) > 0

def test_xpinn():
    trainer = XPINNTrainer()
    res = trainer.train_xpinn(epochs=10, n_sub_pts=20)
    assert "u_pred" in res

def test_bayes_pinn():
    vi_res = BayesianPINNSolver.run_variational_inference(num_mc_samples=5, nx=10, nt=10)
    assert len(vi_res["mean_field"]) == 10

def test_fno():
    fno_res = FNOSolver.solve_fno_benchmark(nx=20)
    assert len(fno_res["u_pred"]) == 20

def test_deeponet():
    deep_res = DeepONetSolver.solve_deeponet_benchmark(nx=10, nt=10)
    assert len(deep_res["u_pred"]) == 10

def test_gno():
    gno_res = GNOSolver.solve_gno_on_domain("l_shape")
    assert "u_pred" in gno_res

def test_diffusion():
    diff_res = DiffusionPosteriorSampler.sample_inverse_posterior(num_samples=10, nx=20)
    assert len(diff_res["mean_field"]) == 20

def test_api_endpoints():
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_bench = client.get("/api/benchmark/suite")
    assert r_bench.status_code == 200
    assert len(r_bench.json()["benchmark_suite"]) == 5

    r_lat = client.get("/api/inference/latency")
    assert r_lat.status_code == 200
    assert "per_sample_latency_us" in r_lat.json()
