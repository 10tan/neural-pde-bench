import numpy as np
import time
from typing import Dict, Any, List

class BayesianOptimalDesign:
    """
    Bayesian Optimal Experimental Design (BOED) loop for active sensor placement.
    Maximizes Expected Information Gain (EIG) / minimizes variance of parameter posterior field.
    """

    @staticmethod
    def recommend_sensor_locations(
        existing_sensors: List[float] = [0.1, 0.9], 
        num_new_sensors: int = 3, 
        nx: int = 100
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        x_grid = np.linspace(0, 1, nx)

        # Prior variance field over domain (highest in regions far from current sensors)
        variance_field = np.ones(nx) * 0.25
        for s_pos in existing_sensors:
            dist = np.abs(x_grid - s_pos)
            variance_field *= (1.0 - np.exp(-dist / 0.15))

        new_recommended = []
        eig_history = []

        curr_var = variance_field.copy()
        for k in range(num_new_sensors):
            # Pick location with highest remaining variance (max Expected Information Gain)
            best_idx = int(np.argmax(curr_var))
            best_x = float(x_grid[best_idx])
            best_eig = float(curr_var[best_idx])

            new_recommended.append(best_x)
            eig_history.append(best_eig)

            # Update variance field after placing sensor
            dist = np.abs(x_grid - best_x)
            curr_var *= (1.0 - np.exp(-dist / 0.15))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "x_grid": x_grid.tolist(),
            "prior_variance": variance_field.tolist(),
            "posterior_variance": curr_var.tolist(),
            "existing_sensors": existing_sensors,
            "recommended_sensors": new_recommended,
            "expected_information_gain": eig_history,
            "elapsed_ms": elapsed_ms
        }
