# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2D Airfoil Shape Optimization using OpenFOAM-14 (`airFoil2D` tutorial)."""

from pathlib import Path
import numpy as np
from src.optimization.optimization import (
    CST_BOUNDS,
    MAX_SIMULATIONS_PER_EVAL,
    MAX_TOTAL_ATTEMPTS_PER_EVAL,
    Parameters,
    check_geometry,
    evaluate_batch,
    funct,
)

# Parameter bounds for the 6 Kulfan CST airfoil shape weights:
#   x[0] = w_l0 in [-0.35, -0.05]: Lower surface leading-edge nose weight (negative)
#   x[1] = w_l1 in [-0.30,  0.20]: Lower surface mid-chord weight (positive creates high-lift lower cove)
#   x[2] = w_l2 in [-0.30,  0.25]: Lower surface aft weight (positive creates rear-loaded reflex/cove)
#   x[3] = w_u0 in [ 0.08,  0.50]: Upper surface leading-edge suction weight (positive)
#   x[4] = w_u1 in [ 0.05,  0.55]: Upper surface mid-chord camber arch weight (positive)
#   x[5] = w_u2 in [ 0.05,  0.50]: Upper surface aft pressure-recovery weight (positive)
BOUNDS = list(CST_BOUNDS)

# Symmetric NACA 0012 baseline airfoil in 6-parameter Kulfan CST representation
# (matches the baseline geometry of OpenFOAM-14 tutorials/incompressibleFluid/airFoil2D, Cl/Cd = 12.59)
BASELINE_X = np.array(
    [-0.171373, -0.139418, -0.149078, 0.171373, 0.139418, 0.149078],
    dtype=float,
)


def optimize_airfoil(
    parameters: Parameters,
    max_successful_simulations: int = MAX_SIMULATIONS_PER_EVAL,
    max_total_simulations: int = MAX_TOTAL_ATTEMPTS_PER_EVAL,
) -> tuple[np.ndarray, float]:
  """Searches for the 6 Kulfan CST parameters that maximize Cl/Cd in OpenFOAM-14.

  Flow conditions (OpenFOAM-14 incompressibleFluid/airFoil2D):
    - Freestream velocity: U = (25.75, 3.62, 0.0) m/s (|U| = 26.003 m/s, AoA = 8.00 deg)
    - Kinematic viscosity: nu = 1e-5 m^2/s (Re = 2.6e6 based on 1 m chord)
    - Turbulence model: Spalart-Allmaras steady-state RANS (SIMPLE)
    - Structural constraints (enforced by `check_geometry(x)`):
        * All 6 weights within `BOUNDS`
        * Minimum interior thickness `y_u(x) - y_l(x) >= 0.005` on `x/c in [0.01, 0.99]`
        * Maximum chordwise thickness `0.08 <= max_thickness <= 0.24` (8% to 24% chord)

  Available evaluation primitives:
    - `check_geometry(x) -> (is_valid, metrics_dict)`: Fast analytical pre-check
      (microseconds; does not consume OpenFOAM budget).
    - `funct(x, parameters) -> float`: Runs 1 OpenFOAM-14 CFD simulation and
      returns `-Cl/Cd` (negative for minimization; `0.0` on invalid geometry/mesh).
    - `evaluate_batch(candidates, parameters, max_workers=4) -> list[float]`:
      Evaluates a list/array of CST vectors in parallel using OpenFOAM-14 subprocesses.
    - Budget per candidate program: up to `max_successful_simulations = 16`
      converged OpenFOAM-14 simulations (`max_total_simulations = 100` attempts).
  """
# EVOLVE-BLOCK-START
  # Generation 0 baseline: evaluate the canonical OpenFOAM-14 NACA 0012 airfoil.
  # Evolve this algorithm to explore the 6D Kulfan CST space within BOUNDS
  # (using analytical aerodynamic surrogates, parallel batch sampling via
  # `evaluate_batch(..., max_workers=4)`, Gaussian Process / RBF / quadratic
  # response surfaces, or trust-region refinement) across the 16-simulation budget.
  _ = (max_successful_simulations, max_total_simulations, check_geometry, evaluate_batch)
  best_x = BASELINE_X.copy()
  best_score = funct(best_x, parameters)
  return best_x, best_score
# EVOLVE-BLOCK-END


def evaluate(parameters: Parameters) -> tuple[np.ndarray, float]:
  """Executes the airfoil optimization routine and returns (best_x, best_score)."""
  return optimize_airfoil(
      parameters,
      max_successful_simulations=MAX_SIMULATIONS_PER_EVAL,
      max_total_simulations=MAX_TOTAL_ATTEMPTS_PER_EVAL,
  )


if __name__ == "__main__":
  run_parameters = Parameters(
      run_name="local_test",
      cases_folder=Path("cases"),
      is_debug=True,
      csv_path=Path("results.csv"),
  )
  x_opt, score_opt = evaluate(run_parameters)
  print(f"Best CST parameters: {x_opt}")
  print(f"OpenFOAM Score (-Cl/Cd): {score_opt:.4f}")
