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
"""Evaluator for FSI 99.9% Tail VaR / CVaR Monte Carlo Optimization."""

import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from alpha_evolve.models import (
    AlphaEvolveEvaluationInsight,
    AlphaEvolveEvaluationInsights,
    AlphaEvolveEvaluationScore,
    AlphaEvolveEvaluationScores,
    AlphaEvolveProgramEvaluation,
)

SAMPLE_DIR = Path(__file__).absolute().parent
BINARY_PATH = Path("/var/tmp/var_mc_sim")
METRIC_NAME = "effective_speedup_score"

# Gold-standard reference values (computed via 2.1M-path Girsanov IS + Control Variates)
GOLD_REFERENCE = {
    0: {
        "var_99": 440.02,
        "var_999": 599.21,
        "cvar_999": 632.48,
        "baseline_cvar_se": 5.00,
    },
    1: {
        "var_99": 784.29,
        "var_999": 1106.79,
        "cvar_999": 1185.89,
        "baseline_cvar_se": 19.13,
    },
    2: {
        "var_99": 871.13,
        "var_999": 1206.35,
        "cvar_999": 1278.60,
        "baseline_cvar_se": 15.61,
    },
    3: {
        "var_99": 755.05,
        "var_999": 1039.88,
        "cvar_999": 1124.49,
        "baseline_cvar_se": 11.38,
    },
}
BASELINE_WALL_TIME_MS = 3500.0
EXPECTED_PATHS_PER_SEED = 8192.0
SHARED_MPI_OUTPUT = Path("/app/experiment/simulation_output.txt")
COMPILE_FAILED_MARKER = Path("/var/tmp/compilation_failed")
COMPILE_OUTPUT_FILE = Path("/var/tmp/compile_output.txt")


def _build_eval(
    score: float | None, label: str, text: str
) -> dict[str, Any]:
  return AlphaEvolveProgramEvaluation(
      scores=AlphaEvolveEvaluationScores(
          scores=[AlphaEvolveEvaluationScore(metric=METRIC_NAME, score=score)]
      ),
      insights=AlphaEvolveEvaluationInsights(
          insights=[AlphaEvolveEvaluationInsight(label=label, text=text)]
      ),
  ).model_dump()


def evaluate(eval_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Compiles and evaluates the C++ FSI Tail VaR/CVaR Monte Carlo kernel."""
  del eval_inputs
  if COMPILE_FAILED_MARKER.exists():
    err_txt = (
        COMPILE_OUTPUT_FILE.read_text()
        if COMPILE_OUTPUT_FILE.exists()
        else "Cluster MPI compilation failed."
    )
    return _build_eval(
        -100.0, "COMPILATION_ERROR", f"Compilation failed:\n{err_txt}"
    )

  if SHARED_MPI_OUTPUT.exists():
    raw_stdout = SHARED_MPI_OUTPUT.read_text()
  else:
    build = subprocess.run(
        ["make", "all"],
        cwd=SAMPLE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
      return _build_eval(
          -100.0, "COMPILATION_ERROR", f"Compilation failed:\n{build.stderr}"
      )

    try:
      run = subprocess.run(
          [str(BINARY_PATH)],
          cwd=SAMPLE_DIR,
          capture_output=True,
          text=True,
          timeout=120,
          check=False,
      )
    except subprocess.TimeoutExpired:
      return _build_eval(
          -100.0, "RUNTIME_TIMEOUT", "Execution timed out (>120s)."
      )

    if run.returncode != 0:
      return _build_eval(
          -100.0,
          "RUNTIME_CRASH",
          f"Runtime crash (exit {run.returncode}):\n{run.stderr}",
      )
    raw_stdout = run.stdout

  json_str = raw_stdout
  first_brace = raw_stdout.find("{")
  last_brace = raw_stdout.rfind("}")
  if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
    json_str = raw_stdout[first_brace : last_brace + 1]

  try:
    data = json.loads(json_str)
  except json.JSONDecodeError as exc:
    return _build_eval(
        -100.0,
        "PARSING_ERROR",
        f"Malformed JSON output ({exc}):\n{raw_stdout[:500]}",
    )

  regimes = data.get("regimes", [])
  if len(regimes) != 4:
    return _build_eval(
        -100.0, "INVALID_OUTPUT", "Expected 4 market regimes in output."
    )

  variance_reduction_ratios = []
  regime_summaries = []

  for reg in regimes:
    r_id = int(reg["id"])
    gold = GOLD_REFERENCE[r_id]
    if "var_99" in reg:
      mean_var = float(reg["var_99"])
      gold_var = gold["var_99"]
    else:
      mean_var = float(reg["var_999"])
      gold_var = gold["var_999"]
    se_var = float(reg["var_se"])
    mean_cv = float(reg["cvar_999"])
    se_cv = float(reg["cvar_se"])
    if "weight_sum" in reg:
      w_sum = float(reg["weight_sum"])
    else:
      w_sum = float(reg.get("mean_weight", 0.0)) * EXPECTED_PATHS_PER_SEED

    # Guardrail 1: Weight Conservation
    if (
        not math.isfinite(w_sum)
        or abs(w_sum - EXPECTED_PATHS_PER_SEED) / EXPECTED_PATHS_PER_SEED
        > 0.015
    ):
      return _build_eval(
          -100.0,
          "WEIGHT_VIOLATION",
          (
              f"Regime {r_id} weight_sum={w_sum:.2f} deviates from "
              f"expected {EXPECTED_PATHS_PER_SEED:.0f} (>1.5%)."
          ),
      )

    # Guardrail 2: Non-Degenerate Seed Variance
    if (
        not math.isfinite(se_var)
        or not math.isfinite(se_cv)
        or se_var < 1e-4
        or se_cv < 1e-4
    ):
      return _build_eval(
          -100.0,
          "DEGENERATE_SEED_VARIANCE",
          (
              f"Regime {r_id} standard error (var_se={se_var:.6f}, "
              f"cvar_se={se_cv:.6f}) is degenerate across 16 seeds."
          ),
      )

    # Guardrail 3: Unbiasedness against 2.1M-path Gold Reference
    err_cv = abs(mean_cv - gold["cvar_999"])
    rel_err_cv = err_cv / gold["cvar_999"]
    err_var = abs(mean_var - gold_var)
    rel_err_var = err_var / gold_var
    tol_cv = max(3.5 * se_cv, 0.008 * gold["cvar_999"])

    if err_cv > tol_cv or rel_err_cv > 0.025 or rel_err_var > 0.025:
      return _build_eval(
          -50.0,
          "BIAS_VIOLATION",
          (
              f"Regime {r_id} biased: CVaR={mean_cv:.2f}+/-{se_cv:.2f} vs "
              f"Gold={gold['cvar_999']:.2f} (rel_err={rel_err_cv*100:.2f}%), "
              f"VaR={mean_var:.2f} vs Gold={gold_var:.2f} "
              f"(rel_err={rel_err_var*100:.2f}%)."
          ),
      )

    vrr = (gold["baseline_cvar_se"] / max(se_cv, 0.25)) ** 2
    vrr = min(vrr, 64.0)
    variance_reduction_ratios.append(vrr)
    regime_summaries.append(
        f"R{r_id}(CVaR={mean_cv:.1f}+/-{se_cv:.2f}, VRR={vrr:.2f}x)"
    )

  wall_ms = max(1.0, float(data.get("wall_time_ms", BASELINE_WALL_TIME_MS)))
  wall_speedup = BASELINE_WALL_TIME_MS / wall_ms
  mean_vrr = sum(variance_reduction_ratios) / len(variance_reduction_ratios)
  effective_speedup = round(wall_speedup * math.sqrt(mean_vrr), 4)

  feedback = (
      f"Effective Speedup={effective_speedup:.4f}x | "
      f"Wall Speedup={wall_speedup:.2f}x ({wall_ms:.1f} ms vs {BASELINE_WALL_TIME_MS:.1f} ms) | "
      f"Mean VRR={mean_vrr:.2f}x | " + ", ".join(regime_summaries)
  )
  return _build_eval(effective_speedup, "FSI_TAIL_MC_SUMMARY", feedback)


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))
