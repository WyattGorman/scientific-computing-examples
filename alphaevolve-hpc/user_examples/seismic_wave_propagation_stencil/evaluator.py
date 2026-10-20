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
"""Evaluator for 2D Acoustic Seismic Wave Propagation Stencil & Sponge Optimization."""

import json
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
BINARY_PATH = Path("/var/tmp/seismic_sim")
METRIC_NAME = "seismic_accuracy_speedup_score"

# Calibrated baseline 2nd-order 5-point stencil reference metrics
BASELINE_WALL_TIME_MS = 78.0
BASELINE_TRACE_NRMS = 0.9581
BASELINE_INTERIOR_NRMS = 1.7940
BASELINE_REFLECTION_RATIO = 0.6647


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
  """Compiles and evaluates the C++ 2D seismic wavefield propagator."""
  del eval_inputs
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
        timeout=90,
        check=False,
    )
  except subprocess.TimeoutExpired:
    return _build_eval(-100.0, "RUNTIME_TIMEOUT", "Execution timed out (>90s).")

  if run.returncode != 0:
    return _build_eval(
        -100.0,
        "RUNTIME_CRASH",
        f"Runtime crash (exit {run.returncode}):\n{run.stderr}",
    )

  try:
    data = json.loads(run.stdout)
  except json.JSONDecodeError as exc:
    return _build_eval(
        -100.0,
        "PARSING_ERROR",
        f"Malformed JSON output ({exc}):\n{run.stdout[:500]}",
    )

  # Guardrail 1: Extended-time CFL stability (Nt_long = 1400 steps)
  if not data.get("cfl_stable", False):
    return _build_eval(
        -80.0,
        "CFL_INSTABILITY_DETECTED",
        (
            f"Extended-horizon energy ratio "
            f"{data.get('late_energy_ratio', 999.0):.4f} > 1.05 limit or NaN/Inf."
        ),
    )

  m1 = data["marmousi"]
  m2 = data["subsalt_fault"]

  # Guardrail 2: Non-trivial wavefield energy conservation check (prevents u=0 exploit)
  for name, m in [("Marmousi", m1), ("SubSalt_Fault", m2)]:
    e_ratio = float(m.get("trace_energy_ratio", 0.0))
    if e_ratio < 0.35 or e_ratio > 3.00:
      return _build_eval(
          -60.0,
          "WAVEFIELD_ENERGY_VIOLATION",
          (
              f"{name}: trace_energy_ratio={e_ratio:.4f} outside "
              f"physical range [0.35, 3.00]."
          ),
      )

  mean_trace_nrms = 0.5 * (m1["trace_nrms"] + m2["trace_nrms"])
  mean_interior_nrms = 0.5 * (m1["interior_nrms"] + m2["interior_nrms"])
  mean_refl_ratio = 0.5 * (
      m1["late_reflection_ratio"] + m2["late_reflection_ratio"]
  )

  # Guardrail 3: Reject severely diverged wavefields
  if mean_trace_nrms > 1.35 or mean_interior_nrms > 2.40:
    return _build_eval(
        -50.0,
        "WAVEFIELD_ACCURACY_VIOLATION",
        (
            f"Mean Trace NRMS={mean_trace_nrms:.4f}, "
            f"Interior NRMS={mean_interior_nrms:.4f} exceeds accuracy gate."
        ),
    )

  wall_ms = max(1.0, float(data["wall_time_ms"]))
  gcell_s = float(data["throughput_gcell_s"])

  trace_accuracy_gain = BASELINE_TRACE_NRMS / max(mean_trace_nrms, 0.04)
  interior_accuracy_gain = BASELINE_INTERIOR_NRMS / max(
      mean_interior_nrms, 0.04
  )
  reflection_suppression_gain = BASELINE_REFLECTION_RATIO / max(
      mean_refl_ratio, 0.03
  )
  speedup_factor = min(2.5, max(0.35, BASELINE_WALL_TIME_MS / wall_ms))

  composite_score = round(
      (
          0.40 * trace_accuracy_gain
          + 0.35 * interior_accuracy_gain
          + 0.25 * reflection_suppression_gain
      )
      * (0.75 + 0.25 * speedup_factor),
      4,
  )

  feedback = (
      f"Seismic 2D Wavefield Evaluation Complete | "
      f"Score={composite_score:.4f} (Throughput={gcell_s:.3f} GCell/s, Wall={wall_ms:.2f} ms) | "
      f"Mean_Trace_NRMS={mean_trace_nrms:.4f} (Marmousi={m1['trace_nrms']:.4f}, "
      f"SubSalt_Fault={m2['trace_nrms']:.4f}) | "
      f"Mean_Interior_NRMS={mean_interior_nrms:.4f} (Marmousi={m1['interior_nrms']:.4f}, "
      f"SubSalt_Fault={m2['interior_nrms']:.4f}) | "
      f"Late_Reflection_Ratio={mean_refl_ratio:.4f} | "
      f"Extended_CFL_Energy_Ratio={data['late_energy_ratio']:.4f}"
  )

  return _build_eval(composite_score, "SEISMIC_2D_SUMMARY", feedback)


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))
