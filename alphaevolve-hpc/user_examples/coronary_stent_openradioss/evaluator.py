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
"""Evaluator for 3D L605 CoCr coronary stent deployment and recoil optimization."""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from alpha_evolve.models import (
    AlphaEvolveEvaluationInsight,
    AlphaEvolveEvaluationInsights,
    AlphaEvolveEvaluationScore,
    AlphaEvolveEvaluationScores,
    AlphaEvolveProgramEvaluation,
)

SAMPLE_DIR = Path(__file__).absolute().parent
if str(SAMPLE_DIR) in sys.path:
  sys.path.remove(str(SAMPLE_DIR))
sys.path.insert(0, str(SAMPLE_DIR))

from src import stent_radioss_runner  # pylint: disable=g-import-not-at-top
try:
  import main  # pylint: disable=g-import-not-at-top
except Exception:  # pylint: disable=broad-exception-caught
  main = None  # type: ignore[assignment]

METRIC_NAME = "stent_clinical_fitness_score"


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
  """Evaluates 3-stage L605 CoCr coronary stent inflation, recoil, and crush."""
  global main
  del eval_inputs
  t0 = time.perf_counter()
  verify_bin = os.environ.get("ALPHAEVOLVE_STENT_VERIFY_BINARY", "1") == "1"

  try:
    if main is None:
      import importlib  # pylint: disable=g-import-not-at-top
      main = importlib.import_module("main")
    params = main.generate_stent_cell_parameters(
        dict(stent_radioss_runner.DESIGN_BOUNDS)
    )
    if not isinstance(params, Mapping):
      return _build_eval(
          -100.0,
          "CANDIDATE_EXECUTION_ERROR",
          "generate_stent_cell_parameters must return a dict of parameters.",
      )
    sim = stent_radioss_runner.simulate_3stage_stent_deployment(
        dict(params), verify_binary=verify_bin
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return _build_eval(
        -100.0,
        "CANDIDATE_EXECUTION_ERROR",
        f"Candidate execution failed: {exc}",
    )

  if not sim.get("valid", False):
    return _build_eval(
        -100.0,
        "GEOMETRY_OR_CRIMP_VIOLATION",
        f"Invalid stent geometry: {sim.get('error', 'invalid')}",
    )

  recoil_pct = sim["recoil_pct"]
  fs_pct = sim["foreshortening_pct"]
  db_pct = sim["dogboning_pct"]
  max_eps_p = sim["max_eps_p"]
  k_radial = sim["k_radial_kpa"]
  msa_pct = sim["geometry"]["msa_pct"]
  crimp_clear = sim["geometry"]["crimp_clearance_um"]

  strain_penalty = 0.0
  if max_eps_p > stent_radioss_runner.EPS_P_ALLOWABLE:
    excess = max_eps_p - stent_radioss_runner.EPS_P_ALLOWABLE
    strain_penalty = 250.0 * excess + 15.0

  stiffness_term = 3.5 * (k_radial / 24.0)
  recoil_term = -1.35 * max(0.0, recoil_pct - 2.2)
  foreshortening_term = -0.65 * fs_pct
  dogboning_term = -0.45 * db_pct
  strain_margin_bonus = 2.0 * max(
      0.0, stent_radioss_runner.EPS_P_ALLOWABLE - max_eps_p
  )
  msa_target_penalty = -0.08 * ((msa_pct - 14.5) ** 2)

  fitness_score = round(
      stiffness_term
      + recoil_term
      + foreshortening_term
      + dogboning_term
      + strain_margin_bonus
      + msa_target_penalty
      - strain_penalty,
      4,
  )

  elapsed_s = time.perf_counter() - t0
  bin_info = sim.get("openradioss_binary", {})
  bin_status = bin_info.get("status", "skipped")
  vtk_stages = bin_info.get("vtk_stages_parsed", 0)
  csv_rows = bin_info.get("csv_rows", 0)
  feedback = (
      f"Stent 3-Stage Deployment Complete in {elapsed_s:.2f}s "
      f"(OpenRADIOSS={bin_status}, VTK_Stages={vtk_stages}, CSV_Rows={csv_rows}) | "
      f"Fitness={fitness_score:.4f} | "
      f"K_radial={k_radial:.2f} kPa, "
      f"Recoil={recoil_pct:.2f}% (OD_dep={sim['od_deployed_mm']:.3f} mm), "
      f"MaxPlasticStrain={max_eps_p:.4f} (limit={stent_radioss_runner.EPS_P_ALLOWABLE:.3f}), "
      f"MaxVonMises={sim['max_von_mises_mpa']:.1f} MPa, "
      f"Foreshortening={fs_pct:.2f}%, Dogboning={db_pct:.2f}%, "
      f"MSA={msa_pct:.2f}%, CrimpClearance={crimp_clear:.1f} um"
  )

  return _build_eval(fitness_score, "STENT_3STAGE_SUMMARY", feedback)


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))
