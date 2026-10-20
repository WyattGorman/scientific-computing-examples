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
"""Two-stage OpenROAD floorplan & multi-corner STA PPA evaluator."""

import json
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

from src import openroad_runner  # pylint: disable=g-import-not-at-top
try:
  import main  # pylint: disable=g-import-not-at-top
except Exception:  # pylint: disable=broad-exception-caught
  main = None  # type: ignore[assignment]

NETLIST_PATH = SAMPLE_DIR / "models" / "asap7_riscv_npu_tile.json"
METRIC_NAME = "ppa_composite_score"


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
  """Evaluates 7nm ASAP7 macro floorplan across RePlAce, FastRoute, and OpenSTA."""
  global main
  del eval_inputs
  t0 = time.perf_counter()
  netlist = openroad_runner.load_netlist_graph(NETLIST_PATH)
  try:
    if main is None:
      import importlib  # pylint: disable=g-import-not-at-top
      main = importlib.import_module("main")
    floorplan = main.compute_macro_floorplan(netlist)
    if not isinstance(floorplan, Mapping):
      return _build_eval(
          -200.0,
          "FLOORPLAN_ILLEGAL",
          "compute_macro_floorplan must return a mapping of macro placements.",
      )
    stage1 = openroad_runner.evaluate_floorplan_ppa(
        netlist, dict(floorplan), fast_stage=1
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return _build_eval(
        -200.0,
        "CANDIDATE_EXECUTION_ERROR",
        f"Candidate execution failed: {exc}",
    )
  if not stage1.get("valid", False):
    return _build_eval(
        -200.0,
        "FLOORPLAN_ILLEGAL",
        f"Floorplan DRC violation: {stage1.get('error', 'invalid')}",
    )

  if stage1["routing_overflow"] > 1800.0:
    return _build_eval(
        -100.0,
        "STAGE1_ROUTING_CONGESTION_ABORT",
        (
            f"GCell overflow {stage1['routing_overflow']:.1f} > 1800.0 limit."
        ),
    )

  res = openroad_runner.evaluate_floorplan_ppa(
      netlist, floorplan, fast_stage=2
  )
  wns_ps = res["worst_wns_ps"]
  tns_ps = res["total_tns_ps"]
  hpwl_mm = res["weighted_hpwl_mm"]
  power_mw = res["total_power_mw"]
  overflow = res["routing_overflow"]
  max_util = res["max_gcell_utilization"]
  cts_skew = res["cts_skew_base_ps"]

  timing_term = (
      0.35 * min(wns_ps, 35.0)
      if wns_ps >= 0.0
      else 0.85 * wns_ps + 0.08 * tns_ps
  )
  wirelength_term = 25.0 * ((1350.0 - hpwl_mm) / 1350.0)
  power_term = 18.0 * ((66.0 - power_mw) / 66.0)
  congestion_penalty = -0.35 * overflow - 12.0 * max(0.0, max_util - 0.92)

  ppa_score = round(
      timing_term + wirelength_term + power_term + congestion_penalty, 4
  )

  elapsed_s = time.perf_counter() - t0
  ss_wns = res["corners"]["ss_0p63v_100c"]["wns_ps"]
  tt_wns = res["corners"]["tt_0p70v_25c"]["wns_ps"]
  ff_wns = res["corners"]["ff_0p77v_m40c"]["wns_ps"]

  or_bin = res.get("openroad_binary", {})
  or_status = or_bin.get("status", "none")

  feedback = (
      f"OpenROAD 7nm ASAP7 PPA Evaluation Complete in {elapsed_s:.2f}s "
      f"(OpenROAD_Bin={or_status}) | "
      f"PPA_Score={ppa_score:.4f} | "
      f"Worst_WNS={wns_ps:.2f} ps (SS={ss_wns:.1f}, TT={tt_wns:.1f}, FF={ff_wns:.1f}), "
      f"Total_TNS={tns_ps:.2f} ps, CTS_Skew={cts_skew:.2f} ps (spine_pen={res['spine_blockage_penalty_ps']:.1f} ps) | "
      f"Weighted_HPWL={hpwl_mm:.1f} mm, Total_Power={power_mw:.2f} mW | "
      f"Routing_Overflow={overflow:.1f}, Max_GCell_Util={max_util:.3f}, "
      f"Min_Channel={res['min_channel_um']:.1f} um"
  )

  return _build_eval(ppa_score, "OPENROAD_PPA_SUMMARY", feedback)


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))
