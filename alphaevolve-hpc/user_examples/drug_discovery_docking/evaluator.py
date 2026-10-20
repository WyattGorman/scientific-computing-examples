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
"""Two-stage evaluation cascade for 3D structure-based drug discovery."""

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

from src import docking_oracle  # pylint: disable=g-import-not-at-top
try:
  import main  # pylint: disable=g-import-not-at-top
except Exception:  # pylint: disable=broad-exception-caught
  main = None  # type: ignore[assignment]

MODELS_DIR = SAMPLE_DIR / "models"
PRIMARY_POCKET_PATH = MODELS_DIR / "kras_g12d_pocket.json"
HELDOUT_POCKET_PATH = MODELS_DIR / "egfr_t790m_pocket.json"
METRIC_NAME = "multiparameter_docking_score"

SEED_SCAFFOLDS = [
    "c1ccc2c(c1)ncn2",          # Benzimidazole core (N-substitutable)
    "c1ccc2c(c1)ncnc2",         # Quinazoline core (C4-substitutable)
    "c1cnc2cccnc2c1",           # Pyrido/naphthyridine core (C-substitutable)
    "c1ccc2[nH]ccc2c1",        # Indole core
]

TOP_K = 8


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


def evaluate_target(pocket_path: Path, n_confs: int) -> dict[str, Any]:
  """Evaluates `main.propose_and_rank_ligands` on a single 3D receptor pocket."""
  global main
  pocket = docking_oracle.load_pocket(pocket_path)
  try:
    if main is None:
      import importlib  # pylint: disable=g-import-not-at-top
      main = importlib.import_module("main")
    proposed = main.propose_and_rank_ligands(
        pocket, list(SEED_SCAFFOLDS), top_k=TOP_K
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return {
        "valid_count": 0,
        "error": f"Candidate execution error: {exc}",
    }
  if not isinstance(proposed, list) or not proposed:
    return {
        "valid_count": 0,
        "error": "propose_and_rank_ligands must return a non-empty list of SMILES.",
    }

  proposed = [str(s).strip() for s in proposed[:TOP_K] if str(s).strip()]
  if not proposed:
    return {
        "valid_count": 0,
        "error": "No non-empty SMILES strings returned.",
    }

  evals = [
      docking_oracle.evaluate_molecule_3d(smi, pocket, n_confs=n_confs)
      for smi in proposed
  ]
  valid_evals = [e for e in evals if e.get("valid", False)]
  if len(valid_evals) < 3:
    reasons = [e.get("reason", "unknown") for e in evals if not e.get("valid")]
    return {
        "valid_count": len(valid_evals),
        "error": (
            f"Fewer than 3 chemically valid molecules returned "
            f"({len(valid_evals)}/{len(proposed)}). Rejections: {reasons[:3]}"
        ),
    }

  div = docking_oracle.compute_scaffold_diversity(
      [e["smiles"] for e in valid_evals]
  )
  if div < 0.22:
    return {
        "valid_count": len(valid_evals),
        "diversity": div,
        "error": (
            f"Scaffold collapse detected: mean pairwise Tanimoto distance "
            f"{div:.3f} < 0.22 minimum diversity threshold."
        ),
    }

  ranked_valid = sorted(
      valid_evals, key=lambda x: x["molecule_score"], reverse=True
  )
  top3 = ranked_valid[:3]
  mean_top3_score = sum(x["molecule_score"] for x in top3) / len(top3)
  best = top3[0]

  return {
      "target_id": pocket["target_id"],
      "valid_count": len(valid_evals),
      "total_proposed": len(proposed),
      "proposed_smiles": [e["smiles"] for e in valid_evals],
      "diversity": round(div, 4),
      "mean_top3_score": round(mean_top3_score, 4),
      "best_smiles": best["smiles"],
      "best_vina_dg": best["vina_dg_kcal"],
      "best_qed": best["qed"],
      "best_sa_score": best["sa_score"],
      "best_strain_kcal": best["strain_kcal"],
      "best_anchor_match": best["anchor_match_ratio"],
      "used_rdkit": bool(best.get("used_rdkit", False)),
  }


def evaluate(eval_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Evaluates candidate ligand generator across KRAS G12D and held-out EGFR T790M."""
  del eval_inputs
  t0 = time.perf_counter()
  fast_mode = os.environ.get("ALPHAEVOLVE_DOCKING_FAST", "0") == "1"
  s1_confs = 2 if fast_mode else 4
  s2_confs = 4 if fast_mode else 8

  primary_res = evaluate_target(PRIMARY_POCKET_PATH, n_confs=s1_confs)
  if "error" in primary_res:
    return _build_eval(
        -100.0,
        "PRIMARY_TARGET_FAILURE",
        f"KRAS_G12D failed: {primary_res['error']}",
    )

  heldout_res = evaluate_target(HELDOUT_POCKET_PATH, n_confs=s2_confs)
  if "error" in heldout_res:
    return _build_eval(
        -50.0,
        "HELDOUT_TARGET_FAILURE",
        f"EGFR_T790M failed: {heldout_res['error']}",
    )

  primary_s2 = evaluate_target(PRIMARY_POCKET_PATH, n_confs=s2_confs)
  if "error" not in primary_s2:
    primary_res = primary_s2

  is_pocket_conditioned = (
      set(primary_res["proposed_smiles"]) != set(heldout_res["proposed_smiles"])
      and primary_res["best_smiles"] != heldout_res["best_smiles"]
  )
  conditioning_bonus = 0.55 if is_pocket_conditioned else -0.35

  diversity_bonus = 0.5 * (
      0.65 * primary_res["diversity"] + 0.35 * heldout_res["diversity"]
  )
  composite_score = round(
      0.65 * primary_res["mean_top3_score"]
      + 0.35 * heldout_res["mean_top3_score"]
      + diversity_bonus
      + conditioning_bonus,
      4,
  )

  elapsed_s = time.perf_counter() - t0
  used_rdkit = bool(primary_res.get("used_rdkit", False))
  backend_tag = "RDKit_ETKDGv3_MMFF94s" if used_rdkit else "Fallback_3D"
  feedback = (
      f"3D Docking Evaluation Complete in {elapsed_s:.2f}s "
      f"(RDKit={used_rdkit}, backend={backend_tag}) | "
      f"Composite Score={composite_score:.4f} | "
      f"KRAS_G12D(top3={primary_res['mean_top3_score']:.3f}, "
      f"best_dG={primary_res['best_vina_dg']:.2f} kcal/mol, "
      f"QED={primary_res['best_qed']:.2f}, SA={primary_res['best_sa_score']:.2f}, "
      f"strain={primary_res['best_strain_kcal']:.2f} kcal/mol, "
      f"anchors={primary_res['best_anchor_match']:.2f}, "
      f"div={primary_res['diversity']:.2f}, best='{primary_res['best_smiles']}') | "
      f"EGFR_T790M_HeldOut(top3={heldout_res['mean_top3_score']:.3f}, "
      f"best_dG={heldout_res['best_vina_dg']:.2f} kcal/mol, "
      f"QED={heldout_res['best_qed']:.2f}, SA={heldout_res['best_sa_score']:.2f}, "
      f"anchors={heldout_res['best_anchor_match']:.2f}, "
      f"div={heldout_res['diversity']:.2f}, best='{heldout_res['best_smiles']}')"
  )

  return _build_eval(composite_score, "DOCKING_3D_SUMMARY", feedback)


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))
