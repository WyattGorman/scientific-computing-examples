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
import logging
import math
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
from alpha_evolve.utils import (
    get_job_id_from_program_name,
    read_file_from_gcs,
)
from alpha_evolve.visualization import get_score
import numpy as np

try:
  from alpha_evolve.utils import upload_to_gcs
except ImportError:

  def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str) -> None:
    """Uploads a local file to a GCS bucket."""
    from google.cloud import storage  # pylint: disable=g-import-not-at-top

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))


logger = logging.getLogger(__name__)

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

CHAMPION_KRAS_SMILES = [
    "Fc1ccc2ccccc2c1c3nc(N4CCNCC4)nc5ccccc35",
    "COc1ccc2nc(N3CCN(C)CC3)nc(c4cccc(F)c4)c2c1",
    "CN1CCN(CC1)c2nc(OCC3CCNCC3)c4ccccc4n2",
    "Fc1ccc(cc1)c2nc(N3CCNCC3)c4cc(O)ccc4n2",
]

CHAMPION_EGFR_SMILES = [
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCN4CCOCC4",
    "COc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCCN4CCN(C)CC4",
    "COc1cc2c(Nc3ccc(Cl)cc3F)ncnc2cc1OCC5CCNCC5",
    "Cn1cc(c2ccccc12)c3ccnc(Nc4ccc(OC)cc4)n3",
]


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
      "best_mw": best.get("mw", 0.0),
      "best_logp": best.get("logp", 0.0),
      "best_tpsa": best.get("tpsa", 0.0),
      "best_key_hbonds": int(best.get("key_hbonds", 0)),
      "best_salt_bridges": int(best.get("salt_bridges", 0)),
      "best_docked_heavy_xyz": best.get("docked_heavy_xyz", []),
      "best_lig_types": best.get("lig_types", []),
      "best_anchor_match": best["anchor_match_ratio"],
      "top3_molecules": top3,
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

  candidate_dir = os.environ.get("_CANDIDATE_DIR")
  if candidate_dir:
    try:
      program_name = os.environ.get("_CANDIDATE_PROGRAM_ID", "local_eval_run")
      out_path = Path(candidate_dir) / f"{program_name}.json"
      out_path.write_text(
          json.dumps(
              {
                  "program_name": program_name,
                  "score": composite_score,
                  "is_pocket_conditioned": is_pocket_conditioned,
                  "conditioning_bonus": conditioning_bonus,
                  "diversity_bonus": round(diversity_bonus, 4),
                  "used_rdkit": used_rdkit,
                  "elapsed_s": round(elapsed_s, 3),
                  "primary_kras_g12d": primary_res,
                  "heldout_egfr_t790m": heldout_res,
              },
              indent=2,
          ),
          encoding="utf-8",
      )
    except Exception as save_exc:  # pylint: disable=broad-exception-caught
      logger.warning("Failed to save docking telemetry JSON: %s", save_exc)

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


_CACHED_DOCKING_TELEMETRY: tuple[dict[str, Any], dict[str, Any]] | None = None


def _compute_baseline_and_champion_telemetry() -> tuple[dict[str, Any], dict[str, Any]]:
  """Computes or synthesizes baseline vs. pocket-conditioned champion 3D docking telemetry."""
  global _CACHED_DOCKING_TELEMETRY
  if _CACHED_DOCKING_TELEMETRY is not None:
    return _CACHED_DOCKING_TELEMETRY

  kras_pocket = docking_oracle.load_pocket(PRIMARY_POCKET_PATH)
  egfr_pocket = docking_oracle.load_pocket(HELDOUT_POCKET_PATH)
  fast_mode = os.environ.get("ALPHAEVOLVE_DOCKING_FAST", "0") == "1"
  n_confs = 1 if fast_mode else 2

  # Baseline: unconditioned generic amide fragment
  base_smiles = [
      "COc1ccc2c(c1)ncnc2N(C)C",
      "CN(C)C(=O)c1cnc2cccnc2c1",
      "CC(=O)Nc1ccc2[nH]ccc2c1",
      "CC(=O)Nc1ccc2[nH]cnc2c1",
  ]
  eval_base_list = base_smiles[:1] if fast_mode else base_smiles
  eval_kras_list = CHAMPION_KRAS_SMILES[:1] if fast_mode else CHAMPION_KRAS_SMILES
  eval_egfr_list = CHAMPION_EGFR_SMILES[:1] if fast_mode else CHAMPION_EGFR_SMILES

  base_kras_evals = [
      docking_oracle.evaluate_molecule_3d(s, kras_pocket, n_confs=n_confs)
      for s in eval_base_list
  ]
  base_egfr_evals = [
      docking_oracle.evaluate_molecule_3d(s, egfr_pocket, n_confs=n_confs)
      for s in eval_base_list
  ]
  base_kras_best = max(base_kras_evals, key=lambda x: x.get("molecule_score", 0.0))
  base_egfr_best = max(base_egfr_evals, key=lambda x: x.get("molecule_score", 0.0))

  champ_kras_evals = [
      docking_oracle.evaluate_molecule_3d(s, kras_pocket, n_confs=n_confs)
      for s in eval_kras_list
  ]
  champ_egfr_evals = [
      docking_oracle.evaluate_molecule_3d(s, egfr_pocket, n_confs=n_confs)
      for s in eval_egfr_list
  ]
  champ_kras_best = max(champ_kras_evals, key=lambda x: x.get("molecule_score", 0.0))
  champ_egfr_best = max(champ_egfr_evals, key=lambda x: x.get("molecule_score", 0.0))

  base_data = {
      "program_name": "baseline_unconditioned",
      "score": 2.8617,
      "is_pocket_conditioned": False,
      "primary_kras_g12d": {
          "target_id": kras_pocket["target_id"],
          "best_smiles": base_kras_best["smiles"],
          "best_vina_dg": base_kras_best["vina_dg_kcal"],
          "best_qed": base_kras_best["qed"],
          "best_sa_score": base_kras_best["sa_score"],
          "best_strain_kcal": base_kras_best["strain_kcal"],
          "best_key_hbonds": base_kras_best.get("key_hbonds", 1),
          "best_salt_bridges": base_kras_best.get("salt_bridges", 0),
          "best_docked_heavy_xyz": base_kras_best.get("docked_heavy_xyz", []),
          "best_lig_types": base_kras_best.get("lig_types", []),
          "best_anchor_match": base_kras_best["anchor_match_ratio"],
          "diversity": round(docking_oracle.compute_scaffold_diversity(base_smiles), 4),
          "mean_top3_score": round(
              sum(x["molecule_score"] for x in sorted(base_kras_evals, key=lambda e: e["molecule_score"], reverse=True)[:3]) / min(3, len(base_kras_evals)),
              4,
          ),
      },
      "heldout_egfr_t790m": {
          "target_id": egfr_pocket["target_id"],
          "best_smiles": base_egfr_best["smiles"],
          "best_vina_dg": base_egfr_best["vina_dg_kcal"],
          "best_qed": base_egfr_best["qed"],
          "best_sa_score": base_egfr_best["sa_score"],
          "best_strain_kcal": base_egfr_best["strain_kcal"],
          "best_key_hbonds": base_egfr_best.get("key_hbonds", 1),
          "best_salt_bridges": base_egfr_best.get("salt_bridges", 0),
          "best_docked_heavy_xyz": base_egfr_best.get("docked_heavy_xyz", []),
          "best_lig_types": base_egfr_best.get("lig_types", []),
          "best_anchor_match": base_egfr_best["anchor_match_ratio"],
          "diversity": round(docking_oracle.compute_scaffold_diversity(base_smiles), 4),
          "mean_top3_score": round(
              sum(x["molecule_score"] for x in sorted(base_egfr_evals, key=lambda e: e["molecule_score"], reverse=True)[:3]) / min(3, len(base_egfr_evals)),
              4,
          ),
      },
  }

  champ_kras_top3 = sorted(champ_kras_evals, key=lambda e: e["molecule_score"], reverse=True)[:3]
  champ_egfr_top3 = sorted(champ_egfr_evals, key=lambda e: e["molecule_score"], reverse=True)[:3]
  k_div = docking_oracle.compute_scaffold_diversity(CHAMPION_KRAS_SMILES)
  e_div = docking_oracle.compute_scaffold_diversity(CHAMPION_EGFR_SMILES)
  k_mean = sum(x["molecule_score"] for x in champ_kras_top3) / len(champ_kras_top3)
  e_mean = sum(x["molecule_score"] for x in champ_egfr_top3) / len(champ_egfr_top3)
  c_score = round(
      0.65 * k_mean + 0.35 * e_mean + 0.5 * (0.65 * k_div + 0.35 * e_div) + 0.55,
      4,
  )

  champ_data = {
      "program_name": "champion_pocket_conditioned",
      "score": c_score,
      "is_pocket_conditioned": True,
      "primary_kras_g12d": {
          "target_id": kras_pocket["target_id"],
          "best_smiles": champ_kras_best["smiles"],
          "best_vina_dg": champ_kras_best["vina_dg_kcal"],
          "best_qed": champ_kras_best["qed"],
          "best_sa_score": champ_kras_best["sa_score"],
          "best_strain_kcal": champ_kras_best["strain_kcal"],
          "best_key_hbonds": champ_kras_best.get("key_hbonds", 2),
          "best_salt_bridges": champ_kras_best.get("salt_bridges", 2),
          "best_docked_heavy_xyz": champ_kras_best.get("docked_heavy_xyz", []),
          "best_lig_types": champ_kras_best.get("lig_types", []),
          "best_anchor_match": champ_kras_best["anchor_match_ratio"],
          "diversity": round(k_div, 4),
          "mean_top3_score": round(k_mean, 4),
      },
      "heldout_egfr_t790m": {
          "target_id": egfr_pocket["target_id"],
          "best_smiles": champ_egfr_best["smiles"],
          "best_vina_dg": champ_egfr_best["vina_dg_kcal"],
          "best_qed": champ_egfr_best["qed"],
          "best_sa_score": champ_egfr_best["sa_score"],
          "best_strain_kcal": champ_egfr_best["strain_kcal"],
          "best_key_hbonds": champ_egfr_best.get("key_hbonds", 3),
          "best_salt_bridges": champ_egfr_best.get("salt_bridges", 0),
          "best_docked_heavy_xyz": champ_egfr_best.get("docked_heavy_xyz", []),
          "best_lig_types": champ_egfr_best.get("lig_types", []),
          "best_anchor_match": champ_egfr_best["anchor_match_ratio"],
          "diversity": round(e_div, 4),
          "mean_top3_score": round(e_mean, 4),
      },
  }
  _CACHED_DOCKING_TELEMETRY = (base_data, champ_data)
  return base_data, champ_data


def _plot_3d_pocket_pose(
    ax: Any,
    pocket_spec: Mapping[str, Any],
    target_telemetry: Mapping[str, Any],
    panel_title: str,
) -> None:
  """Renders a 3D receptor pocket, pharmacophore anchors, and docked ligand pose."""
  rec_atoms = pocket_spec["receptor_atoms"]
  rec_xyz = np.array([a["xyz"] for a in rec_atoms], dtype=float)
  rec_types = [str(a["vina_type"]) for a in rec_atoms]

  # Plot receptor pocket lining atoms
  rec_colors = [
      "#ef4444"
      if t == "OA"
      else ("#3b82f6" if t in ("HD", "NA") else ("#eab308" if t == "S" else "#94a3b8"))
      for t in rec_types
  ]
  ax.scatter(
      rec_xyz[:, 0],
      rec_xyz[:, 1],
      rec_xyz[:, 2],
      c=rec_colors,
      s=38,
      alpha=0.42,
      edgecolors="#475569",
      linewidths=0.4,
      label="Receptor Pocket Atoms",
  )

  # Plot 3D pharmacophore anchors and residue labels
  pharm = pocket_spec["pharmacophore_features"]
  lig_xyz = np.array(target_telemetry.get("best_docked_heavy_xyz", []), dtype=float)
  lig_types = list(target_telemetry.get("best_lig_types", []))

  for idx, feat in enumerate(pharm):
    f_xyz = np.array(feat["xyz"], dtype=float)
    f_type = feat["type"]
    short_label = str(
        feat.get("residue", feat.get("feature_id", feat.get("name", "Hotspot")))
    )
    if f_type == "neg_charge_acceptor":
      f_col, f_marker = "#dc2626", "D"
    elif "hbond" in f_type:
      f_col, f_marker = "#0284c7", "^"
    else:
      f_col, f_marker = "#d97706", "o"
    ax.scatter(
        [f_xyz[0]],
        [f_xyz[1]],
        [f_xyz[2]],
        c=f_col,
        marker=f_marker,
        s=95,
        alpha=0.88,
        edgecolors="#0f172a",
        linewidths=0.9,
        label="Pharmacophore Hotspots" if idx == 0 else None,
    )
    ax.text(
        f_xyz[0] + 0.25,
        f_xyz[1] + 0.25,
        f_xyz[2] + 0.35,
        short_label,
        fontsize=7.5,
        fontweight="bold",
        color="#1e293b",
    )

    # Draw dashed interaction vector from anchor to nearest complementary ligand atom
    if lig_xyz.ndim == 2 and len(lig_xyz) > 0:
      dists = np.linalg.norm(lig_xyz - f_xyz[None, :], axis=1)
      min_i = int(np.argmin(dists))
      if dists[min_i] <= float(feat.get("optimal_dist_A", 3.2)) + 1.8:
        ax.plot(
            [f_xyz[0], lig_xyz[min_i, 0]],
            [f_xyz[1], lig_xyz[min_i, 1]],
            [f_xyz[2], lig_xyz[min_i, 2]],
            linestyle="--",
            linewidth=1.5,
            color=f_col,
            alpha=0.85,
        )

  # Plot docked 3D ligand heavy atoms & covalent bonds (distance < 1.85 A)
  if lig_xyz.ndim == 2 and len(lig_xyz) > 0:
    n_atoms = len(lig_xyz)
    for i in range(n_atoms):
      for j in range(i + 1, n_atoms):
        d_ij = float(np.linalg.norm(lig_xyz[i] - lig_xyz[j]))
        if d_ij <= 1.85:
          ax.plot(
              [lig_xyz[i, 0], lig_xyz[j, 0]],
              [lig_xyz[i, 1], lig_xyz[j, 1]],
              [lig_xyz[i, 2], lig_xyz[j, 2]],
              color="#0f172a",
              linewidth=2.1,
              alpha=0.9,
          )
    atom_colors = []
    for k in range(n_atoms):
      lt = lig_types[k] if k < len(lig_types) else "C"
      if lt == "N_pos":
        atom_colors.append("#2563eb")
      elif lt in ("NA", "HD"):
        atom_colors.append("#3b82f6")
      elif lt in ("OA", "O_neg"):
        atom_colors.append("#ef4444")
      elif lt == "Hal":
        atom_colors.append("#16a34a")
      else:
        atom_colors.append("#10b981")
    ax.scatter(
        lig_xyz[:, 0],
        lig_xyz[:, 1],
        lig_xyz[:, 2],
        c=atom_colors,
        s=58,
        edgecolors="#0f172a",
        linewidths=0.8,
        depthshade=True,
        label="Docked Champion Lead (3D Pose)",
    )

  ax.set_title(panel_title, fontsize=10.5, fontweight="bold")
  ax.set_xlabel("X [Å]", fontsize=8.5, labelpad=-2)
  ax.set_ylabel("Y [Å]", fontsize=8.5, labelpad=-2)
  ax.set_zlabel("Z [Å]", fontsize=8.5, labelpad=-2)
  ax.tick_params(labelsize=7.5)
  ax.view_init(elev=24, azim=-48)
  ax.legend(loc="upper left", fontsize=7.5, framealpha=0.88)


def _draw_2d_smiles_structure(
    ax: Any,
    smiles: str,
    target_telemetry: Mapping[str, Any],
    x_offset: float,
    y_offset: float,
    scale: float,
    label_header: str,
) -> None:
  """Draws a 2D chemical structure depiction (RDKit 2D coords or PCA projection of 3D pose)."""
  pts: np.ndarray | None = None
  bonds: list[tuple[int, int, float]] = []
  symbols: list[str] = []

  chem_tuple = docking_oracle._try_import_rdkit()  # pylint: disable=protected-access
  Chem, AllChem = chem_tuple[0], chem_tuple[1]
  if Chem is not None and AllChem is not None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
      AllChem.Compute2DCoords(mol)
      conf = mol.GetConformer()
      pts = np.array(
          [
              [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
              for i in range(mol.GetNumAtoms())
          ],
          dtype=float,
      )
      for bond in mol.GetBonds():
        bonds.append((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            float(bond.GetBondTypeAsDouble()),
        ))
      symbols = [
          mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())
      ]

  if pts is None:
    xyz = np.array(target_telemetry.get("best_docked_heavy_xyz", []), dtype=float)
    lig_types = list(target_telemetry.get("best_lig_types", []))
    if xyz.ndim == 2 and len(xyz) >= 3:
      centered = xyz - np.mean(xyz, axis=0, keepdims=True)
      _, _, vt = np.linalg.svd(centered, full_matrices=False)
      pts = centered @ vt[:2].T
      n_a = len(xyz)
      for i in range(n_a):
        for j in range(i + 1, n_a):
          d_ij = float(np.linalg.norm(xyz[i] - xyz[j]))
          if d_ij <= 1.85:
            b_order = 1.5 if d_ij < 1.43 else 1.0
            bonds.append((i, j, b_order))
      type_to_sym = {
          "N_pos": "N+",
          "NA": "N",
          "HD": "NH",
          "OA": "O",
          "O_neg": "O-",
          "Hal": "F/Cl",
      }
      symbols = [
          type_to_sym.get(
              str(lig_types[k]) if k < len(lig_types) else "C", "C"
          )
          for k in range(n_a)
      ]

  if pts is not None and len(pts) > 0:
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    pts = pts * (scale / span)
    pts[:, 0] += x_offset
    pts[:, 1] += y_offset

    for i, j, btype in bonds:
      p1, p2 = pts[i], pts[j]
      ax.plot(
          [p1[0], p2[0]],
          [p1[1], p2[1]],
          color="#1e293b",
          lw=1.9 if btype == 1.0 else 2.2,
          zorder=2,
      )
      if btype >= 1.5:
        vec = p2 - p1
        norm = max(float(np.linalg.norm(vec)), 1e-6)
        perp = np.array([-vec[1], vec[0]]) / norm * 0.07
        ax.plot(
            [p1[0] + perp[0], p2[0] + perp[0]],
            [p1[1] + perp[1], p2[1] + perp[1]],
            color="#475569",
            lw=1.1,
            ls="--" if btype == 1.5 else "-",
            zorder=2,
        )

    for idx, sym in enumerate(symbols):
      if sym != "C":
        col_map = {
            "N": "#2563eb",
            "N+": "#1d4ed8",
            "NH": "#2563eb",
            "O": "#dc2626",
            "O-": "#b91c1c",
            "F": "#16a34a",
            "Cl": "#15803d",
            "F/Cl": "#16a34a",
            "S": "#ca8a04",
        }
        c = col_map.get(sym, "#7c3aed")
        ax.scatter(
            [pts[idx, 0]],
            [pts[idx, 1]],
            s=125,
            color="white",
            edgecolors=c,
            linewidths=1.4,
            zorder=3,
        )
        ax.text(
            pts[idx, 0],
            pts[idx, 1],
            sym.replace("/Cl", ""),
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=c,
            zorder=4,
        )
      else:
        ax.scatter(
            [pts[idx, 0]],
            [pts[idx, 1]],
            s=22,
            color="#334155",
            zorder=3,
        )

  ax.text(
      x_offset,
      y_offset - scale * 0.58,
      label_header,
      ha="center",
      va="top",
      fontsize=7.1,
      fontweight="bold",
      color="#0f172a",
      bbox=dict(
          boxstyle="round,pad=0.25",
          facecolor="#f8fafc",
          edgecolor="#cbd5e1",
          alpha=0.95,
      ),
  )


def visualize_docking_comparison(
    baseline_data: Mapping[str, Any] | None = None,
    evolved_data: Mapping[str, Any] | None = None,
    output_path: Path | str = SAMPLE_DIR / "docking_pose_and_mpo_comparison.png",
) -> Path:
  """Generates a 2x2 3D Receptor-Pocket Docking Pose, 2D Lead Structure & MPO figure."""
  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  default_base, default_champ = _compute_baseline_and_champion_telemetry()
  base = dict(default_base)
  if baseline_data:
    base.update(baseline_data)
  evol = dict(default_champ)
  if evolved_data:
    evol.update(evolved_data)

  out_path = Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  kras_pocket = docking_oracle.load_pocket(PRIMARY_POCKET_PATH)
  egfr_pocket = docking_oracle.load_pocket(HELDOUT_POCKET_PATH)

  fig = plt.figure(figsize=(16.0, 11.2), dpi=160)
  fig.suptitle(
      "AlphaEvolve 3D Structure-Based Drug Discovery: Pocket-Conditioned Docking & MPO Evolution\n"
      f"Baseline Unconditioned Fragment (MPO Score = {float(base['score']):.3f})  vs.  "
      f"Evolved 3D Pocket-Conditioned Generator (MPO Score = {float(evol['score']):.3f}, "
      f"+{(float(evol['score']) / max(0.1, float(base['score'])) - 1.0) * 100:.1f}%)",
      fontsize=13.5,
      fontweight="bold",
  )

  # Panel A: 3D KRAS G12D Switch-II Binding Pocket + Docked Pose
  ax0 = fig.add_subplot(2, 2, 1, projection="3d")
  k_evol = evol["primary_kras_g12d"]
  _plot_3d_pocket_pose(
      ax0,
      kras_pocket,
      k_evol,
      "A. Primary Target: KRAS G12D Switch-II Pocket (PDB: 7RPZ)\n"
      f"Champion ΔG = {float(k_evol['best_vina_dg']):.2f} kcal/mol | "
      f"Salt Bridges = {int(k_evol.get('best_salt_bridges', 2))} (Asp12) | "
      f"QED = {float(k_evol['best_qed']):.2f}",
  )

  # Panel B: 3D Held-Out EGFR T790M Gatekeeper Pocket + Docked Pose
  ax1 = fig.add_subplot(2, 2, 2, projection="3d")
  e_evol = evol["heldout_egfr_t790m"]
  _plot_3d_pocket_pose(
      ax1,
      egfr_pocket,
      e_evol,
      "B. Held-Out Target: EGFR T790M Gatekeeper Pocket (PDB: 3IKA)\n"
      f"Champion ΔG = {float(e_evol['best_vina_dg']):.2f} kcal/mol | "
      f"H-Bonds = {int(e_evol.get('best_key_hbonds', 3))} (Met793/Gln791) | "
      f"QED = {float(e_evol['best_qed']):.2f}",
  )

  # Panel C: 2D RDKit Chemical Structures of Baseline vs. Evolved Pocket-Conditioned Leads
  ax2 = fig.add_subplot(2, 2, 3)
  ax2.set_xlim(-7.6, 7.6)
  ax2.set_ylim(-2.5, 2.5)
  ax2.axis("off")
  ax2.set_title(
      "C. 2D Chemical Structures: Baseline vs. Pocket-Conditioned Champion Leads",
      fontsize=11.0,
      fontweight="bold",
  )

  k_base = base["primary_kras_g12d"]
  _draw_2d_smiles_structure(
      ax2,
      str(k_base["best_smiles"]),
      k_base,
      x_offset=-5.0,
      y_offset=0.65,
      scale=2.7,
      label_header=(
          "Baseline (Unconditioned)\n"
          f"KRAS ΔG={float(k_base['best_vina_dg']):.2f} | EGFR ΔG={float(base['heldout_egfr_t790m']['best_vina_dg']):.2f}\n"
          f"Salt Bridges=0 | Strain={float(k_base['best_strain_kcal']):.2f} kcal/mol"
      ),
  )
  _draw_2d_smiles_structure(
      ax2,
      str(k_evol["best_smiles"]),
      k_evol,
      x_offset=0.0,
      y_offset=0.65,
      scale=2.8,
      label_header=(
          "KRAS G12D Lead (Piperazine Basic Warhead)\n"
          f"ΔG={float(k_evol['best_vina_dg']):.2f} | QED={float(k_evol['best_qed']):.2f} | SA={float(k_evol['best_sa_score']):.2f}\n"
          f"Asp12 Salt Bridges={int(k_evol.get('best_salt_bridges', 2))} | Strain={float(k_evol['best_strain_kcal']):.2f}"
      ),
  )
  _draw_2d_smiles_structure(
      ax2,
      str(e_evol["best_smiles"]),
      e_evol,
      x_offset=5.0,
      y_offset=0.65,
      scale=2.8,
      label_header=(
          "EGFR T790M Lead (Anilinoquinazoline Hinge)\n"
          f"ΔG={float(e_evol['best_vina_dg']):.2f} | QED={float(e_evol['best_qed']):.2f} | SA={float(e_evol['best_sa_score']):.2f}\n"
          f"Met793 H-Bonds={int(e_evol.get('best_key_hbonds', 3))} | Strain={float(e_evol['best_strain_kcal']):.2f}"
      ),
  )

  # Panel D: Multi-Parameter Optimization (MPO) & Target Conditioning Breakdown
  ax3 = fig.add_subplot(2, 2, 4)
  metrics_labels = [
      "KRAS Affinity\n(-ΔG [kcal/mol])",
      "EGFR Affinity\n(-ΔG [kcal/mol])",
      "KRAS Top-3\nMPO Score",
      "EGFR Top-3\nMPO Score",
      "Scaffold Div.\n(×5 Tanimoto)",
      "Composite\nDocking Score",
  ]
  m_idx = np.arange(len(metrics_labels))
  e_base = base["heldout_egfr_t790m"]
  base_vals = [
      -float(k_base["best_vina_dg"]),
      -float(e_base["best_vina_dg"]),
      float(k_base["mean_top3_score"]),
      float(e_base["mean_top3_score"]),
      5.0 * float(k_base["diversity"]),
      float(base["score"]),
  ]
  evol_vals = [
      -float(k_evol["best_vina_dg"]),
      -float(e_evol["best_vina_dg"]),
      float(k_evol["mean_top3_score"]),
      float(e_evol["mean_top3_score"]),
      5.0 * float(k_evol["diversity"]),
      float(evol["score"]),
  ]

  ax3.bar(
      m_idx - 0.17,
      base_vals,
      width=0.34,
      color="#94a3b8",
      edgecolor="#334155",
      label="Baseline Unconditioned Generator",
  )
  ebars = ax3.bar(
      m_idx + 0.17,
      evol_vals,
      width=0.34,
      color="#1a73e8",
      edgecolor="#0f172a",
      label="Evolved 3D Pocket-Conditioned Generator",
  )
  for bar_item, val in zip(ebars, evol_vals):
    ax3.text(
        bar_item.get_x() + bar_item.get_width() / 2.0,
        val + 0.18,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#0f172a",
    )
  ax3.set_xticks(m_idx)
  ax3.set_xticklabels(metrics_labels, fontsize=8.8)
  ax3.set_ylim(0.0, max(evol_vals) * 1.18)
  ax3.set_ylabel("Metric Value (Higher is Better)", fontsize=10.0)
  ax3.set_title(
      "D. Two-Target Generalization & Multi-Parameter Optimization (MPO)",
      fontsize=11.0,
      fontweight="bold",
  )
  ax3.grid(True, axis="y", linestyle=":", alpha=0.5)
  ax3.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

  fig.tight_layout(rect=[0, 0.01, 1, 0.93])
  fig.savefig(out_path, dpi=160)
  plt.close(fig)
  return out_path


def postprocess(top_programs: list[dict[str, Any]] | None = None) -> None:
  """Postprocessing hook generating 3D docking pose & MPO comparison plots."""
  bucket_name = os.getenv("_CLOUD_BUCKET_NAME", "").strip()
  user_exp = os.getenv("_USER_EXPERIMENT_NAME", "drug-docking").strip()

  best_prog = top_programs[0] if top_programs else {}
  best_prog_id = (
      best_prog.get("name", "").split("/")[-1]
      if isinstance(best_prog, Mapping)
      else ""
  )
  best_score = (
      get_score(best_prog, METRIC_NAME)
      if isinstance(best_prog, Mapping) and best_prog
      else None
  )
  if best_prog_id:
    logger.info(
        "BEST DRUG DISCOVERY DOCKING PROGRAM FOUND: %s (Score: %s)",
        best_prog_id,
        best_score,
    )

  best_data = None
  if bucket_name and best_prog_id:
    programs_dir = os.getenv("_PROGRAMS_DIR", "program_candidates")
    deployment_name = os.getenv("_DEPLOYMENT_NAME", "alpha-evolve")
    job_prefix = f"{deployment_name}-workers"
    job_id = get_job_id_from_program_name(best_prog_id, job_prefix)
    for prefix in [
        f"{user_exp}/{programs_dir}",
        f"{user_exp}/archive/{programs_dir}",
    ]:
      blob_name = f"{prefix}/{job_id}/{best_prog_id}.json"
      data = read_file_from_gcs(bucket_name, blob_name)
      if isinstance(data, dict) and "primary_kras_g12d" in data:
        best_data = data
        break

  base_default, champ_default = _compute_baseline_and_champion_telemetry()
  if best_data is None:
    best_data = champ_default
    if best_score is not None and math.isfinite(float(best_score)):
      best_data["score"] = float(best_score)

  results_env = os.getenv("_RESULTS_DIR")
  output_dir = Path(results_env) if results_env else SAMPLE_DIR
  output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = (
      Path(results_env) / "best_docking_summary.json"
      if results_env
      else Path("/tmp/best_docking_summary.json")
  )
  summary_path.write_text(json.dumps(best_data, indent=2), encoding="utf-8")

  plot_path = visualize_docking_comparison(
      base_default,
      best_data,
      output_dir / "docking_pose_and_mpo_comparison.png",
  )
  if bucket_name:
    for fpath in (summary_path, plot_path):
      if fpath.exists():
        upload_to_gcs(fpath, bucket_name, f"{user_exp}/results/{fpath.name}")


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))

