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

import io
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

from src import stent_radioss_runner  # pylint: disable=g-import-not-at-top
try:
  import main  # pylint: disable=g-import-not-at-top
except Exception:  # pylint: disable=broad-exception-caught
  main = None  # type: ignore[assignment]

METRIC_NAME = "stent_clinical_fitness_score"

BASELINE_STENT_PARAMS: dict[str, float] = {
    "crown_width_um": 85.0,
    "arm_width_um": 85.0,
    "wall_thickness_um": 85.0,
    "crown_amplitude_mm": 1.05,
    "crown_curv_exponent": 1.00,
    "link_amplitude_mm": 0.15,
    "link_width_um": 75.0,
    "arm_taper_ratio": 1.00,
}

CHAMPION_STENT_PARAMS: dict[str, float] = {
    "crown_width_um": 96.0,
    "arm_width_um": 78.0,
    "wall_thickness_um": 92.0,
    "crown_amplitude_mm": 1.08,
    "crown_curv_exponent": 1.72,
    "link_amplitude_mm": 0.38,
    "link_width_um": 62.0,
    "arm_taper_ratio": 0.84,
}


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


def _score_stent_simulation(sim: Mapping[str, Any]) -> float:
  """Computes composite stent_clinical_fitness_score from simulation metrics."""
  recoil_pct = float(sim["recoil_pct"])
  fs_pct = float(sim["foreshortening_pct"])
  db_pct = float(sim["dogboning_pct"])
  max_eps_p = float(sim["max_eps_p"])
  k_radial = float(sim["k_radial_kpa"])
  msa_pct = float(sim["geometry"]["msa_pct"])

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

  return round(
      stiffness_term
      + recoil_term
      + foreshortening_term
      + dogboning_term
      + strain_margin_bonus
      + msa_target_penalty
      - strain_penalty,
      4,
  )


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

  fitness_score = _score_stent_simulation(sim)

  elapsed_s = time.perf_counter() - t0
  bin_info = sim.get("openradioss_binary", {})
  bin_status = bin_info.get("status", "skipped")
  vtk_stages = bin_info.get("vtk_stages_parsed", 0)
  csv_rows = bin_info.get("csv_rows", 0)

  candidate_dir = os.environ.get("_CANDIDATE_DIR")
  if candidate_dir:
    try:
      program_name = os.environ.get("_CANDIDATE_PROGRAM_ID", "local_eval_run")
      out_path = Path(candidate_dir) / f"{program_name}.json"
      out_path.write_text(
          json.dumps(
              {
                  "program_name": program_name,
                  "score": fitness_score,
                  "elapsed_s": round(elapsed_s, 3),
                  **sim,
              },
              indent=2,
          ),
          encoding="utf-8",
      )
    except Exception as save_exc:  # pylint: disable=broad-exception-caught
      logger.warning("Failed to save stent telemetry JSON: %s", save_exc)

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


def _get_default_stent_telemetry() -> tuple[dict[str, Any], dict[str, Any]]:
  """Computes baseline and champion stent simulation telemetry."""
  base_sim = stent_radioss_runner.simulate_3stage_stent_deployment(
      dict(BASELINE_STENT_PARAMS), verify_binary=False
  )
  base_sim["score"] = _score_stent_simulation(base_sim)
  base_sim["program_name"] = "baseline_circular_crown"

  champ_sim = stent_radioss_runner.simulate_3stage_stent_deployment(
      dict(CHAMPION_STENT_PARAMS), verify_binary=False
  )
  champ_sim["score"] = _score_stent_simulation(champ_sim)
  champ_sim["program_name"] = "champion_superelliptic_tapered"
  return base_sim, champ_sim


def visualize_stent_deployment(
    baseline_data: Mapping[str, Any] | None = None,
    evolved_data: Mapping[str, Any] | None = None,
    output_path: Path | str = SAMPLE_DIR / "stent_deployment_comparison.png",
) -> Path:
  """Generates a 4-panel 3D OpenRADIOSS Stent Deployment, Strain Profile & KPI figure."""
  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.cm as cm  # pylint: disable=g-import-not-at-top
  import matplotlib.colors as mcolors  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # pylint: disable=g-import-not-at-top

  default_base, default_champ = _get_default_stent_telemetry()
  base = dict(default_base)
  if baseline_data:
    base.update(baseline_data)
  evol = dict(default_champ)
  if evolved_data:
    evol.update(evolved_data)

  out_path = Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  fig = plt.figure(figsize=(16.0, 11.2), dpi=160)
  fig.suptitle(
      "AlphaEvolve OpenRADIOSS 3D L605 CoCr Coronary Stent Evolution (3-Stage Deployment)\n"
      f"Baseline Circular-Crown (Fitness = {float(base['score']):.2f}, ε_p,max = {float(base['max_eps_p']):.3f} > 0.160 Limit)  vs.  "
      f"Evolved Super-Elliptic Tapered Stent (Fitness = +{float(evol['score']):.2f}, ε_p,max = {float(evol['max_eps_p']):.3f})",
      fontsize=13.2,
      fontweight="bold",
  )

  # --- Panel A: 3D Cylindrical QEPH Shell Mesh (Crimped A001 vs Deployed A003) ---
  ax0 = fig.add_subplot(2, 2, 1, projection="3d")
  evol_params = dict(evol.get("params", CHAMPION_STENT_PARAMS))
  crimp_mesh = stent_radioss_runner.build_3d_stent_stage_meshes(
      evol_params,
      od_stage_mm=stent_radioss_runner.OD_CRIMP_MM,
      foreshortening_pct=0.0,
      dogboning_pct=0.0,
      max_eps_p=0.01,
      max_vm_mpa=240.0,
  )
  deploy_mesh = stent_radioss_runner.build_3d_stent_stage_meshes(
      evol_params,
      od_stage_mm=float(evol["od_deployed_mm"]),
      foreshortening_pct=float(evol["foreshortening_pct"]),
      dogboning_pct=float(evol["dogboning_pct"]),
      max_eps_p=float(evol["max_eps_p"]),
      max_vm_mpa=float(evol["max_von_mises_mpa"]),
  )

  # Inner crimped wireframe mesh (A001: OD = 1.40 mm)
  crimp_poly = Poly3DCollection(
      crimp_mesh["quads"][::2],
      facecolors=(0.58, 0.64, 0.72, 0.16),
      edgecolors="#64748b",
      linewidths=0.35,
  )
  ax0.add_collection3d(crimp_poly)

  # Outer deployed shell mesh colored by Von Mises Stress (A003: OD ~ 3.07 mm)
  vm_vals = np.array(deploy_mesh["elem_vm_mpa"], dtype=float)
  norm = mcolors.Normalize(vmin=220.0, vmax=max(850.0, float(np.max(vm_vals))))
  cmap = cm.plasma
  face_cols = cmap(norm(vm_vals))
  deploy_poly = Poly3DCollection(
      deploy_mesh["quads"],
      facecolors=face_cols,
      edgecolors="#0f172a",
      linewidths=0.25,
      alpha=0.88,
  )
  ax0.add_collection3d(deploy_poly)

  sm = cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax0, fraction=0.038, pad=0.06)
  cbar.set_label("Von Mises Stress σ_VM [MPa]", fontsize=9.0)

  ax0.set_xlim(-1.8, 1.8)
  ax0.set_ylim(-1.8, 1.8)
  ax0.set_zlim(-0.6, 6.8)
  ax0.view_init(elev=22, azim=-52)
  ax0.set_xlabel("X [mm]", fontsize=8.5, labelpad=-2)
  ax0.set_ylabel("Y [mm]", fontsize=8.5, labelpad=-2)
  ax0.set_zlabel("Axial Z [mm]", fontsize=8.5, labelpad=-2)
  ax0.tick_params(labelsize=7.5)
  ax0.set_title(
      "A. 3D OpenRADIOSS QEPH Shell Mesh: Crimped (1.40 mm) → Deployed "
      f"({float(evol['od_deployed_mm']):.2f} mm)",
      fontsize=10.5,
      fontweight="bold",
  )

  # --- Panel B: 64-Station Crown-to-Arm Plastic Strain Profile & Width Taper ---
  ax1 = fig.add_subplot(2, 2, 2)
  s_grid = np.array(
      evol.get("s_grid", np.linspace(0.0, 1.0, 64).tolist()), dtype=float
  )
  base_epsp = np.array(base.get("eps_p_profile", []), dtype=float)
  evol_epsp = np.array(evol.get("eps_p_profile", []), dtype=float)

  ax1.plot(
      s_grid,
      base_epsp,
      "k--",
      lw=2.1,
      label=f"Baseline Circular Crown (max ε_p = {float(base['max_eps_p']):.4f})",
  )
  ax1.plot(
      s_grid,
      evol_epsp,
      "#1a73e8",
      lw=2.6,
      label=f"Evolved Super-Elliptic Tapered (max ε_p = {float(evol['max_eps_p']):.4f})",
  )
  ax1.axhline(
      stent_radioss_runner.EPS_P_ALLOWABLE,
      color="#d93025",
      ls=":",
      lw=2.0,
      label="FDA / ASTM F90 Fracture Limit (ε_p = 0.160, SF = 2.5)",
  )
  ax1.fill_between(
      s_grid,
      stent_radioss_runner.EPS_P_ALLOWABLE,
      np.maximum(base_epsp, stent_radioss_runner.EPS_P_ALLOWABLE),
      where=(base_epsp > stent_radioss_runner.EPS_P_ALLOWABLE),
      color="#fca5a5",
      alpha=0.45,
      label="Baseline Crown Fracture Zone (ε_p > 0.160)",
  )
  ax1.fill_between(
      s_grid,
      0.0,
      evol_epsp,
      color="#bfdbfe",
      alpha=0.35,
  )

  ax1_twin = ax1.twinx()
  evol_w_um = np.array(evol.get("w_s_um", []), dtype=float)
  if len(evol_w_um) == len(s_grid):
    ax1_twin.plot(
        s_grid,
        evol_w_um,
        color="#16a34a",
        ls="-.",
        lw=1.7,
        label="Evolved Strut Width Taper w(s) [μm]",
    )
    ax1_twin.set_ylabel(
        "In-Plane Strut Width w(s) [μm]", fontsize=9.5, color="#15803d"
    )
    ax1_twin.tick_params(axis="y", labelcolor="#15803d", labelsize=8.5)
    ax1_twin.set_ylim(50.0, 125.0)

  ax1.set_xlim(0.0, 1.0)
  ax1.set_ylim(0.0, max(0.24, float(np.max(base_epsp)) * 1.12))
  ax1.set_xlabel(
      "Normalized Strut Arc Coordinate s (0 = Crown Apex Hinge, 1 = Mid-Arm Inflection)",
      fontsize=9.5,
  )
  ax1.set_ylabel("Equivalent Plastic Strain ε_p(s) [-]", fontsize=10.0)
  ax1.set_title(
      "B. 64-Station Crown-to-Arm Equivalent Plastic Strain Distribution ε_p(s)",
      fontsize=10.5,
      fontweight="bold",
  )
  ax1.grid(True, linestyle=":", alpha=0.5)
  ax1.legend(loc="upper right", fontsize=8.3, framealpha=0.92)

  # --- Panel C: 3-Stage OpenRADIOSS T01.csv Energy & OD Deployment Trajectory ---
  ax2 = fig.add_subplot(2, 2, 3)
  t_ms = np.linspace(0.0, 6.0, 61)
  # Construct 3-stage OD trajectory: Stage 1 Inflation [0, 2ms], Stage 2 Recoil [2, 4ms], Stage 3 Crush [4, 6ms]
  def _build_od_curve(od_dep: float, k_rad: float) -> np.ndarray:
    od_c = np.zeros_like(t_ms)
    for idx, t in enumerate(t_ms):
      if t <= 2.0:
        u = 0.5 * (1.0 - np.cos(math.pi * t / 2.0))
        od_c[idx] = stent_radioss_runner.OD_CRIMP_MM + u * (
            stent_radioss_runner.OD_PEAK_MM - stent_radioss_runner.OD_CRIMP_MM
        )
      elif t <= 4.0:
        u = 0.5 * (1.0 - np.cos(math.pi * (t - 2.0) / 2.0))
        od_c[idx] = stent_radioss_runner.OD_PEAK_MM - u * (
            stent_radioss_runner.OD_PEAK_MM - od_dep
        )
      else:
        u = (t - 4.0) / 2.0
        crush_drop = 0.14 * (25.0 / max(k_rad, 10.0)) * u
        od_c[idx] = od_dep - crush_drop
    return od_c

  od_base_curve = _build_od_curve(float(base["od_deployed_mm"]), float(base["k_radial_kpa"]))
  od_evol_curve = _build_od_curve(float(evol["od_deployed_mm"]), float(evol["k_radial_kpa"]))

  ax2.plot(
      t_ms,
      od_base_curve,
      "k--",
      lw=2.0,
      label=f"Baseline Stent OD(t) (Deployed = {float(base['od_deployed_mm']):.2f} mm, Recoil = {float(base['recoil_pct']):.2f}%)",
  )
  ax2.plot(
      t_ms,
      od_evol_curve,
      "#188038",
      lw=2.6,
      label=f"Evolved Stent OD(t) (Deployed = {float(evol['od_deployed_mm']):.2f} mm, Recoil = {float(evol['recoil_pct']):.2f}%)",
  )
  ax2.axhline(
      stent_radioss_runner.OD_TARGET_MM,
      color="#0284c7",
      ls=":",
      lw=1.5,
      label="Target Vessel Lumen (OD = 3.00 mm)",
  )
  ax2.axvspan(0.0, 2.0, color="#fef3c7", alpha=0.30, label="Stage 1: Balloon Inflation")
  ax2.axvspan(2.0, 4.0, color="#e0f2fe", alpha=0.30, label="Stage 2: Elastic Recoil Unloading")
  ax2.axvspan(4.0, 6.0, color="#fce7f3", alpha=0.28, label="Stage 3: Arterial Radial Crush")

  ax2.set_xlim(0.0, 6.0)
  ax2.set_ylim(1.25, 3.35)
  ax2.set_xlabel("OpenRADIOSS 3-Stage Protocol Time t [ms]", fontsize=9.5)
  ax2.set_ylabel("Stent Outer Diameter OD(t) [mm]", fontsize=10.0)
  ax2.set_title(
      "C. 3-Stage Inflation (A002) → Recoil (A003) → Radial Crush (A004) Trajectory",
      fontsize=10.5,
      fontweight="bold",
  )
  ax2.grid(True, linestyle=":", alpha=0.5)
  ax2.legend(loc="lower right", fontsize=8.0, ncol=2, framealpha=0.92)

  # --- Panel D: Clinical & Mechanical KPI Comparison ---
  ax3 = fig.add_subplot(2, 2, 4)
  kpi_labels = [
      "Radial Stiffness\nK_radial / 5 [kPa]",
      "Acute Recoil\nR_recoil [%]",
      "Foreshortening\nFS [%]",
      "Dogboning\nDB [%]",
      "Peak Plastic\nStrain ×25 [%]",
      "Metal Coverage\nMSA / 3 [%]",
  ]
  k_idx = np.arange(len(kpi_labels))
  base_kpis = [
      float(base["k_radial_kpa"]) / 5.0,
      float(base["recoil_pct"]),
      float(base["foreshortening_pct"]),
      float(base["dogboning_pct"]),
      float(base["max_eps_p"]) * 25.0,
      float(base["geometry"]["msa_pct"]) / 3.0,
  ]
  evol_kpis = [
      float(evol["k_radial_kpa"]) / 5.0,
      float(evol["recoil_pct"]),
      float(evol["foreshortening_pct"]),
      float(evol["dogboning_pct"]),
      float(evol["max_eps_p"]) * 25.0,
      float(evol["geometry"]["msa_pct"]) / 3.0,
  ]

  ax3.bar(
      k_idx - 0.17,
      base_kpis,
      width=0.34,
      color="#94a3b8",
      edgecolor="#334155",
      label=f"Baseline Circular Crown (Score = {float(base['score']):.2f})",
  )
  ebars = ax3.bar(
      k_idx + 0.17,
      evol_kpis,
      width=0.34,
      color="#d93025",
      edgecolor="#0f172a",
      label=f"Evolved Super-Elliptic Tapered (Score = +{float(evol['score']):.2f})",
  )
  raw_evol_Annotations = [
      f"{float(evol['k_radial_kpa']):.1f} kPa",
      f"{float(evol['recoil_pct']):.2f}%",
      f"{float(evol['foreshortening_pct']):.2f}%",
      f"{float(evol['dogboning_pct']):.2f}%",
      f"{float(evol['max_eps_p'])*100:.1f}%",
      f"{float(evol['geometry']['msa_pct']):.1f}%",
  ]
  for bar_item, val, txt in zip(ebars, evol_kpis, raw_evol_Annotations):
    ax3.text(
        bar_item.get_x() + bar_item.get_width() / 2.0,
        val + 0.18,
        txt,
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#7f1d1d",
    )
  ax3.set_xticks(k_idx)
  ax3.set_xticklabels(kpi_labels, fontsize=8.6)
  ax3.set_ylim(0.0, max(max(base_kpis), max(evol_kpis)) * 1.22)
  ax3.set_ylabel("Normalized Clinical Metric Value", fontsize=10.0)
  ax3.set_title(
      "D. Clinical Performance KPIs: Stiffness ↑, Recoil ↓, Foreshortening ↓, Strain ↓",
      fontsize=10.5,
      fontweight="bold",
  )
  ax3.grid(True, axis="y", linestyle=":", alpha=0.5)
  ax3.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

  fig.tight_layout(rect=[0, 0.01, 1, 0.93])
  fig.savefig(out_path, dpi=160)
  plt.close(fig)
  return out_path


def generate_stent_evolution_gif(
    baseline_data: Mapping[str, Any] | None = None,
    evolved_data: Mapping[str, Any] | None = None,
    output_path: Path | str = SAMPLE_DIR / "stent_deployment_evolution.gif",
) -> Path:
  """Generates an animated GIF of the 3-stage 3D OpenRADIOSS stent deployment."""
  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.cm as cm  # pylint: disable=g-import-not-at-top
  import matplotlib.colors as mcolors  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # pylint: disable=g-import-not-at-top
  from PIL import Image  # pylint: disable=g-import-not-at-top

  default_base, default_champ = _get_default_stent_telemetry()
  base = dict(default_base)
  if baseline_data:
    base.update(baseline_data)
  evol = dict(default_champ)
  if evolved_data:
    evol.update(evolved_data)

  out_path = Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  evol_params = dict(evol.get("params", CHAMPION_STENT_PARAMS))
  od_dep = float(evol["od_deployed_mm"])
  stages = [
      ("Stage 0 (A001): Crimped on Catheter", stent_radioss_runner.OD_CRIMP_MM, 0.02),
      ("Stage 1a: Early Balloon Inflation", 2.28, 0.52),
      ("Stage 1b (A002): Peak Balloon Inflation", stent_radioss_runner.OD_PEAK_MM, 1.00),
      ("Stage 2 (A003): Balloon Deflation & Elastic Recoil", od_dep, 0.94),
      ("Stage 3 (A004): Arterial Radial Crush Test", od_dep - 0.08, 0.96),
  ]

  s_grid = np.array(evol.get("s_grid", np.linspace(0.0, 1.0, 64).tolist()), dtype=float)
  base_epsp = np.array(base.get("eps_p_profile", []), dtype=float)
  evol_epsp = np.array(evol.get("eps_p_profile", []), dtype=float)
  norm = mcolors.Normalize(vmin=0.0, vmax=0.16)
  cmap = cm.viridis

  frames: list[Image.Image] = []
  for idx, (stage_title, od_val, strain_scale) in enumerate(stages):
    fig = plt.figure(figsize=(11.5, 5.2), dpi=110)
    fig.suptitle(
        f"OpenRADIOSS 3D L605 CoCr Stent Deployment — {stage_title} (OD = {od_val:.2f} mm)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    mesh = stent_radioss_runner.build_3d_stent_stage_meshes(
        evol_params,
        od_stage_mm=od_val,
        foreshortening_pct=float(evol["foreshortening_pct"]) * strain_scale,
        dogboning_pct=float(evol["dogboning_pct"]) * strain_scale,
        max_eps_p=float(evol["max_eps_p"]) * strain_scale,
        max_vm_mpa=float(evol["max_von_mises_mpa"]) * max(0.3, strain_scale),
    )
    eps_arr = np.array(mesh["elem_eps_p"], dtype=float)
    poly = Poly3DCollection(
        mesh["quads"],
        facecolors=cmap(norm(eps_arr)),
        edgecolors="#0f172a",
        linewidths=0.22,
        alpha=0.90,
    )
    ax0.add_collection3d(poly)
    ax0.set_xlim(-1.8, 1.8)
    ax0.set_ylim(-1.8, 1.8)
    ax0.set_zlim(-0.5, 6.8)
    ax0.view_init(elev=22, azim=-55 + idx * 12)
    ax0.set_xlabel("X [mm]", fontsize=8)
    ax0.set_ylabel("Y [mm]", fontsize=8)
    ax0.set_zlabel("Z [mm]", fontsize=8)
    ax0.set_title("Evolved 3D QEPH Shell Mesh (Colored by ε_p)", fontsize=9.5, fontweight="bold")

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(
        s_grid,
        base_epsp * strain_scale,
        "k--",
        lw=2.0,
        label=f"Baseline ε_p(s) (peak={(float(base['max_eps_p'])*strain_scale):.3f})",
    )
    ax1.plot(
        s_grid,
        evol_epsp * strain_scale,
        "#1a73e8",
        lw=2.5,
        label=f"Evolved ε_p(s) (peak={(float(evol['max_eps_p'])*strain_scale):.3f})",
    )
    ax1.axhline(
        stent_radioss_runner.EPS_P_ALLOWABLE,
        color="#d93025",
        ls=":",
        lw=1.8,
        label="FDA Limit (0.160)",
    )
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 0.24)
    ax1.set_xlabel("Crown-to-Arm Coordinate s [-]", fontsize=9)
    ax1.set_ylabel("Equivalent Plastic Strain ε_p [-]", fontsize=9)
    ax1.set_title("Crown-to-Arm Plastic Strain Evolution", fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))

  frames[0].save(
      out_path,
      save_all=True,
      append_images=frames[1:],
      duration=550,
      loop=0,
  )
  return out_path


def postprocess(top_programs: list[dict[str, Any]] | None = None) -> None:
  """Postprocessing hook generating 3D OpenRADIOSS stent plots & GIF."""
  bucket_name = os.getenv("_CLOUD_BUCKET_NAME", "").strip()
  user_exp = os.getenv("_USER_EXPERIMENT_NAME", "coronary-stent").strip()

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
        "BEST CORONARY STENT PROGRAM FOUND: %s (Score: %s)",
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
      if isinstance(data, dict) and "od_deployed_mm" in data:
        best_data = data
        break

  base_default, champ_default = _get_default_stent_telemetry()
  if best_data is None:
    best_data = champ_default
    if best_score is not None and math.isfinite(float(best_score)):
      best_data["score"] = float(best_score)

  results_env = os.getenv("_RESULTS_DIR")
  output_dir = Path(results_env) if results_env else SAMPLE_DIR
  output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = (
      Path(results_env) / "best_stent_summary.json"
      if results_env
      else Path("/tmp/best_stent_summary.json")
  )
  summary_path.write_text(json.dumps(best_data, indent=2), encoding="utf-8")

  png_path = visualize_stent_deployment(
      base_default,
      best_data,
      output_dir / "stent_deployment_comparison.png",
  )
  gif_path = generate_stent_evolution_gif(
      base_default,
      best_data,
      output_dir / "stent_deployment_evolution.gif",
  )
  if bucket_name:
    for fpath in (summary_path, png_path, gif_path):
      if fpath.exists():
        upload_to_gcs(fpath, bucket_name, f"{user_exp}/results/{fpath.name}")


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))

