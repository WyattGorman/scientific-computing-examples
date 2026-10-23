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
"""Evaluator for 3D OpenFOAM PERRINN 2017 Formula 1 aerodynamic optimization."""

from __future__ import annotations

import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

try:
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
except ImportError:

  class _DictModel(dict):

    def __init__(self, **kwargs: Any):
      super().__init__(**kwargs)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
      del kwargs
      return dict(self)

  AlphaEvolveEvaluationInsight = _DictModel
  AlphaEvolveEvaluationInsights = _DictModel
  AlphaEvolveEvaluationScore = _DictModel
  AlphaEvolveEvaluationScores = _DictModel
  AlphaEvolveProgramEvaluation = _DictModel

  def get_job_id_from_program_name(
      program_name: str, prefix: str = ""
  ) -> str:
    del prefix
    return program_name.split("/")[-1]

  def read_file_from_gcs(bucket_name: str, blob_name: str) -> str | None:
    del bucket_name, blob_name
    return None

  def get_score(program: Mapping[str, Any], metric_name: str) -> float:
    scores = (
        program.get("evaluation", {}).get("scores", {}).get("scores", [])
    )
    for item in scores:
      if item.get("metric") == metric_name:
        return float(item.get("score", float("-inf")))
    return float("-inf")


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

_MODULE_DIR = Path(__file__).absolute().parent
SAMPLE_DIR = _MODULE_DIR
if str(_MODULE_DIR) in sys.path:
  sys.path.remove(str(_MODULE_DIR))
sys.path.insert(0, str(_MODULE_DIR))

from src.cfd import openfoam_f1_runner  # pylint: disable=g-import-not-at-top
from src.geometry.perrinn_f1_stl import build_perrinn_f1_stl  # pylint: disable=g-import-not-at-top

try:
  import main  # pylint: disable=g-import-not-at-top
except Exception:  # pylint: disable=broad-exception-caught
  main = None  # type: ignore[assignment]

METRIC_NAME = "f1_aero_lap_score"

BASELINE_F1_PARAMS: dict[str, float] = {
    "front_ride_height_mm": 40.0,
    "rear_ride_height_mm": 62.0,
    "diffuser_angle_deg": 10.0,
    "diffuser_strake_height_mm": 26.0,
    "fw_flap_angle_deg": 16.5,
    "fw_outwash_toe_deg": 2.0,
    "rw_aoa_deg": 21.5,
    "sidepod_undercut_ratio": 0.15,
}

CHAMPION_F1_PARAMS: dict[str, float] = {
    "front_ride_height_mm": 22.8,
    "rear_ride_height_mm": 76.5,
    "diffuser_angle_deg": 16.15,
    "diffuser_strake_height_mm": 52.0,
    "fw_flap_angle_deg": 22.0,
    "fw_outwash_toe_deg": 4.2,
    "rw_aoa_deg": 36.2,
    "sidepod_undercut_ratio": 0.32,
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


def evaluate(eval_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Evaluates a candidate PERRINN 2017 Formula 1 3D aerodynamic setup."""
  global main
  del eval_inputs
  t0 = time.perf_counter()

  spec = openfoam_f1_runner.load_f1_spec()
  design_bounds = {
      k: (float(v[0]), float(v[1])) for k, v in spec["design_bounds"].items()
  }

  try:
    if main is None:
      import importlib  # pylint: disable=g-import-not-at-top

      main = importlib.import_module("main")
    params = main.design_f1_aero_package(dict(design_bounds))
    if not isinstance(params, Mapping):
      return _build_eval(
          -100.0,
          "CANDIDATE_EXECUTION_ERROR",
          "design_f1_aero_package must return a dictionary of 8 parameters.",
      )
    sim = openfoam_f1_runner.evaluate_f1_aero_cfd(dict(params), spec=spec)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return _build_eval(
        -100.0,
        "CANDIDATE_EXECUTION_ERROR",
        f"Candidate execution failed: {exc}",
    )

  if not sim.get("valid", False):
    return _build_eval(
        -100.0,
        "FIA_LEGALITY_OR_CFD_VIOLATION",
        f"Invalid F1 aero setup: {sim.get('reason', 'invalid')}",
    )

  lap_score = float(sim["f1_aero_lap_score"])
  neg_scz = float(sim["neg_SCz_m2"])
  scx = float(sim["SCx_m2"])
  l_over_d = float(sim["L_over_D"])
  fb = float(sim["front_aero_balance"])
  n_cells = int(sim["n_cells"])
  comps = sim["components"]
  leg = sim["legality"]
  elapsed_s = time.perf_counter() - t0

  candidate_dir = os.environ.get("_CANDIDATE_DIR")
  if candidate_dir:
    try:
      program_name = os.environ.get("_CANDIDATE_PROGRAM_ID", "local_eval_run")
      payload = {
          "program_name": program_name,
          "score": lap_score,
          "elapsed_s": round(elapsed_s, 3),
          **sim,
      }
      cand_path = Path(candidate_dir)
      cand_path.mkdir(parents=True, exist_ok=True)
      serialized = json.dumps(payload, indent=2)
      (cand_path / f"{program_name}.json").write_text(
          serialized, encoding="utf-8"
      )
      (cand_path / "f1_aero_metrics.json").write_text(
          serialized, encoding="utf-8"
      )
    except Exception as save_exc:  # pylint: disable=broad-exception-caught
      logger.warning("Failed to save F1 CFD telemetry JSON: %s", save_exc)

  feedback = (
      f"PERRINN 2017 F1 3D Aero Evaluation ({sim['solver_backend']}, "
      f"{n_cells:,} hex cells, {elapsed_s:.2f}s) | "
      f"f1_aero_lap_score={lap_score:+.4f} | "
      f"Full-Car Downforce -SCz={neg_scz:.4f} m^2, Drag SCx={scx:.4f} m^2, "
      f"L/D={l_over_d:.4f} | "
      f"Front Aero Balance={fb * 100.0:.2f}% (Target 44.50% ± 2.80%) | "
      f"Patch -SCz [m^2]: Underbody={comps['underbody']['neg_SCz_m2']:+.3f}, "
      f"FrontWing={comps['frontWing']['neg_SCz_m2']:+.3f}, "
      f"RearWing={comps['rearWing']['neg_SCz_m2']:+.3f}, "
      f"Chassis={comps['chassis']['neg_SCz_m2']:+.3f}, "
      f"FrontWheel={comps['frontWheel']['neg_SCz_m2']:+.3f}, "
      f"RearWheel={comps['rearWheel']['neg_SCz_m2']:+.3f} | "
      f"FIA Clearances: Rake={leg['rake_angle_deg']:.2f} deg, "
      f"DiffuserBox={leg['diffuser_box_height_mm']:.1f}/175.0 mm, "
      f"StrakeClearance={leg['strake_clearance_mm']:.1f} mm, "
      f"SidepodInlet={leg['radiator_inlet_area_m2']:.3f} m^2"
  )
  return _build_eval(lap_score, "F1_OPENFOAM_3D_AERO_SUMMARY", feedback)


_HISTORY_CACHE: dict[str, Any] | None = None


def _load_history() -> dict[str, Any]:
  global _HISTORY_CACHE
  if _HISTORY_CACHE is None:
    _HISTORY_CACHE = json.loads(
        openfoam_f1_runner.HISTORY_PATH.read_text(encoding="utf-8")
    )
  return _HISTORY_CACHE


def _get_default_f1_telemetry() -> tuple[dict[str, Any], dict[str, Any]]:
  """Returns baseline (Gen 0) and discovered champion (Gen 12) CFD telemetry."""
  history = _load_history()
  ms0 = dict(history["generation_milestones"][0])
  ms12 = dict(history["generation_milestones"][-1])
  ms0["surface_polygons"] = history["surface_cp_meshes"]["baseline_gen0"]
  ms0["score"] = float(ms0["f1_aero_lap_score"])
  ms0["program_name"] = "gen00_seed_baseline"

  ms12["surface_polygons"] = history["surface_cp_meshes"]["champion_gen12"]
  ms12["score"] = float(ms12["f1_aero_lap_score"])
  ms12["program_name"] = "gen12_discovered_champion"
  return ms0, ms12


def _floor_and_diffuser_box_mm(
    params: Mapping[str, Any], x_grid_m: np.ndarray
) -> np.ndarray:
  """Computes ground clearance (throat) & FIA Art. 3.5.3 diffuser box height [mm]."""
  h_f = float(params.get("front_ride_height_mm", 30.0))
  h_r = float(params.get("rear_ride_height_mm", 70.0))
  diff_deg = float(params.get("diffuser_angle_deg", 14.0))
  fw_flap = float(params.get("fw_flap_angle_deg", 18.0))

  z_mm = np.full_like(x_grid_m, np.nan, dtype=float)
  for i, x in enumerate(x_grid_m):
    if -0.92 <= x <= -0.32:
      u = (x + 0.92) / 0.60
      z_mm[i] = h_f + 28.0 + 44.0 * u * (fw_flap / 20.0)
    elif 0.35 <= x <= 3.25:
      u = (x - 0.35) / (3.25 - 0.35)
      z_mm[i] = h_f + u * (h_r - h_f)
    elif 3.25 < x <= 3.775:
      # FIA Art. 3.5.3 reference-plane diffuser box ramp (capped at 175.0 mm)
      u = (x - 3.25) / 0.525
      diff_exit_mm = 20.0 + 525.0 * math.tan(math.radians(diff_deg))
      z_mm[i] = h_r + u * (diff_exit_mm - h_r)
  return z_mm


def _build_continuous_full_car_stl_cp(
    params: Mapping[str, Any],
    surface_polys: list[dict[str, Any]],
    components: Mapping[str, Any],
    y_offset: float = 0.0,
    z_floor_projection: float | None = None,
    tri_stride: int = 1,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
  """Builds continuous watertight 3D PERRINN F1 STL colored by OpenFOAM Cp field."""
  clean_p = {
      k: float(params.get(k, CHAMPION_F1_PARAMS[k]))
      for k in CHAMPION_F1_PARAMS
  }
  with tempfile.TemporaryDirectory(prefix="f1_viz_stl_") as tmpd:
    stl_path = Path(tmpd) / "car.stl"
    build_perrinn_f1_stl(clean_p, stl_path)
    stl_txt = stl_path.read_text(encoding="utf-8")

  patch_samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  by_p: dict[str, list[tuple[list[float], float]]] = {}
  for item in surface_polys:
    pname = str(item.get("p", "chassis"))
    v = np.array(item["v"], dtype=float)
    if v.shape[0] >= 3:
      by_p.setdefault(pname, []).append(
          (v.mean(axis=0).tolist(), float(item["cp"]))
      )
  for pname, rows in by_p.items():
    pts_arr = np.array([r[0] for r in rows], dtype=float)
    cp_arr = np.array([r[1] for r in rows], dtype=float)
    patch_samples[pname] = (pts_arr, cp_arr)

  car_verts: list[np.ndarray] = []
  car_cps: list[float] = []
  proj_verts: list[np.ndarray] = []
  proj_cps: list[float] = []

  for block in stl_txt.split("solid ")[1:]:
    header, _, body = block.partition("\n")
    curr_solid = header.strip()
    v_tokens = [
        line.strip()[7:]
        for line in body.splitlines()
        if line.lstrip().startswith("vertex ")
    ]
    if not v_tokens:
      continue
    coords = np.fromstring(" ".join(v_tokens), sep=" ", dtype=float).reshape(
        -1, 3, 3
    )
    if tri_stride > 1 and len(coords) > 24:
      coords = coords[::tri_stride]

    valid_y = np.max(coords[:, :, 1], axis=1) > 1e-4
    coords = coords[valid_y]
    if len(coords) == 0:
      continue
    coords[:, :, 1] = np.maximum(0.0, coords[:, :, 1])
    coords[:, :, 2] = np.maximum(0.0, coords[:, :, 2])

    n_vecs = np.cross(
        coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0]
    )
    n_norms = np.linalg.norm(n_vecs, axis=1)
    good = n_norms > 1e-10
    coords = coords[good]
    n_hats = n_vecs[good] / n_norms[good, None]
    if len(coords) == 0:
      continue
    centroids = coords.mean(axis=1)

    comp_scz = float(components.get(curr_solid, {}).get("neg_SCz_m2", 0.0))
    if curr_solid in patch_samples:
      s_pts, s_cps = patch_samples[curr_solid]
      d2 = np.sum(
          (centroids[:, None, :] - s_pts[None, :, :]) ** 2, axis=2
      )
      w = 1.0 / (d2 + 0.008)
      cp_idw = np.sum(w * s_cps[None, :], axis=1) / np.sum(w, axis=1)
    else:
      cp_idw = np.full(len(coords), -0.35 * comp_scz)

    nz = n_hats[:, 2]
    nx = n_hats[:, 0]
    if curr_solid in ("underbody", "frontWing", "rearWing"):
      cp_vals = np.where(
          nz < -0.25,
          np.minimum(cp_idw, -0.35 - 0.72 * max(comp_scz, 0.1)),
          np.where(
              nz > 0.25,
              np.maximum(cp_idw * 0.45, 0.08 + 0.12 * max(comp_scz, 0.0)),
              cp_idw,
          ),
      )
    elif "Wheel" in curr_solid:
      cp_vals = 0.55 * np.maximum(0.0, -nx) - 0.45 * np.abs(nz) + 0.35 * cp_idw
    else:
      cp_vals = (
          0.65 * cp_idw
          + 0.35 * np.maximum(0.0, -nx)
          - 0.18 * np.maximum(0.0, nz)
      )
    cp_vals = np.clip(cp_vals, -1.60, 0.75)

    pos_tris = coords.copy()
    pos_tris[:, :, 1] += y_offset
    neg_tris = coords[:, ::-1, :].copy()
    neg_tris[:, :, 1] = -neg_tris[:, :, 1] + y_offset

    car_verts.extend(pos_tris)
    car_verts.extend(neg_tris)
    car_cps.extend(cp_vals.tolist())
    car_cps.extend(cp_vals.tolist())

    if z_floor_projection is not None and curr_solid in (
        "frontWing",
        "underbody",
        "rearWing",
    ):
      down_mask = nz < -0.25
      if np.any(down_mask):
        p_pos = pos_tris[down_mask].copy()
        p_pos[:, :, 2] = z_floor_projection
        p_neg = neg_tris[down_mask].copy()
        p_neg[:, :, 2] = z_floor_projection
        p_c = cp_vals[down_mask].tolist()
        proj_verts.extend(p_pos)
        proj_verts.extend(p_neg)
        proj_cps.extend(p_c)
        proj_cps.extend(p_c)

  return (
      car_verts,
      np.array(car_cps, dtype=float),
      proj_verts,
      np.array(proj_cps, dtype=float),
  )


def visualize_f1_aero_comparison(
    baseline_data: Mapping[str, Any] | None = None,
    evolved_data: Mapping[str, Any] | None = None,
    output_path: Path | str | None = None,
) -> Path:
  """Generates a 4-panel 3D OpenFOAM PERRINN 2017 Formula 1 aero comparison PNG."""
  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.cm as cm  # pylint: disable=g-import-not-at-top
  import matplotlib.colors as mcolors  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # pylint: disable=g-import-not-at-top

  default_base, default_champ = _get_default_f1_telemetry()
  base = dict(default_base)
  if baseline_data:
    base.update(baseline_data)
  evol = dict(default_champ)
  if evolved_data:
    evol.update(evolved_data)

  if output_path is None:
    output_path = SAMPLE_DIR / "f1_aero_comparison.png"
  out_path = Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  is_unit_test = bool(
      os.environ.get("TEST_TMPDIR") or os.environ.get("PYTEST_CURRENT_TEST")
  )
  png_dpi = 72 if is_unit_test else 160
  tri_stride = 8 if is_unit_test else 1

  history = _load_history()
  milestones = history["generation_milestones"]
  all_cands = history["all_evaluated_candidates"]

  base_score = float(base.get("score", base.get("f1_aero_lap_score", -5.7525)))
  evol_score = float(evol.get("score", evol.get("f1_aero_lap_score", 1.8704)))
  base_scz = float(base["neg_SCz_m2"])
  evol_scz = float(evol["neg_SCz_m2"])
  base_ld = float(base["L_over_D"])
  evol_ld = float(evol["L_over_D"])
  base_fb = float(base["front_aero_balance"]) * 100.0
  evol_fb = float(evol["front_aero_balance"]) * 100.0

  fig = plt.figure(figsize=(16.8, 11.5), dpi=png_dpi)
  fig.patch.set_facecolor("#f8fafc")
  fig.suptitle(
      "AlphaEvolve × OpenFOAM 3D Spalart-Allmaras RANS: PERRINN 2017 Formula 1 Aerodynamic Evolution\n"
      f"Gen 0 Seed (-SCz = {base_scz:.2f} m², L/D = {base_ld:.2f}, Front Bal = {base_fb:.1f}%, Score = {base_score:+.2f})  →  "
      f"Gen 12 Champion (-SCz = {evol_scz:.2f} m² [+{(evol_scz / max(base_scz, 1e-6) - 1.0) * 100.0:.0f}%], "
      f"L/D = {evol_ld:.2f}, Front Bal = {evol_fb:.1f}%, Score = {evol_score:+.2f})",
      fontsize=11.3,
      fontweight="bold",
      color="#0f172a",
  )

  # --- Panel A: Continuous 3D Full-Car PERRINN F1 STL + OpenFOAM Cp + Floor Mirror ---
  ax0 = fig.add_subplot(2, 2, 1, projection="3d")
  ax0.set_facecolor("#f8fafc")
  cmap_cp = cm.RdYlBu_r
  norm_cp = mcolors.Normalize(vmin=-1.55, vmax=0.75)

  base_polys = list(
      base.get("surface_polygons") or default_base["surface_polygons"]
  )
  evol_polys = list(
      evol.get("surface_polygons") or default_champ["surface_polygons"]
  )

  z_mirror = -0.52
  b_car, b_cp, b_proj, b_pcp = _build_continuous_full_car_stl_cp(
      base.get("parameters", BASELINE_F1_PARAMS),
      base_polys,
      base["components"],
      y_offset=1.32,
      z_floor_projection=z_mirror,
      tri_stride=tri_stride,
  )
  e_car, e_cp, e_proj, e_pcp = _build_continuous_full_car_stl_cp(
      evol.get("parameters", CHAMPION_F1_PARAMS),
      evol_polys,
      evol["components"],
      y_offset=-1.32,
      z_floor_projection=z_mirror,
      tri_stride=tri_stride,
  )

  for p_verts, p_vals in ((b_proj, b_pcp), (e_proj, e_pcp)):
    if p_verts:
      shadow_coll = Poly3DCollection(
          p_verts,
          facecolors=cmap_cp(norm_cp(p_vals)),
          edgecolors="none",
          alpha=0.82,
      )
      ax0.add_collection3d(shadow_coll)

  for c_verts, c_vals, edge_col in (
      (b_car, b_cp, "#334155"),
      (e_car, e_cp, "#0f172a"),
  ):
    car_coll = Poly3DCollection(
        c_verts,
        facecolors=cmap_cp(norm_cp(c_vals)),
        edgecolors=edge_col,
        linewidths=0.14,
        alpha=0.96,
    )
    ax0.add_collection3d(car_coll)

  ax0.text2D(
      0.03,
      0.93,
      f"Upper Track — Gen 0 Seed: -SCz={base_scz:.2f} m², FB={base_fb:.1f}%",
      transform=ax0.transAxes,
      fontsize=7.9,
      fontweight="bold",
      color="#334155",
      va="top",
      ha="left",
      bbox=dict(
          boxstyle="round,pad=0.25",
          facecolor="#ffffff",
          edgecolor="#94a3b8",
          alpha=0.92,
      ),
  )
  ax0.text2D(
      0.97,
      0.05,
      f"Lower Track — Gen 12 Champion: -SCz={evol_scz:.2f} m², FB={evol_fb:.1f}%",
      transform=ax0.transAxes,
      fontsize=8.0,
      fontweight="bold",
      color="#1d4ed8",
      va="bottom",
      ha="right",
      bbox=dict(
          boxstyle="round,pad=0.25",
          facecolor="#eff6ff",
          edgecolor="#60a5fa",
          alpha=0.94,
      ),
  )

  sm = cm.ScalarMappable(cmap=cmap_cp, norm=norm_cp)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax0, fraction=0.036, pad=0.05, shrink=0.82)
  cbar.set_label(
      "OpenFOAM Surface Pressure Coeff. Cp [-]\n(Deep Blue = Ground-Effect Suction)",
      fontsize=8.3,
  )

  ax0.set_xlim(-1.05, 4.15)
  ax0.set_ylim(-2.48, 2.48)
  ax0.set_zlim(-0.56, 1.22)
  ax0.set_box_aspect((5.20, 4.96, 1.78))
  ax0.view_init(elev=23, azim=-134)
  ax0.set_xlabel("Streamwise X [m]", fontsize=8.3, labelpad=-2)
  ax0.set_ylabel("Spanwise Y [m]", fontsize=8.3, labelpad=-2)
  ax0.set_zlabel("Z [m] & Floor Suction Mirror", fontsize=8.3, labelpad=-2)
  ax0.tick_params(labelsize=7.2)
  ax0.set_title(
      "A. 3D Full-Car PERRINN 2017 F1 OpenFOAM Surface Cp & Underfloor Mirror",
      fontsize=10.1,
      fontweight="bold",
  )

  # --- Panel B: Streamwise Suction -Cp(x) & Raked Venturi Floor/Diffuser ---
  ax1 = fig.add_subplot(2, 2, 2)
  ax1.set_facecolor("#ffffff")
  cp_x = np.array(
      evol.get("cp_x_m", default_champ["cp_x_m"]), dtype=float
  )
  base_neg_cp = -np.array(
      base.get("cp_suction", default_base["cp_suction"]), dtype=float
  )
  evol_neg_cp = -np.array(
      evol.get("cp_suction", default_champ["cp_suction"]), dtype=float
  )

  ax1.axvspan(-0.95, -0.28, color="#e0f2fe", alpha=0.38, label="2017 Delta Front Wing")
  ax1.axvspan(0.35, 3.25, color="#eef2ff", alpha=0.42, label="Venturi Underfloor Throat")
  ax1.axvspan(3.25, 3.92, color="#fef3c7", alpha=0.42, label="FIA 175 mm Diffuser & Rear Wing")

  ax1.plot(
      cp_x,
      base_neg_cp,
      "k--",
      lw=2.0,
      label=f"Gen 0 Seed Suction -Cp(x) (Floor -SCz = {base['components']['underbody']['neg_SCz_m2']:.2f} m²)",
  )
  ax1.plot(
      cp_x,
      evol_neg_cp,
      color="#1d4ed8",
      lw=2.6,
      label=f"Gen 12 Champion Suction -Cp(x) (Floor -SCz = {evol['components']['underbody']['neg_SCz_m2']:.2f} m², 5.5×)",
  )
  ax1.fill_between(
      cp_x,
      base_neg_cp,
      evol_neg_cp,
      where=(evol_neg_cp >= base_neg_cp),
      color="#93c5fd",
      alpha=0.38,
      label="Ground-Effect Suction Gain",
  )
  ax1.axhline(0.0, color="#64748b", lw=0.9, ls=":")

  ax1_twin = ax1.twinx()
  x_dense = np.linspace(-0.92, 3.775, 220)
  z_base_mm = _floor_and_diffuser_box_mm(
      base.get("parameters", BASELINE_F1_PARAMS), x_dense
  )
  z_evol_mm = _floor_and_diffuser_box_mm(
      evol.get("parameters", CHAMPION_F1_PARAMS), x_dense
  )
  ax1_twin.plot(
      x_dense,
      z_base_mm,
      color="#64748b",
      ls=":",
      lw=1.7,
      label="Gen 0 Floor / Diffuser Box Height [mm]",
  )
  ax1_twin.plot(
      x_dense,
      z_evol_mm,
      color="#15803d",
      ls="-.",
      lw=2.1,
      label="Gen 12 Raked Venturi + 16.15° Diffuser (172 mm)",
  )
  ax1_twin.axhline(
      175.0,
      color="#dc2626",
      ls="--",
      lw=1.4,
      label="FIA Art. 3.5.3 Diffuser Box Ceiling (175 mm)",
  )
  ax1_twin.set_ylabel(
      "Floor Clearance / FIA Diffuser Box Height [mm]",
      fontsize=9.0,
      color="#15803d",
  )
  ax1_twin.tick_params(axis="y", labelcolor="#15803d", labelsize=8.2)
  ax1_twin.set_ylim(0.0, 220.0)

  ax1.set_xlim(-0.98, 3.95)
  ax1.set_ylim(
      min(-0.32, float(np.min(base_neg_cp)) - 0.08),
      max(0.92, float(np.max(evol_neg_cp)) * 1.30),
  )
  ax1.set_xlabel("Streamwise Coordinate X from Front Axle [m]", fontsize=9.3)
  ax1.set_ylabel("Downward-Surface Suction  -Cp(x)  [-]", fontsize=9.6)
  ax1.set_title(
      "B. Streamwise Suction -Cp(x) & Raked Venturi Diffuser Geometry",
      fontsize=10.1,
      fontweight="bold",
  )
  ax1.grid(True, linestyle=":", alpha=0.5)
  h1, l1 = ax1.get_legend_handles_labels()
  h2, l2 = ax1_twin.get_legend_handles_labels()
  ax1.legend(
      h1[3:5] + h2[1:],
      l1[3:5] + l2[1:],
      loc="upper left",
      fontsize=7.5,
      framealpha=0.93,
  )

  # --- Panel C: 6-Patch OpenFOAM Force Breakdown & Pitch-Balance Window ---
  ax2 = fig.add_subplot(2, 2, 3)
  ax2.set_facecolor("#ffffff")
  patch_order = [
      ("underbody", "Venturi Underbody\n& 175mm Diffuser"),
      ("frontWing", "2017 Delta\nFront Wing"),
      ("rearWing", "Dual-Element\nRear Wing"),
      ("chassis", "Monocoque &\nUndercut Sidepods"),
      ("frontWheel", "Rotating Front\nPirelli Wheels"),
      ("rearWheel", "Rotating Rear\nPirelli Wheels"),
  ]
  y_pos = np.arange(len(patch_order))
  b_scz_list = [
      float(base["components"][k]["neg_SCz_m2"]) for k, _ in patch_order
  ]
  e_scz_list = [
      float(evol["components"][k]["neg_SCz_m2"]) for k, _ in patch_order
  ]
  e_scx_list = [
      float(evol["components"][k]["SCx_m2"]) for k, _ in patch_order
  ]

  bar_h = 0.36
  ax2.barh(
      y_pos + bar_h / 2.0,
      b_scz_list,
      height=bar_h,
      color="#94a3b8",
      edgecolor="#334155",
      label=f"Gen 0 Seed -SCz (Total = {base_scz:.2f} m², SCx = {float(base['SCx_m2']):.2f} m²)",
  )
  ax2.barh(
      y_pos - bar_h / 2.0,
      e_scz_list,
      height=bar_h,
      color="#2563eb",
      edgecolor="#0f172a",
      label=f"Gen 12 Champion -SCz (Total = {evol_scz:.2f} m², SCx = {float(evol['SCx_m2']):.2f} m²)",
  )
  ax2.axvline(0.0, color="#0f172a", lw=1.1)

  for idx, (bv, ev, ex) in enumerate(
      zip(b_scz_list, e_scz_list, e_scx_list)
  ):
    offset = 0.025 if ev >= 0.0 else -0.025
    align = "left" if ev >= 0.0 else "right"
    ax2.text(
        ev + offset,
        idx - bar_h / 2.0,
        f"{ev:+.2f} m² (SCx={ex:.2f})",
        va="center",
        ha=align,
        fontsize=7.6,
        fontweight="bold",
        color="#1e3a8a",
    )
    b_off = 0.025 if bv >= 0.0 else -0.025
    b_al = "left" if bv >= 0.0 else "right"
    ax2.text(
        bv + b_off,
        idx + bar_h / 2.0,
        f"{bv:+.2f} m²",
        va="center",
        ha=b_al,
        fontsize=7.3,
        color="#475569",
    )

  balance_callout = (
      "Longitudinal Pitch Stability (Target: 44.5% ± 2.8%)\n"
      f"• Gen 0 Seed:     FB = {base_fb:.1f}%  (Snap-Oversteer Penalty)\n"
      f"• Gen 12 Champ:  FB = {evol_fb:.1f}%  (Inside Neutral Window)"
  )
  ax2.text(
      0.98,
      0.04,
      balance_callout,
      transform=ax2.transAxes,
      ha="right",
      va="bottom",
      fontsize=7.9,
      bbox=dict(
          boxstyle="round,pad=0.38",
          facecolor="#ecfdf5",
          edgecolor="#16a34a",
          alpha=0.95,
      ),
  )

  ax2.set_yticks(y_pos)
  ax2.set_yticklabels([lbl for _, lbl in patch_order], fontsize=8.4)
  ax2.invert_yaxis()
  ax2.set_xlim(-0.82, 1.52)
  ax2.set_xlabel(
      "Component Vertical Downforce Area  -SCz  [m²]   (Positive = Downforce, Negative = Lift)",
      fontsize=9.1,
  )
  ax2.set_title(
      "C. 6-Patch OpenFOAM Force Breakdown & FIA Pitch-Balance Convergence",
      fontsize=10.1,
      fontweight="bold",
  )
  ax2.grid(True, axis="x", linestyle=":", alpha=0.5)
  ax2.legend(
      loc="center right",
      bbox_to_anchor=(0.98, 0.36),
      fontsize=7.7,
      framealpha=0.93,
  )

  # --- Panel D: 21-Run 3D OpenFOAM Evolutionary Frontier & L/D ---
  ax3 = fig.add_subplot(2, 2, 4)
  ax3.set_facecolor("#ffffff")
  cand_gens = [int(c["generation"]) for c in all_cands]
  cand_scores = [float(c["f1_aero_lap_score"]) for c in all_cands]

  ms_gens = [int(m["generation"]) for m in milestones]
  ms_scores = [float(m["f1_aero_lap_score"]) for m in milestones]
  ms_scz = [float(m["neg_SCz_m2"]) for m in milestones]
  ms_ld = [float(m["L_over_D"]) for m in milestones]

  ax3.scatter(
      cand_gens,
      cand_scores,
      color="#94a3b8",
      edgecolors="#475569",
      s=46,
      alpha=0.78,
      zorder=3,
      label=f"All {len(all_cands)} 3D OpenFOAM RANS Candidates",
  )
  ax3.plot(
      ms_gens,
      ms_scores,
      "o-",
      color="#1d4ed8",
      lw=2.6,
      ms=6.5,
      zorder=5,
      label="Best-So-Far f1_aero_lap_score Frontier",
  )
  ax3.axhline(0.0, color="#64748b", ls=":", lw=1.0)

  ax3_twin = ax3.twinx()
  ax3_twin.plot(
      ms_gens,
      ms_scz,
      "s--",
      color="#15803d",
      lw=2.0,
      ms=5.0,
      label="Full-Car Downforce Area -SCz [m²]",
  )
  ax3_twin.plot(
      ms_gens,
      ms_ld,
      "^:",
      color="#b45309",
      lw=1.9,
      ms=5.0,
      label="Full-Car Aero Efficiency L/D [-]",
  )
  ax3_twin.set_ylabel(
      "Downforce -SCz [m²]  /  Aero Efficiency L/D [-]",
      fontsize=9.1,
      color="#15803d",
  )
  ax3_twin.tick_params(axis="y", labelcolor="#15803d", labelsize=8.2)
  ax3_twin.set_ylim(0.35, 1.95)

  ax3.annotate(
      f"Gen 1: Rake Unlock\n({ms_scores[0]:+.2f} → {ms_scores[1]:+.2f})",
      xy=(1, ms_scores[1]),
      xytext=(1.8, -3.55),
      arrowprops=dict(arrowstyle="->", color="#334155", lw=1.1),
      fontsize=7.8,
      fontweight="bold",
  )
  ax3.annotate(
      f"Gen 12 Champion\nScore = {ms_scores[-1]:+.2f}\n-SCz = {ms_scz[-1]:.2f} m², L/D = {ms_ld[-1]:.2f}",
      xy=(12, ms_scores[-1]),
      xytext=(7.1, -1.65),
      arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.3),
      fontsize=8.0,
      fontweight="bold",
      color="#1e3a8a",
      bbox=dict(
          boxstyle="round,pad=0.3",
          facecolor="#eff6ff",
          edgecolor="#60a5fa",
      ),
  )

  ax3.set_xlim(-0.5, 12.6)
  ax3.set_ylim(-6.3, 2.7)
  ax3.set_xlabel("AlphaEvolve Generation", fontsize=9.3)
  ax3.set_ylabel(
      "Primary Fitness: f1_aero_lap_score", fontsize=9.6, color="#1d4ed8"
  )
  ax3.tick_params(axis="y", labelcolor="#1d4ed8")
  ax3.set_title(
      "D. 21-Candidate 3D OpenFOAM RANS Campaign (~57,000 Hex Cells / Run)",
      fontsize=10.1,
      fontweight="bold",
  )
  ax3.grid(True, linestyle=":", alpha=0.5)
  d1, dl1 = ax3.get_legend_handles_labels()
  d2, dl2 = ax3_twin.get_legend_handles_labels()
  ax3.legend(
      d1 + d2, dl1 + dl2, loc="lower right", fontsize=7.5, framealpha=0.92
  )

  fig.tight_layout(rect=[0, 0.01, 1, 0.92])
  fig.savefig(out_path, dpi=png_dpi)
  plt.close(fig)
  return out_path


def generate_f1_aero_evolution_gif(
    baseline_data: Mapping[str, Any] | None = None,
    evolved_data: Mapping[str, Any] | None = None,
    output_path: Path | str | None = None,
) -> Path:
  """Generates an animated multi-generation 3D OpenFOAM F1 evolution GIF."""
  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.cm as cm  # pylint: disable=g-import-not-at-top
  import matplotlib.colors as mcolors  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # pylint: disable=g-import-not-at-top
  from PIL import Image  # pylint: disable=g-import-not-at-top

  del baseline_data, evolved_data
  if output_path is None:
    output_path = SAMPLE_DIR / "f1_aero_evolution.gif"
  out_path = Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  is_unit_test = bool(
      os.environ.get("TEST_TMPDIR") or os.environ.get("PYTEST_CURRENT_TEST")
  )
  stride = 12 if is_unit_test else 1
  gif_dpi = 56 if is_unit_test else 96
  tri_stride = 8 if is_unit_test else 1

  history = _load_history()
  milestones = history["generation_milestones"]
  all_cands = history["all_evaluated_candidates"]
  meshes = history["surface_cp_meshes"]

  ms0 = milestones[0]
  ms_gens = [int(m["generation"]) for m in milestones]
  ms_scores = [float(m["f1_aero_lap_score"]) for m in milestones]
  ms_scz = [float(m["neg_SCz_m2"]) for m in milestones]
  ms_fb = [float(m["front_aero_balance"]) * 100.0 for m in milestones]
  ms_ld = [float(m["L_over_D"]) for m in milestones]
  cand_gens = [int(c["generation"]) for c in all_cands]
  cand_scores = [float(c["f1_aero_lap_score"]) for c in all_cands]

  cp_x = np.array(ms0["cp_x_m"], dtype=float)
  base_neg_cp = -np.array(ms0["cp_suction"], dtype=float)
  x_dense = np.linspace(-0.92, 3.775, 180)
  z_base_mm = _floor_and_diffuser_box_mm(ms0["parameters"], x_dense)

  cmap_cp = cm.RdYlBu_r
  norm_cp = mcolors.Normalize(vmin=-1.55, vmax=0.75)

  frames: list[Image.Image] = []
  durations: list[int] = []

  for idx, ms in enumerate(milestones):
    if idx % stride != 0 and idx != len(milestones) - 1:
      continue
    gen = int(ms["generation"])
    p = ms["parameters"]
    score_now = float(ms["f1_aero_lap_score"])
    scz_now = float(ms["neg_SCz_m2"])
    scx_now = float(ms["SCx_m2"])
    ld_now = float(ms["L_over_D"])
    fb_now = float(ms["front_aero_balance"]) * 100.0

    if gen == 0:
      raw_mesh = meshes["baseline_gen0"]
    elif gen < 8:
      raw_mesh = meshes["mid_gen4"]
    else:
      raw_mesh = meshes["champion_gen12"]

    fig = plt.figure(figsize=(13.8, 8.9), dpi=gif_dpi)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        f"AlphaEvolve × OpenFOAM 3D F1 Evolution — Gen {gen:02d}/12  "
        f"({ms['candidate_id']})\n"
        f"-SCz = {scz_now:.3f} m²  |  SCx = {scx_now:.3f} m²  |  "
        f"L/D = {ld_now:.3f}  |  Front Balance = {fb_now:.1f}%  |  "
        f"Lap Score = {score_now:+.3f}",
        fontsize=11.4,
        fontweight="bold",
        color="#0f172a",
    )

    # Panel A: Rotating Continuous 3D Full-Car PERRINN F1 STL + OpenFOAM Cp
    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    ax0.set_facecolor("#f8fafc")
    car_v, car_cp, proj_v, proj_cp = _build_continuous_full_car_stl_cp(
        p,
        raw_mesh,
        ms["components"],
        y_offset=0.0,
        z_floor_projection=-0.46,
        tri_stride=tri_stride,
    )
    if proj_v:
      ax0.add_collection3d(
          Poly3DCollection(
              proj_v,
              facecolors=cmap_cp(norm_cp(proj_cp)),
              edgecolors="none",
              alpha=0.82,
          )
      )
    ax0.add_collection3d(
        Poly3DCollection(
            car_v,
            facecolors=cmap_cp(norm_cp(car_cp)),
            edgecolors="#0f172a",
            linewidths=0.14,
            alpha=0.96,
        )
    )
    azim_deg = -142.0 + 78.0 * (idx / max(len(milestones) - 1, 1))
    ax0.set_xlim(-1.05, 4.10)
    ax0.set_ylim(-1.25, 1.25)
    ax0.set_zlim(-0.52, 1.15)
    ax0.set_box_aspect((5.15, 2.50, 1.67))
    ax0.view_init(elev=22, azim=azim_deg)
    ax0.set_xlabel("X [m]", fontsize=8.0, labelpad=-3)
    ax0.set_ylabel("Y [m]", fontsize=8.0, labelpad=-3)
    ax0.set_zlabel("Z [m]", fontsize=8.0, labelpad=-3)
    ax0.tick_params(labelsize=7.0)
    ax0.set_title(
        f"A. 3D PERRINN F1 OpenFOAM Surface Cp (h_f={p['front_ride_height_mm']:.1f}mm, "
        f"h_r={p['rear_ride_height_mm']:.1f}mm, Diff={p['diffuser_angle_deg']:.1f}°)",
        fontsize=9.1,
        fontweight="bold",
    )

    # Panel B: Live Venturi Floor Clearance & Suction -Cp(x)
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.set_facecolor("#ffffff")
    curr_neg_cp = -np.array(ms["cp_suction"], dtype=float)
    z_curr_mm = _floor_and_diffuser_box_mm(p, x_dense)
    ax1.plot(cp_x, base_neg_cp, "k--", lw=1.6, label="Gen 0 Seed -Cp(x)")
    ax1.plot(
        cp_x,
        curr_neg_cp,
        color="#1d4ed8",
        lw=2.4,
        label=f"Gen {gen} Suction -Cp(x) (Floor -SCz={ms['components']['underbody']['neg_SCz_m2']:.2f} m²)",
    )
    ax1.fill_between(
        cp_x,
        base_neg_cp,
        curr_neg_cp,
        where=(curr_neg_cp >= base_neg_cp),
        color="#93c5fd",
        alpha=0.38,
    )
    ax1_t = ax1.twinx()
    ax1_t.plot(x_dense, z_base_mm, color="#94a3b8", ls=":", lw=1.4)
    ax1_t.plot(
        x_dense,
        z_curr_mm,
        color="#15803d",
        ls="-.",
        lw=2.0,
        label="Floor / Diffuser Box [mm]",
    )
    ax1_t.axhline(175.0, color="#dc2626", ls="--", lw=1.2)
    ax1_t.set_ylim(0.0, 215.0)
    ax1_t.set_ylabel("Floor / Diffuser Box [mm]", fontsize=8.3, color="#15803d")
    ax1_t.tick_params(axis="y", labelcolor="#15803d", labelsize=7.5)

    ax1.set_xlim(-0.98, 3.92)
    ax1.set_ylim(-0.32, 1.02)
    ax1.set_xlabel("Streamwise X [m]", fontsize=8.5)
    ax1.set_ylabel("Suction -Cp(x) [-]", fontsize=8.5)
    ax1.set_title(
        f"B. Ground-Effect Suction & Raked Diffuser (Toe={p['fw_outwash_toe_deg']:.1f}°, "
        f"Undercut={p['sidepod_undercut_ratio']:.2f})",
        fontsize=9.1,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    # Panel C: Downforce vs. Front Aero Balance Trajectory
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_facecolor("#ffffff")
    ax2.axvspan(
        41.7,
        47.3,
        color="#dcfce7",
        alpha=0.75,
        label="PERRINN Neutral Window (44.5% ± 2.8%)",
    )
    ax2.axvline(44.5, color="#16a34a", ls="--", lw=1.5)
    ax2.plot(
        ms_fb[: idx + 1],
        ms_scz[: idx + 1],
        "o-",
        color="#2563eb",
        lw=2.2,
        ms=5.5,
        label="Evolutionary Trajectory",
    )
    ax2.scatter(
        [fb_now],
        [scz_now],
        color="#dc2626",
        edgecolors="#0f172a",
        s=95,
        zorder=6,
        label=f"Gen {gen} (L/D = {ld_now:.2f})",
    )
    ax2.set_xlim(38.0, 83.0)
    ax2.set_ylim(0.50, 1.80)
    ax2.set_xlabel("Front Aerodynamic Balance FB [%]", fontsize=8.5)
    ax2.set_ylabel("Full-Car Downforce Area -SCz [m²]", fontsize=8.5)
    ax2.set_title(
        "C. Downforce vs. Pitch-Moment Stability Convergence",
        fontsize=9.1,
        fontweight="bold",
    )
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    # Panel D: Live OpenFOAM Evolutionary Frontier
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_facecolor("#ffffff")
    seen_mask = [g <= gen for g in cand_gens]
    ax3.scatter(
        [g for g, m in zip(cand_gens, seen_mask) if m],
        [s for s, m in zip(cand_scores, seen_mask) if m],
        color="#94a3b8",
        edgecolors="#475569",
        s=40,
        alpha=0.80,
        label="Evaluated OpenFOAM Runs",
    )
    ax3.plot(
        ms_gens[: idx + 1],
        ms_scores[: idx + 1],
        "o-",
        color="#1d4ed8",
        lw=2.4,
        ms=5.5,
        label="Best Lap Score",
    )
    ax3.scatter(
        [gen], [score_now], color="#dc2626", edgecolors="#0f172a", s=90, zorder=6
    )
    ax3_t = ax3.twinx()
    ax3_t.plot(
        ms_gens[: idx + 1],
        ms_ld[: idx + 1],
        "^--",
        color="#b45309",
        lw=1.8,
        ms=4.5,
        label="Aero L/D [-]",
    )
    ax3_t.set_ylim(0.40, 1.35)
    ax3_t.set_ylabel("Aero Efficiency L/D [-]", fontsize=8.3, color="#b45309")
    ax3_t.tick_params(axis="y", labelcolor="#b45309", labelsize=7.5)

    ax3.set_xlim(-0.5, 12.6)
    ax3.set_ylim(-6.3, 2.6)
    ax3.set_xlabel("Generation", fontsize=8.5)
    ax3.set_ylabel("f1_aero_lap_score", fontsize=8.5, color="#1d4ed8")
    ax3.set_title(
        "D. 3D OpenFOAM Spalart-Allmaras RANS Score & L/D Progression",
        fontsize=9.1,
        fontweight="bold",
    )
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    fig.tight_layout(rect=[0, 0.01, 1, 0.91])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=gif_dpi)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))
    durations.append(900 if idx in (0, len(milestones) - 1) else 460)

  frames[0].save(
      out_path,
      save_all=True,
      append_images=frames[1:],
      duration=durations,
      loop=0,
  )
  return out_path


def postprocess(top_programs: list[dict[str, Any]] | None = None) -> None:
  """Postprocessing hook generating 3D OpenFOAM F1 aero comparison PNG & GIF."""
  if top_programs is not None and not top_programs:
    return
  bucket_name = os.getenv("_CLOUD_BUCKET_NAME", "").strip()
  user_exp = os.getenv("_USER_EXPERIMENT_NAME", "f1-racecar-openfoam").strip()

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
        "BEST PERRINN F1 AERO PROGRAM FOUND: %s (Score: %s)",
        best_prog_id,
        best_score,
    )

  best_data = None
  if isinstance(best_prog, Mapping) and best_prog.get("candidate_dir"):
    cand_dir = Path(str(best_prog["candidate_dir"]))
    for jpath in sorted(cand_dir.glob("*.json")):
      try:
        loaded = json.loads(jpath.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "neg_SCz_m2" in loaded:
          best_data = loaded
          break
      except Exception:  # pylint: disable=broad-exception-caught
        continue

  if best_data is None and bucket_name and best_prog_id:
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
      if isinstance(data, str):
        try:
          data = json.loads(data)
        except Exception:  # pylint: disable=broad-exception-caught
          data = None
      if isinstance(data, dict) and "neg_SCz_m2" in data:
        best_data = data
        break

  base_default, champ_default = _get_default_f1_telemetry()
  if best_data is None:
    best_data = dict(champ_default)
    if best_score is not None and math.isfinite(float(best_score)):
      best_data["score"] = float(best_score)

  results_env = os.getenv("_RESULTS_DIR")
  output_dir = Path(results_env) if results_env else SAMPLE_DIR
  output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = (
      Path(results_env) / "best_f1_aero_summary.json"
      if results_env
      else output_dir / "best_f1_aero_summary.json"
      if output_dir != _MODULE_DIR
      else Path("/tmp/best_f1_aero_summary.json")
  )
  summary_path.write_text(json.dumps(best_data, indent=2), encoding="utf-8")

  png_path = visualize_f1_aero_comparison(
      base_default,
      best_data,
      output_dir / "f1_aero_comparison.png",
  )
  gif_path = generate_f1_aero_evolution_gif(
      base_default,
      best_data,
      output_dir / "f1_aero_evolution.gif",
  )
  if bucket_name:
    for fpath in (summary_path, png_path, gif_path):
      if fpath.exists():
        upload_to_gcs(fpath, bucket_name, f"{user_exp}/results/{fpath.name}")


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))
