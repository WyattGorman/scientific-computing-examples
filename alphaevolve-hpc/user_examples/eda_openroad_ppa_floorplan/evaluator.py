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
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
import numpy as np

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

BASELINE_FLOORPLAN: dict[str, Any] = {
    "macros": {
        "SRAM_ICACHE_0": {"x_um": 120.0, "y_um": 160.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_ICACHE_1": {"x_um": 520.0, "y_um": 160.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_DCACHE_0": {"x_um": 120.0, "y_um": 390.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_DCACHE_1": {"x_um": 520.0, "y_um": 390.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_NPU_WT_0": {"x_um": 120.0, "y_um": 620.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_NPU_WT_1": {"x_um": 520.0, "y_um": 620.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_NPU_ACT_0": {"x_um": 120.0, "y_um": 850.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_L2_BANK_0": {"x_um": 520.0, "y_um": 850.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
    },
    "routing_config": {
        "target_density": 0.65,
        "layer_derating": 0.15,
    },
}

CHAMPION_FLOORPLAN: dict[str, Any] = {
    "macros": {
        "SRAM_ICACHE_0": {"x_um": 34.0, "y_um": 120.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_ICACHE_1": {"x_um": 34.0, "y_um": 310.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_DCACHE_0": {"x_um": 34.0, "y_um": 520.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_DCACHE_1": {"x_um": 34.0, "y_um": 740.0, "orientation": "R0", "halo_um": [10.0, 10.0]},
        "SRAM_NPU_WT_0": {"x_um": 936.0, "y_um": 120.0, "orientation": "MY", "halo_um": [10.0, 10.0]},
        "SRAM_NPU_WT_1": {"x_um": 936.0, "y_um": 330.0, "orientation": "MY", "halo_um": [10.0, 10.0]},
        "SRAM_NPU_ACT_0": {"x_um": 966.0, "y_um": 550.0, "orientation": "MY", "halo_um": [10.0, 10.0]},
        "SRAM_L2_BANK_0": {"x_um": 946.0, "y_um": 820.0, "orientation": "MY", "halo_um": [10.0, 10.0]},
    },
    "routing_config": {
        "target_density": 0.62,
        "layer_derating": 0.12,
    },
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


def score_floorplan_ppa(res: Mapping[str, Any]) -> float:
  """Computes the composite PPA score from Stage 2 OpenROAD evaluation results."""
  wns_ps = float(res["worst_wns_ps"])
  tns_ps = float(res["total_tns_ps"])
  hpwl_mm = float(res["weighted_hpwl_mm"])
  power_mw = float(res["total_power_mw"])
  overflow = float(res["routing_overflow"])
  max_util = float(res["max_gcell_utilization"])

  timing_term = (
      0.35 * min(wns_ps, 35.0)
      if wns_ps >= 0.0
      else 0.85 * wns_ps + 0.08 * tns_ps
  )
  wirelength_term = 25.0 * ((1350.0 - hpwl_mm) / 1350.0)
  power_term = 18.0 * ((66.0 - power_mw) / 66.0)
  congestion_penalty = -0.35 * overflow - 12.0 * max(0.0, max_util - 0.92)
  return round(
      timing_term + wirelength_term + power_term + congestion_penalty, 4
  )


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
      netlist, dict(floorplan), fast_stage=2
  )
  wns_ps = res["worst_wns_ps"]
  tns_ps = res["total_tns_ps"]
  hpwl_mm = res["weighted_hpwl_mm"]
  power_mw = res["total_power_mw"]
  overflow = res["routing_overflow"]
  max_util = res["max_gcell_utilization"]
  cts_skew = res["cts_skew_base_ps"]

  ppa_score = score_floorplan_ppa(res)

  elapsed_s = time.perf_counter() - t0
  ss_wns = res["corners"]["ss_0p63v_100c"]["wns_ps"]
  tt_wns = res["corners"]["tt_0p70v_25c"]["wns_ps"]
  ff_wns = res["corners"]["ff_0p77v_m40c"]["wns_ps"]

  or_bin = res.get("openroad_binary", {})
  or_status = or_bin.get("status", "none")

  candidate_dir = os.environ.get("_CANDIDATE_DIR")
  if candidate_dir:
    try:
      out_file = Path(candidate_dir) / "floorplan_ppa_metrics.json"
      out_file.write_text(
          json.dumps(
              {
                  "ppa_composite_score": ppa_score,
                  "floorplan": dict(floorplan),
                  "worst_wns_ps": wns_ps,
                  "total_tns_ps": tns_ps,
                  "weighted_hpwl_mm": hpwl_mm,
                  "total_power_mw": power_mw,
                  "routing_overflow": overflow,
                  "max_gcell_utilization": max_util,
                  "cts_skew_base_ps": cts_skew,
                  "spine_blockage_penalty_ps": res["spine_blockage_penalty_ps"],
                  "min_channel_um": res["min_channel_um"],
                  "corners": res["corners"],
                  "cluster_centroids": {
                      k: list(v) for k, v in res["cluster_centroids"].items()
                  },
                  "boxes": res.get("boxes", []),
                  "utilization_grid": res.get("utilization_grid", []),
                  "openroad_binary_status": or_status,
              },
              indent=2,
          ),
          encoding="utf-8",
      )
    except OSError:
      pass

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


def _draw_die_floorplan(
    ax: Any,
    netlist: Mapping[str, Any],
    ppa_res: Mapping[str, Any],
    title: str,
    score: float,
) -> None:
  """Renders 2D die floorplan, macros, active pin edges, clusters, and flylines."""
  from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # pylint: disable=g-import-not-at-top

  dx0, dy0, dx1, dy1 = netlist.get("die_bbox_um", [0.0, 0.0, 1200.0, 1200.0])
  cx0, cy0, cx1, cy1 = netlist["core_bbox_um"]

  # Die and core boundaries
  ax.add_patch(
      Rectangle(
          (dx0, dy0),
          dx1 - dx0,
          dy1 - dy0,
          facecolor="#F8FAFC",
          edgecolor="#334155",
          linewidth=1.5,
      )
  )
  ax.add_patch(
      Rectangle(
          (cx0, cy0),
          cx1 - cx0,
          cy1 - cy0,
          facecolor="#FFFFFF",
          edgecolor="#64748B",
          linestyle="--",
          linewidth=1.1,
      )
  )

  # Central TritonCTS H-tree clock spine corridor [380, 820] x [150, 1050]
  ax.add_patch(
      Rectangle(
          (380.0, 150.0),
          440.0,
          900.0,
          facecolor="#FEF3C7",
          edgecolor="#D97706",
          linestyle=":",
          linewidth=1.1,
          alpha=0.38,
      )
  )
  ax.text(
      600.0,
      1072.0,
      "TritonCTS Central Clock Spine Corridor",
      ha="center",
      va="bottom",
      fontsize=7.2,
      color="#92400E",
      fontweight="bold",
  )

  boxes = ppa_res.get("boxes", [])
  box_by_id = {b["id"]: b for b in boxes}
  cluster_coords = ppa_res.get("cluster_centroids", {})
  io_pads = netlist["io_pads"]

  def _node_xy(nid: str) -> tuple[float, float]:
    if nid in box_by_id:
      return (float(box_by_id[nid]["px"]), float(box_by_id[nid]["py"]))
    if nid in cluster_coords:
      pt = cluster_coords[nid]
      return (float(pt[0]), float(pt[1]))
    pad = io_pads[nid]
    return (float(pad["x_um"]), float(pad["y_um"]))

  # Draw weighted hypergraph flylines first (underneath macros & clusters)
  for net in netlist["hypergraph_nets"]:
    u, v = net["src"], net["dst"]
    w = float(net["weight"])
    x1, y1 = _node_xy(u)
    x2, y2 = _node_xy(v)
    lw = 0.55 + 1.45 * (w / 128.0)
    alpha = 0.25 + 0.35 * (w / 128.0)
    color = "#DC2626" if w >= 120.0 else "#2563EB" if w >= 80.0 else "#64748B"
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=2)

  # Draw placed SRAM macros + halos + active M4 pin edge highlight
  for b in boxes:
    x0, y0, w, h = float(b["x0"]), float(b["y0"]), float(b["w"]), float(b["h"])
    hx, hy = float(b["hx"]), float(b["hy"])
    orient = str(b["orient"])
    pin_side = str(b["pin_side"])

    # Halo keep-out rectangle
    ax.add_patch(
        Rectangle(
            (x0 - hx, y0 - hy),
            w + 2 * hx,
            h + 2 * hy,
            facecolor="none",
            edgecolor="#94A3B8",
            linestyle=":",
            linewidth=0.8,
            zorder=3,
        )
    )

    is_npu = "NPU" in b["id"] or "L2" in b["id"]
    macro_fill = "#DBEAFE" if not is_npu else "#E0E7FF"
    macro_edge = "#1E40AF" if not is_npu else "#3730A3"
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            facecolor=macro_fill,
            edgecolor=macro_edge,
            linewidth=1.4,
            alpha=0.92,
            zorder=4,
        )
    )

    # Highlight active M4 signal pin edge
    if pin_side == "E":
      ax.plot([x0 + w, x0 + w], [y0, y0 + h], color="#DC2626", linewidth=3.0, zorder=5)
      arr_dx = 26.0
    elif pin_side == "W":
      ax.plot([x0, x0], [y0, y0 + h], color="#16A34A", linewidth=3.0, zorder=5)
      arr_dx = -26.0
    else:
      arr_dx = 0.0

    px, py = float(b["px"]), float(b["py"])
    if arr_dx != 0.0:
      ax.add_patch(
          FancyArrowPatch(
              (px, py),
              (px + arr_dx, py),
              arrowstyle="->",
              mutation_scale=8,
              color="#DC2626" if pin_side == "E" else "#16A34A",
              linewidth=1.2,
              zorder=6,
          )
      )

    short_id = b["id"].replace("SRAM_", "")
    ax.text(
        x0 + 0.5 * w,
        y0 + 0.5 * h,
        f"{short_id}\n[{orient} | pin:{pin_side}]",
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color="#0F172A",
        zorder=6,
    )

  # Draw standard-cell functional clusters (RePlAce centroids)
  for c in netlist["stdcell_clusters"]:
    cid = c["id"]
    if cid not in cluster_coords:
      continue
    cx, cy = cluster_coords[cid]
    rad = max(24.0, min(48.0, (float(c["area_um2"]) ** 0.5) * 0.25))
    ax.add_patch(
        Circle(
            (cx, cy),
            radius=rad,
            facecolor="#DCFCE7",
            edgecolor="#15803D",
            linewidth=1.3,
            alpha=0.88,
            zorder=5,
        )
    )
    short_cl = cid.replace("CL_", "")
    ax.text(
        cx,
        cy,
        short_cl,
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color="#14532D",
        zorder=6,
    )

  # Draw peripheral IO pads
  for io_id, pad in io_pads.items():
    px, py = float(pad["x_um"]), float(pad["y_um"])
    ax.scatter([px], [py], s=48, marker="s", color="#D97706", edgecolor="#0F172A", zorder=7)
    ax.text(
        px,
        py + (22.0 if py < 600 else -26.0),
        io_id.replace("IO_", ""),
        ha="center",
        va="center",
        fontsize=6.3,
        fontweight="bold",
        color="#78350F",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEF3C7", edgecolor="none", alpha=0.85),
        zorder=7,
    )

  ax.set_xlim(-20.0, 1220.0)
  ax.set_ylim(-20.0, 1220.0)
  ax.set_aspect("equal")
  ax.set_xlabel("Die X Coordinate (um)", fontsize=8.5)
  ax.set_ylabel("Die Y Coordinate (um)", fontsize=8.5)
  ax.set_title(
      f"{title}\n"
      f"PPA Score={score:+.2f} | WNS={ppa_res['worst_wns_ps']:+.1f} ps | "
      f"HPWL={ppa_res['weighted_hpwl_mm']:.0f} mm | Spine Pen={ppa_res['spine_blockage_penalty_ps']:.1f} ps",
      fontsize=9.2,
      fontweight="bold",
  )


def visualize_floorplan_ppa_comparison(
    baseline_fp: Mapping[str, Any] | None = None,
    champion_fp: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    netlist: Mapping[str, Any] | None = None,
) -> Path:
  """Generates a 4-panel OpenROAD floorplan, congestion & multi-corner STA plot."""
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from matplotlib.patches import Rectangle  # pylint: disable=g-import-not-at-top

  if netlist is None:
    netlist = openroad_runner.load_netlist_graph(NETLIST_PATH)
  if baseline_fp is None:
    baseline_fp = BASELINE_FLOORPLAN
  if champion_fp is None:
    champion_fp = CHAMPION_FLOORPLAN
  if output_path is None:
    output_path = SAMPLE_DIR / "openroad_floorplan_ppa_comparison.png"
  out_path = Path(output_path)

  base_res = openroad_runner.evaluate_floorplan_ppa(
      dict(netlist), dict(baseline_fp), fast_stage=2, verify_binary=False
  )
  champ_res = openroad_runner.evaluate_floorplan_ppa(
      dict(netlist), dict(champion_fp), fast_stage=2, verify_binary=False
  )
  base_score = score_floorplan_ppa(base_res)
  champ_score = score_floorplan_ppa(champ_res)

  fig = plt.figure(figsize=(16.5, 13.0), dpi=160)
  fig.patch.set_facecolor("#F8FAFC")
  outer_gs = fig.add_gridspec(
      2, 2, hspace=0.28, wspace=0.22, left=0.05, right=0.97, top=0.91, bottom=0.06
  )

  fig.suptitle(
      "AlphaEvolve Semiconductor Physical Design: 7nm ASAP7 RISC-V + NPU Macro Floorplanning & PPA Signoff\n"
      "Baseline 2-Column Grid (Central Core & Clock Spine Blockage) vs. Champion Perimeter-Banked Floorplan (MY Pin Flip)",
      fontsize=13.0,
      fontweight="bold",
      color="#0F172A",
      y=0.97,
  )

  # Panel (A): Baseline 2D Die Floorplan
  ax_a = fig.add_subplot(outer_gs[0, 0])
  _draw_die_floorplan(
      ax_a,
      netlist,
      base_res,
      "(A) Baseline 2-Column Macro Grid (Obstructs Core & Clock Spine)",
      base_score,
  )

  # Panel (B): Champion Perimeter-Banked 2D Die Floorplan
  ax_b = fig.add_subplot(outer_gs[0, 1])
  _draw_die_floorplan(
      ax_b,
      netlist,
      champ_res,
      "(B) Champion Evolved Perimeter-Banked Floorplan (West R0 / East MY)",
      champ_score,
  )

  # Panel (C): Dual 24x24 GCell FastRoute RUDY Congestion Heatmaps (Baseline vs Champion)
  gs_c = outer_gs[1, 0].subgridspec(1, 2, wspace=0.18)
  ax_c1 = fig.add_subplot(gs_c[0, 0])
  ax_c2 = fig.add_subplot(gs_c[0, 1])

  base_grid = np.asarray(base_res.get("utilization_grid", np.zeros((24, 24))), dtype=np.float64)
  champ_grid = np.asarray(champ_res.get("utilization_grid", np.zeros((24, 24))), dtype=np.float64)
  cx0, cy0, cx1, cy1 = netlist["core_bbox_um"]
  vmax_util = max(1.15, float(np.max(base_grid)), float(np.max(champ_grid)))

  for sub_ax, grid, res_d, subtitle in [
      (
          ax_c1,
          base_grid,
          base_res,
          f"(C1) Baseline 24x24 GCell Util\nMax={base_res['max_gcell_utilization']:.2f} | Ovf={base_res['routing_overflow']:.1f}",
      ),
      (
          ax_c2,
          champ_grid,
          champ_res,
          f"(C2) Champion 24x24 GCell Util\nMax={champ_res['max_gcell_utilization']:.2f} | Ovf={champ_res['routing_overflow']:.1f}",
      ),
  ]:
    im = sub_ax.imshow(
        grid,
        origin="lower",
        extent=[cx0, cx1, cy0, cy1],
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax_util,
        aspect="equal",
    )
    for b in res_d.get("boxes", []):
      sub_ax.add_patch(
          Rectangle(
              (float(b["x0"]), float(b["y0"])),
              float(b["w"]),
              float(b["h"]),
              facecolor="#CBD5E1",
              alpha=0.35,
              edgecolor="#1E3A8A",
              linewidth=1.2,
              linestyle="--",
          )
      )
    sub_ax.set_title(subtitle, fontsize=8.8, fontweight="bold")
    sub_ax.set_xlabel("Core X (um)", fontsize=8.0)
    if sub_ax is ax_c1:
      sub_ax.set_ylabel("Core Y (um)", fontsize=8.0)
    else:
      sub_ax.set_yticklabels([])
  cbar = fig.colorbar(im, ax=[ax_c1, ax_c2], fraction=0.042, pad=0.03)
  cbar.set_label("GCell Routing Demand / Supply", fontsize=8.0)

  # Panel (D): Multi-Corner OpenSTA Timing Slacks & PPA Signoff Comparison
  ax_d = fig.add_subplot(outer_gs[1, 1])
  ss_base_paths = base_res["corners"]["ss_0p63v_100c"]["path_slacks_ps"]
  ss_champ_paths = champ_res["corners"]["ss_0p63v_100c"]["path_slacks_ps"]
  path_names = list(ss_base_paths.keys())
  short_paths = [p.replace("P", "P").split("_")[0] + ":" + "_".join(p.split("_")[1:3]) for p in path_names]

  # Combine 6 critical paths (at SS corner) + 3 PVT corner WNS bars
  labels = short_paths + ["WNS (SS)", "WNS (TT)", "WNS (FF)"]
  base_vals = [float(ss_base_paths[k]) for k in path_names] + [
      float(base_res["corners"]["ss_0p63v_100c"]["wns_ps"]),
      float(base_res["corners"]["tt_0p70v_25c"]["wns_ps"]),
      float(base_res["corners"]["ff_0p77v_m40c"]["wns_ps"]),
  ]
  champ_vals = [float(ss_champ_paths[k]) for k in path_names] + [
      float(champ_res["corners"]["ss_0p63v_100c"]["wns_ps"]),
      float(champ_res["corners"]["tt_0p70v_25c"]["wns_ps"]),
      float(champ_res["corners"]["ff_0p77v_m40c"]["wns_ps"]),
  ]

  y_pos = np.arange(len(labels))
  bar_h = 0.36
  ax_d.barh(
      y_pos - bar_h / 2,
      base_vals,
      height=bar_h,
      color="#DC2626",
      alpha=0.85,
      label=f"Baseline (Worst WNS={base_res['worst_wns_ps']:+.1f} ps, TNS={base_res['total_tns_ps']:.1f} ps)",
  )
  ax_d.barh(
      y_pos + bar_h / 2,
      champ_vals,
      height=bar_h,
      color="#059669",
      alpha=0.88,
      label=f"Champion (Worst WNS={champ_res['worst_wns_ps']:+.1f} ps, TNS={champ_res['total_tns_ps']:.1f} ps)",
  )
  ax_d.axvline(0.0, color="#0F172A", linewidth=1.3, linestyle="--", label="Setup Signoff Target (0.0 ps)")
  ax_d.axhline(len(short_paths) - 0.5, color="#94A3B8", linewidth=0.9, linestyle=":")
  ax_d.set_yticks(y_pos)
  ax_d.set_yticklabels(labels, fontsize=7.8, fontweight="bold")
  ax_d.set_xlim(-175.0, 185.0)
  ax_d.set_xlabel("Setup Timing Slack (ps) at 500 ps (2.0 GHz) Clock", fontsize=8.5)
  ax_d.set_title(
      "(D) 3-Corner OpenSTA Setup Timing Closure & PPA Signoff Summary",
      fontsize=9.5,
      fontweight="bold",
  )
  ax_d.grid(True, axis="x", linestyle=":", alpha=0.5)
  ax_d.legend(loc="lower right", fontsize=7.4, framealpha=0.92)

  summary_box = (
      f"PPA Signoff Summary (7nm ASAP7)\n"
      f"---------------------------------------\n"
      f"Metric          Baseline    Champion\n"
      f"PPA Score:     {base_score:+8.2f}    {champ_score:+8.2f}\n"
      f"Worst WNS(ps): {base_res['worst_wns_ps']:+8.1f}    {champ_res['worst_wns_ps']:+8.1f}\n"
      f"Total TNS(ps): {base_res['total_tns_ps']:+8.1f}    {champ_res['total_tns_ps']:+8.1f}\n"
      f"HPWL (mm):     {base_res['weighted_hpwl_mm']:8.1f}    {champ_res['weighted_hpwl_mm']:8.1f}\n"
      f"Power (mW):    {base_res['total_power_mw']:8.2f}    {champ_res['total_power_mw']:8.2f}\n"
      f"CTS Skew(ps):  {base_res['cts_skew_base_ps']:8.1f}    {champ_res['cts_skew_base_ps']:8.1f}"
  )
  ax_d.text(
      0.015,
      0.34,
      summary_box,
      transform=ax_d.transAxes,
      fontsize=6.9,
      fontfamily="monospace",
      va="center",
      bbox=dict(
          boxstyle="round,pad=0.35",
          facecolor="#F8FAFC",
          edgecolor="#64748B",
          alpha=0.95,
      ),
  )

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=160)
  plt.close(fig)
  return out_path


def postprocess(top_programs: list[Any] | None = None) -> None:
  """Post-processes top evolved floorplan programs and renders comparison plot."""
  netlist = openroad_runner.load_netlist_graph(NETLIST_PATH)
  champ_fp: Mapping[str, Any] = CHAMPION_FLOORPLAN
  if top_programs:
    best = top_programs[0]
    cand_dir = getattr(best, "candidate_dir", None) or (
        best.get("candidate_dir") if isinstance(best, Mapping) else None
    )
    if cand_dir:
      metrics_path = Path(cand_dir) / "floorplan_ppa_metrics.json"
      if metrics_path.is_file():
        try:
          loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
          if isinstance(loaded.get("floorplan"), Mapping):
            champ_fp = loaded["floorplan"]
        except (OSError, json.JSONDecodeError):
          pass

  visualize_floorplan_ppa_comparison(
      baseline_fp=BASELINE_FLOORPLAN,
      champion_fp=champ_fp,
      output_path=SAMPLE_DIR / "openroad_floorplan_ppa_comparison.png",
      netlist=netlist,
  )


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))

