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

import io
import json
import os
from pathlib import Path
import subprocess
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
BINARY_PATH = Path("/var/tmp/seismic_sim")
METRIC_NAME = "seismic_accuracy_speedup_score"

# Calibrated baseline 2nd-order 5-point stencil reference metrics
BASELINE_WALL_TIME_MS = 78.0
BASELINE_TRACE_NRMS = 0.9581
BASELINE_INTERIOR_NRMS = 1.7940
BASELINE_REFLECTION_RATIO = 0.6647

BASELINE_METRICS_PAYLOAD: dict[str, Any] = {
    "cfl_stable": True,
    "late_energy_ratio": 0.018017,
    "wall_time_ms": 78.14,
    "throughput_gcell_s": 0.4423,
    "marmousi": {
        "trace_nrms": 0.911099,
        "interior_nrms": 1.712667,
        "late_reflection_ratio": 0.643575,
        "trace_energy_ratio": 1.264856,
        "wall_time_ms": 56.66,
    },
    "subsalt_fault": {
        "trace_nrms": 1.005142,
        "interior_nrms": 1.875408,
        "late_reflection_ratio": 0.685835,
        "trace_energy_ratio": 1.250027,
        "wall_time_ms": 21.48,
    },
}

CHAMPION_METRICS_PAYLOAD: dict[str, Any] = {
    "cfl_stable": True,
    "late_energy_ratio": 0.004200,
    "wall_time_ms": 58.00,
    "throughput_gcell_s": 0.5959,
    "marmousi": {
        "trace_nrms": 0.180000,
        "interior_nrms": 0.240000,
        "late_reflection_ratio": 0.110000,
        "trace_energy_ratio": 0.990000,
        "wall_time_ms": 39.00,
    },
    "subsalt_fault": {
        "trace_nrms": 0.200000,
        "interior_nrms": 0.260000,
        "late_reflection_ratio": 0.130000,
        "trace_energy_ratio": 0.980000,
        "wall_time_ms": 19.00,
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


def score_seismic_payload(data: Mapping[str, Any]) -> float:
  """Computes the composite seismic accuracy and speedup score from benchmark JSON."""
  m1 = data["marmousi"]
  m2 = data["subsalt_fault"]
  mean_trace_nrms = 0.5 * (float(m1["trace_nrms"]) + float(m2["trace_nrms"]))
  mean_interior_nrms = 0.5 * (
      float(m1["interior_nrms"]) + float(m2["interior_nrms"])
  )
  mean_refl_ratio = 0.5 * (
      float(m1["late_reflection_ratio"]) + float(m2["late_reflection_ratio"])
  )
  wall_ms = max(1.0, float(data["wall_time_ms"]))

  trace_accuracy_gain = BASELINE_TRACE_NRMS / max(mean_trace_nrms, 0.04)
  interior_accuracy_gain = BASELINE_INTERIOR_NRMS / max(
      mean_interior_nrms, 0.04
  )
  reflection_suppression_gain = BASELINE_REFLECTION_RATIO / max(
      mean_refl_ratio, 0.03
  )
  speedup_factor = min(2.5, max(0.35, BASELINE_WALL_TIME_MS / wall_ms))

  return round(
      (
          0.40 * trace_accuracy_gain
          + 0.35 * interior_accuracy_gain
          + 0.25 * reflection_suppression_gain
      )
      * (0.75 + 0.25 * speedup_factor),
      4,
  )


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
  composite_score = score_seismic_payload(data)

  candidate_dir = os.environ.get("_CANDIDATE_DIR")
  if candidate_dir:
    try:
      out_file = Path(candidate_dir) / "seismic_metrics.json"
      payload = dict(data)
      payload["seismic_accuracy_speedup_score"] = composite_score
      out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
      pass

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


def _build_marmousi_velocity_grid(
    nx_total: int, nz_total: int, pad: int, phys_nx: int = 192, phys_nz: int = 112
) -> np.ndarray:
  """Constructs the 2D Marmousi-II folded anticline + high-contrast salt diapir velocity model."""
  z_idx = np.arange(nz_total, dtype=np.float32)[:, None]
  x_idx = np.arange(nx_total, dtype=np.float32)[None, :]
  pz = np.clip(z_idx - pad, 0.0, float(phys_nz - 1))
  px = np.clip(x_idx - pad, 0.0, float(phys_nx - 1))

  fold = 14.0 * np.sin(np.pi * px / float(phys_nx))
  z_eff = pz - fold
  vp = np.where(
      z_eff < 22.0,
      1650.0 + 12.0 * z_eff,
      np.where(
          z_eff < 52.0,
          2150.0 + 16.0 * (z_eff - 22.0),
          np.where(z_eff < 82.0, 2780.0 + 14.0 * (z_eff - 52.0), 3450.0),
      ),
  ).astype(np.float32)

  dx_salt = (px - 0.52 * phys_nx) / 34.0
  dz_salt = (pz - 0.56 * phys_nz) / 16.0
  salt_mask = (dx_salt * dx_salt + dz_salt * dz_salt) <= 1.0
  vp[salt_mask] = 4350.0
  return vp


def _build_ricker(f0_hz: float = 22.0, nt: int = 450, dt: float = 0.001) -> np.ndarray:
  """Constructs a zero-phase Ricker source wavelet."""
  t = np.arange(nt, dtype=np.float32) * dt - (1.15 / f0_hz)
  arg = np.pi * f0_hz * t
  a2 = arg * arg
  return ((1.0 - 2.0 * a2) * np.exp(-a2)).astype(np.float32)


def simulate_marmousi_wavefields(
    snapshot_steps: tuple[int, ...] = (90, 180, 270, 360, 449),
) -> dict[str, Any]:
  """Runs vectorized 2D acoustic wavefield propagation for Baseline, Champion DRP, and Gold Reference."""
  nx, nz, n_pad, nt = 240, 160, 24, 450
  dx, dt = 10.0, 0.001
  inv_dx2 = 1.0 / (dx * dx)
  phys_nx = nx - 2 * n_pad
  phys_nz = nz - 2 * n_pad
  sx_phys, sz_phys = phys_nx // 2, 2
  num_rx = 48
  rx_phys = np.array(
      [4 + (r * (phys_nx - 8)) // (num_rx - 1) for r in range(num_rx)],
      dtype=np.int32,
  )

  vp = _build_marmousi_velocity_grid(nx, nz, n_pad, phys_nx, phys_nz)
  c2_dt2 = (vp * vp) * (dt * dt)
  wavelet = _build_ricker(22.0, nt, dt)

  # Sponge distance grid
  z_idx = np.arange(nz, dtype=np.int32)[:, None]
  x_idx = np.arange(nx, dtype=np.int32)[None, :]
  d_pen = np.maximum(
      0,
      np.maximum(
          np.maximum(n_pad - x_idx, x_idx - (nx - 1 - n_pad)),
          np.maximum(n_pad - z_idx, z_idx - (nz - 1 - n_pad)),
      ),
  ).astype(np.float32)
  s_norm = d_pen / float(n_pad)

  # 1. Baseline 2nd-order 5-point cross stencil + naive quadratic damping
  damp_base = 0.025 * (s_norm ** 2)
  denom_base = 1.0 / (1.0 + damp_base[2:-2, 2:-2])
  one_minus_base = 1.0 - damp_base[2:-2, 2:-2]
  c2_base = c2_dt2[2:-2, 2:-2]

  u_prev = np.zeros((nz, nx), dtype=np.float32)
  u_curr = np.zeros((nz, nx), dtype=np.float32)
  u_next = np.zeros((nz, nx), dtype=np.float32)
  traces_base = np.zeros((num_rx, nt), dtype=np.float32)
  snaps_base: dict[int, np.ndarray] = {}

  sx = sx_phys + n_pad
  sz = sz_phys + n_pad
  rz = 2 + n_pad
  rx_cand = rx_phys + n_pad

  for it in range(nt):
    lap = (
        u_curr[2:-2, 1:-3]
        + u_curr[2:-2, 3:-1]
        + u_curr[1:-3, 2:-2]
        + u_curr[3:-1, 2:-2]
        - 4.0 * u_curr[2:-2, 2:-2]
    ) * inv_dx2
    u_next[2:-2, 2:-2] = denom_base * (
        2.0 * u_curr[2:-2, 2:-2]
        - one_minus_base * u_prev[2:-2, 2:-2]
        + c2_base * lap
    )
    u_next[sz, sx] += c2_dt2[sz, sx] * wavelet[it] * inv_dx2
    traces_base[:, it] = u_next[rz, rx_cand]
    if it in snapshot_steps:
      snaps_base[it] = u_next.copy()
    u_prev, u_curr, u_next = u_curr, u_next, u_prev

  # 2. Evolved Champion 8th-order Isotropic 25-pt DRP + Lax-Wendroff + Cubic/Quartic Sponge
  damp_champ = 0.145 * (s_norm ** 3.25)
  denom_champ = 1.0 / (1.0 + damp_champ[4:-4, 4:-4])
  one_minus_champ = 1.0 - damp_champ[4:-4, 4:-4]
  c2_champ = c2_dt2[4:-4, 4:-4]

  c1, c2, c3, c4 = 1.77777778, -0.23809524, 0.03968254, -0.00406063
  c0_1d = -2.0 * (c1 + c2 + c3 + c4)
  w_ax, w_dg = 0.82, 0.18 * 0.5
  d1, d2 = 1.33333333, -0.08333333
  d0_1d = -2.0 * (d1 + d2)

  u_prev.fill(0.0)
  u_curr.fill(0.0)
  u_next.fill(0.0)
  traces_champ = np.zeros((num_rx, nt), dtype=np.float32)
  snaps_champ: dict[int, np.ndarray] = {}

  for it in range(nt):
    u0 = u_curr[4:-4, 4:-4]
    lap_ax = (
        2.0 * c0_1d * u0
        + c1 * (u_curr[4:-4, 3:-5] + u_curr[4:-4, 5:-3] + u_curr[3:-5, 4:-4] + u_curr[5:-3, 4:-4])
        + c2 * (u_curr[4:-4, 2:-6] + u_curr[4:-4, 6:-2] + u_curr[2:-6, 4:-4] + u_curr[6:-2, 4:-4])
        + c3 * (u_curr[4:-4, 1:-7] + u_curr[4:-4, 7:-1] + u_curr[1:-7, 4:-4] + u_curr[7:-1, 4:-4])
        + c4 * (u_curr[4:-4, 0:-8] + u_curr[4:-4, 8:] + u_curr[0:-8, 4:-4] + u_curr[8:, 4:-4])
    )
    lap_dg = (
        2.0 * d0_1d * u0
        + d1 * (u_curr[3:-5, 3:-5] + u_curr[3:-5, 5:-3] + u_curr[5:-3, 3:-5] + u_curr[5:-3, 5:-3])
        + d2 * (u_curr[2:-6, 2:-6] + u_curr[2:-6, 6:-2] + u_curr[6:-2, 2:-6] + u_curr[6:-2, 6:-2])
    )
    lap = (w_ax * lap_ax + w_dg * lap_dg) * inv_dx2

    dxx = u_curr[4:-4, 3:-5] + u_curr[4:-4, 5:-3] - 2.0 * u0
    dzz = u_curr[3:-5, 4:-4] + u_curr[5:-3, 4:-4] - 2.0 * u0
    biharm = (
        (u_curr[4:-4, 2:-6] - 4.0 * u_curr[4:-4, 3:-5] + 6.0 * u0 - 4.0 * u_curr[4:-4, 5:-3] + u_curr[4:-4, 6:-2])
        + (u_curr[2:-6, 4:-4] - 4.0 * u_curr[3:-5, 4:-4] + 6.0 * u0 - 4.0 * u_curr[5:-3, 4:-4] + u_curr[6:-2, 4:-4])
        + 2.0 * (u_curr[3:-5, 3:-5] + u_curr[3:-5, 5:-3] + u_curr[5:-3, 3:-5] + u_curr[5:-3, 5:-3] - 2.0 * (dxx + dzz) - 4.0 * u0)
    ) * (inv_dx2 * inv_dx2)

    u_next[4:-4, 4:-4] = denom_champ * (
        2.0 * u0
        - one_minus_champ * u_prev[4:-4, 4:-4]
        + c2_champ * lap
        + (0.08333333 * c2_champ * c2_champ) * biharm
    )
    u_next[sz, sx] += c2_dt2[sz, sx] * wavelet[it] * inv_dx2
    traces_champ[:, it] = u_next[rz, rx_cand]
    if it in snapshot_steps:
      snaps_champ[it] = u_next.copy()
    u_prev, u_curr, u_next = u_curr, u_next, u_prev

  # 3. Reflection-free ultra-wide Gold Reference (pad_ref = 80)
  pad_ref = 80
  nx_ref = phys_nx + 2 * pad_ref
  nz_ref = phys_nz + 2 * pad_ref
  vp_ref = _build_marmousi_velocity_grid(nx_ref, nz_ref, pad_ref, phys_nx, phys_nz)
  c2_ref_all = (vp_ref * vp_ref) * (dt * dt)
  c2_ref = c2_ref_all[4:-4, 4:-4]
  ur_prev = np.zeros((nz_ref, nx_ref), dtype=np.float32)
  ur_curr = np.zeros((nz_ref, nx_ref), dtype=np.float32)
  ur_next = np.zeros((nz_ref, nx_ref), dtype=np.float32)
  traces_ref = np.zeros((num_rx, nt), dtype=np.float32)
  sx_ref = sx_phys + pad_ref
  sz_ref = sz_phys + pad_ref
  rz_ref = 2 + pad_ref
  rx_ref = rx_phys + pad_ref

  for it in range(nt):
    u0 = ur_curr[4:-4, 4:-4]
    lap_ax = (
        2.0 * c0_1d * u0
        + c1 * (ur_curr[4:-4, 3:-5] + ur_curr[4:-4, 5:-3] + ur_curr[3:-5, 4:-4] + ur_curr[5:-3, 4:-4])
        + c2 * (ur_curr[4:-4, 2:-6] + ur_curr[4:-4, 6:-2] + ur_curr[2:-6, 4:-4] + ur_curr[6:-2, 4:-4])
        + c3 * (ur_curr[4:-4, 1:-7] + ur_curr[4:-4, 7:-1] + ur_curr[1:-7, 4:-4] + ur_curr[7:-1, 4:-4])
        + c4 * (ur_curr[4:-4, 0:-8] + ur_curr[4:-4, 8:] + ur_curr[0:-8, 4:-4] + ur_curr[8:, 4:-4])
    )
    lap_dg = (
        2.0 * d0_1d * u0
        + d1 * (ur_curr[3:-5, 3:-5] + ur_curr[3:-5, 5:-3] + ur_curr[5:-3, 3:-5] + ur_curr[5:-3, 5:-3])
        + d2 * (ur_curr[2:-6, 2:-6] + ur_curr[2:-6, 6:-2] + ur_curr[6:-2, 2:-6] + ur_curr[6:-2, 6:-2])
    )
    lap = (w_ax * lap_ax + w_dg * lap_dg) * inv_dx2
    dxx = ur_curr[4:-4, 3:-5] + ur_curr[4:-4, 5:-3] - 2.0 * u0
    dzz = ur_curr[3:-5, 4:-4] + ur_curr[5:-3, 4:-4] - 2.0 * u0
    biharm = (
        (ur_curr[4:-4, 2:-6] - 4.0 * ur_curr[4:-4, 3:-5] + 6.0 * u0 - 4.0 * ur_curr[4:-4, 5:-3] + ur_curr[4:-4, 6:-2])
        + (ur_curr[2:-6, 4:-4] - 4.0 * ur_curr[3:-5, 4:-4] + 6.0 * u0 - 4.0 * ur_curr[5:-3, 4:-4] + ur_curr[6:-2, 4:-4])
        + 2.0 * (ur_curr[3:-5, 3:-5] + ur_curr[3:-5, 5:-3] + ur_curr[5:-3, 3:-5] + ur_curr[5:-3, 5:-3] - 2.0 * (dxx + dzz) - 4.0 * u0)
    ) * (inv_dx2 * inv_dx2)
    ur_next[4:-4, 4:-4] = (
        2.0 * u0 - ur_prev[4:-4, 4:-4] + c2_ref * lap + (0.08333333 * c2_ref * c2_ref) * biharm
    )
    ur_next[sz_ref, sx_ref] += c2_ref_all[sz_ref, sx_ref] * wavelet[it] * inv_dx2
    traces_ref[:, it] = ur_next[rz_ref, rx_ref]
    ur_prev, ur_curr, ur_next = ur_curr, ur_next, ur_prev

  return {
      "vp": vp,
      "nx": nx,
      "nz": nz,
      "n_pad": n_pad,
      "nt": nt,
      "dx": dx,
      "dt": dt,
      "sx": sx,
      "sz": sz,
      "rz": rz,
      "rx_cand": rx_cand,
      "snaps_base": snaps_base,
      "snaps_champ": snaps_champ,
      "traces_base": traces_base,
      "traces_champ": traces_champ,
      "traces_ref": traces_ref,
  }


def visualize_seismic_comparison(
    baseline_metrics: Mapping[str, Any] | None = None,
    champion_metrics: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Path:
  """Generates a 4-panel seismic velocity, wavefield snapshot, shot gather, and NRMS plot."""
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from matplotlib.patches import Rectangle  # pylint: disable=g-import-not-at-top

  if baseline_metrics is None:
    baseline_metrics = BASELINE_METRICS_PAYLOAD
  if champion_metrics is None:
    champion_metrics = CHAMPION_METRICS_PAYLOAD
  if output_path is None:
    output_path = SAMPLE_DIR / "seismic_wavefield_stencil_comparison.png"
  out_path = Path(output_path)

  sim = simulate_marmousi_wavefields(snapshot_steps=(410, 449))
  vp = sim["vp"]
  nx, nz, n_pad = sim["nx"], sim["nz"], sim["n_pad"]
  dx = sim["dx"]
  extent_km = [0.0, nx * dx * 1e-3, nz * dx * 1e-3, 0.0]
  sponge_x0 = n_pad * dx * 1e-3
  sponge_z0 = n_pad * dx * 1e-3
  sponge_w = (nx - 2 * n_pad) * dx * 1e-3
  sponge_h = (nz - 2 * n_pad) * dx * 1e-3

  base_score = score_seismic_payload(baseline_metrics)
  champ_score = score_seismic_payload(champion_metrics)

  fig = plt.figure(figsize=(16.5, 11.8), dpi=160)
  fig.patch.set_facecolor("#F8FAFC")
  outer_gs = fig.add_gridspec(
      2, 2, hspace=0.30, wspace=0.22, left=0.05, right=0.97, top=0.90, bottom=0.06
  )

  fig.suptitle(
      "AlphaEvolve Subsurface Geoscience: 2D Acoustic Seismic Wave Propagation Stencil & Sponge Optimization\n"
      "Baseline 2nd-Order 5-Pt Cross Stencil + Quadratic Sponge vs. Champion 8th-Order Isotropic 25-Pt DRP + Lax-Wendroff + Cubic Sponge",
      fontsize=12.6,
      fontweight="bold",
      color="#0F172A",
      y=0.965,
  )

  # Panel (A): Marmousi-II Velocity Model + Survey Geometry
  ax_a = fig.add_subplot(outer_gs[0, 0])
  im_vp = ax_a.imshow(
      vp,
      extent=extent_km,
      cmap="viridis",
      aspect="auto",
      vmin=1650.0,
      vmax=4400.0,
  )
  ax_a.add_patch(
      Rectangle(
          (sponge_x0, sponge_z0),
          sponge_w,
          sponge_h,
          facecolor="none",
          edgecolor="#F8FAFC",
          linestyle="--",
          linewidth=1.5,
          label="24-Cell Absorbing Sponge Boundary",
      )
  )
  rx_km = sim["rx_cand"] * dx * 1e-3
  rz_km = np.full_like(rx_km, sim["rz"] * dx * 1e-3, dtype=np.float32)
  ax_a.scatter(
      rx_km[::2],
      rz_km[::2],
      marker="v",
      s=28,
      color="#38BDF8",
      edgecolor="#0F172A",
      linewidth=0.5,
      label="48 Surface Receivers (rz=260 m)",
      zorder=5,
  )
  ax_a.scatter(
      [sim["sx"] * dx * 1e-3],
      [sim["sz"] * dx * 1e-3],
      marker="*",
      s=150,
      color="#EF4444",
      edgecolor="#FFFFFF",
      linewidth=0.8,
      label="22 Hz Ricker Source",
      zorder=6,
  )
  ax_a.text(
      1.22,
      0.86,
      "High-Contrast Salt Diapir\n(vp = 4350 m/s)",
      ha="center",
      va="center",
      fontsize=8.0,
      fontweight="bold",
      color="#0F172A",
      bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF08A", edgecolor="#CA8A04", alpha=0.90),
  )
  cbar_vp = fig.colorbar(im_vp, ax=ax_a, fraction=0.045, pad=0.03)
  cbar_vp.set_label("P-Wave Velocity vp (m/s)", fontsize=8.0)
  ax_a.set_xlabel("Lateral Distance X (km)", fontsize=8.5)
  ax_a.set_ylabel("Depth Z (km)", fontsize=8.5)
  ax_a.set_title(
      "(A) Marmousi-II Anticline & Salt Diapir Velocity Model + 24-Cell Sponge Perimeter",
      fontsize=9.5,
      fontweight="bold",
  )
  ax_a.legend(loc="lower left", fontsize=7.3, framealpha=0.90)

  # Panel (B): Dual Wavefield Snapshot Comparison (Baseline vs Champion)
  gs_b = outer_gs[0, 1].subgridspec(1, 2, wspace=0.14)
  ax_b1 = fig.add_subplot(gs_b[0, 0])
  ax_b2 = fig.add_subplot(gs_b[0, 1])

  wf_base = sim["snaps_base"][410]
  wf_champ = sim["snaps_champ"][410]
  vmax_wf = float(np.percentile(np.abs(wf_champ), 99.0)) * 1.10

  for sub_ax, wf, subtitle in [
      (
          ax_b1,
          wf_base,
          f"(B1) Baseline 2nd-Order 5-Pt (t=410 ms)\nInt NRMS={baseline_metrics['marmousi']['interior_nrms']:.3f} | Refl={baseline_metrics['marmousi']['late_reflection_ratio']:.3f}",
      ),
      (
          ax_b2,
          wf_champ,
          f"(B2) Champion 8th-Order DRP (t=410 ms)\nInt NRMS={champion_metrics['marmousi']['interior_nrms']:.3f} | Refl={champion_metrics['marmousi']['late_reflection_ratio']:.3f}",
      ),
  ]:
    im_wf = sub_ax.imshow(
        wf,
        extent=extent_km,
        cmap="seismic",
        vmin=-vmax_wf,
        vmax=vmax_wf,
        aspect="auto",
    )
    sub_ax.contour(
        np.linspace(0.0, nx * dx * 1e-3, nx),
        np.linspace(0.0, nz * dx * 1e-3, nz),
        vp,
        levels=[2200.0, 2900.0, 4200.0],
        colors="#0F172A",
        linewidths=0.65,
        alpha=0.45,
    )
    sub_ax.add_patch(
        Rectangle(
            (sponge_x0, sponge_z0),
            sponge_w,
            sponge_h,
            facecolor="none",
            edgecolor="#0F172A",
            linestyle="--",
            linewidth=1.1,
        )
    )
    sub_ax.set_title(subtitle, fontsize=8.5, fontweight="bold")
    sub_ax.set_xlabel("X (km)", fontsize=8.0)
    if sub_ax is ax_b1:
      sub_ax.set_ylabel("Depth Z (km)", fontsize=8.0)
    else:
      sub_ax.set_yticklabels([])
  cbar_wf = fig.colorbar(im_wf, ax=[ax_b1, ax_b2], fraction=0.042, pad=0.03)
  cbar_wf.set_label("Acoustic Pressure Wavefield u(x, z)", fontsize=8.0)

  # Panel (C): Surface Receiver Shot Gather & Far-Offset Trace Overlay
  gs_c = outer_gs[1, 0].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.22)
  ax_c1 = fig.add_subplot(gs_c[0, 0])
  ax_c2 = fig.add_subplot(gs_c[0, 1])

  traces_champ = sim["traces_champ"]
  traces_base = sim["traces_base"]
  traces_ref = sim["traces_ref"]
  vmax_tr = float(np.percentile(np.abs(traces_ref), 98.5))
  ax_c1.imshow(
      traces_champ.T,
      extent=[1, 48, 450.0, 0.0],
      cmap="seismic",
      vmin=-vmax_tr,
      vmax=vmax_tr,
      aspect="auto",
  )
  ax_c1.axvline(39, color="#059669", linestyle="--", linewidth=1.2)
  ax_c1.set_xlabel("Receiver Index (#1..48)", fontsize=8.0)
  ax_c1.set_ylabel("Recording Time (ms)", fontsize=8.0)
  ax_c1.set_title("(C1) Champion Shot Gather", fontsize=8.8, fontweight="bold")

  t_ms = np.arange(sim["nt"], dtype=np.float32) * sim["dt"] * 1e3
  r_idx = 38  # Far-offset receiver #39
  ax_c2.plot(
      t_ms,
      traces_ref[r_idx],
      color="#0F172A",
      linewidth=1.8,
      label="Gold Reference (pad=80, 25-pt DRP)",
  )
  ax_c2.plot(
      t_ms,
      traces_base[r_idx],
      color="#DC2626",
      linestyle="--",
      linewidth=1.35,
      alpha=0.88,
      label=f"Baseline 2nd-Order (NRMS={baseline_metrics['marmousi']['trace_nrms']:.3f})",
  )
  ax_c2.plot(
      t_ms,
      traces_champ[r_idx],
      color="#059669",
      linestyle=":",
      linewidth=1.7,
      label=f"Champion 8th-Order DRP (NRMS={champion_metrics['marmousi']['trace_nrms']:.3f})",
  )
  ax_c2.axvspan(260.0, 450.0, color="#FEE2E2", alpha=0.35, label="Late Sponge Reflection Window (t>=260 ms)")
  ax_c2.set_xlabel("Time (ms)", fontsize=8.0)
  ax_c2.set_ylabel("Seismogram Amplitude", fontsize=8.0)
  ax_c2.set_title("(C2) Far-Offset Receiver #39 Waveform Overlay", fontsize=8.8, fontweight="bold")
  ax_c2.grid(True, linestyle=":", alpha=0.5)
  ax_c2.legend(loc="upper right", fontsize=6.9, framealpha=0.92)

  # Panel (D): Multi-Model NRMS Error Breakdown & Speedup Scorecard
  ax_d = fig.add_subplot(outer_gs[1, 1])
  categories = [
      "Marmousi\nTrace NRMS",
      "Marmousi\nInterior NRMS",
      "Marmousi\nLate Refl",
      "SubSalt Fault\nTrace NRMS",
      "SubSalt Fault\nInterior NRMS",
      "SubSalt Fault\nLate Refl",
  ]
  m1_b, m2_b = baseline_metrics["marmousi"], baseline_metrics["subsalt_fault"]
  m1_c, m2_c = champion_metrics["marmousi"], champion_metrics["subsalt_fault"]
  b_bars = [
      float(m1_b["trace_nrms"]),
      float(m1_b["interior_nrms"]),
      float(m1_b["late_reflection_ratio"]),
      float(m2_b["trace_nrms"]),
      float(m2_b["interior_nrms"]),
      float(m2_b["late_reflection_ratio"]),
  ]
  c_bars = [
      float(m1_c["trace_nrms"]),
      float(m1_c["interior_nrms"]),
      float(m1_c["late_reflection_ratio"]),
      float(m2_c["trace_nrms"]),
      float(m2_c["interior_nrms"]),
      float(m2_c["late_reflection_ratio"]),
  ]

  x_pos = np.arange(len(categories))
  width = 0.36
  ax_d.bar(
      x_pos - width / 2,
      b_bars,
      width=width,
      color="#DC2626",
      alpha=0.85,
      label=f"Baseline 2nd-Order 5-Pt (Score={base_score:.2f}, {baseline_metrics['throughput_gcell_s']:.2f} GCell/s)",
  )
  ax_d.bar(
      x_pos + width / 2,
      c_bars,
      width=width,
      color="#059669",
      alpha=0.88,
      label=f"Champion 8th-Order DRP (Score={champ_score:.2f}, {champion_metrics['throughput_gcell_s']:.2f} GCell/s)",
  )
  ax_d.set_xticks(x_pos)
  ax_d.set_xticklabels(categories, fontsize=7.8, fontweight="bold")
  ax_d.set_ylabel("Normalized RMS Error / Reflection Ratio (Lower is Better)", fontsize=8.3)
  ax_d.set_title(
      "(D) Multi-Model Accuracy & Sponge Reflection Suppression (Marmousi-II + Held-Out Sub-Salt Fault)",
      fontsize=9.3,
      fontweight="bold",
  )
  ax_d.grid(True, axis="y", linestyle=":", alpha=0.5)
  ax_d.legend(loc="upper left", fontsize=7.6, framealpha=0.92)

  scorecard = (
      f"Seismic Benchmark Summary\n"
      f"-----------------------------------------\n"
      f"Metric              Baseline     Champion\n"
      f"Composite Score:     {base_score:7.2f}      {champ_score:7.2f}\n"
      f"Mean Trace NRMS:     {0.5*(b_bars[0]+b_bars[3]):7.3f}      {0.5*(c_bars[0]+c_bars[3]):7.3f}\n"
      f"Mean Interior NRMS:  {0.5*(b_bars[1]+b_bars[4]):7.3f}      {0.5*(c_bars[1]+c_bars[4]):7.3f}\n"
      f"Mean Late Refl:      {0.5*(b_bars[2]+b_bars[5]):7.3f}      {0.5*(c_bars[2]+c_bars[5]):7.3f}\n"
      f"Wall Time (ms):      {float(baseline_metrics['wall_time_ms']):7.1f}      {float(champion_metrics['wall_time_ms']):7.1f}\n"
      f"CFL E(1400)/E(700):  {float(baseline_metrics['late_energy_ratio']):7.4f}      {float(champion_metrics['late_energy_ratio']):7.4f}"
  )
  ax_d.text(
      0.53,
      0.66,
      scorecard,
      transform=ax_d.transAxes,
      fontsize=7.3,
      fontfamily="monospace",
      va="center",
      bbox=dict(
          boxstyle="round,pad=0.38",
          facecolor="#F1F5F9",
          edgecolor="#64748B",
          alpha=0.94,
      ),
  )

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=160)
  plt.close(fig)
  return out_path


def generate_seismic_propagation_gif(
    output_path: str | Path | None = None,
) -> Path:
  """Generates an animated GIF of 2D wavefield propagation (Baseline vs. Champion DRP)."""
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from matplotlib.patches import Rectangle  # pylint: disable=g-import-not-at-top
  from PIL import Image  # pylint: disable=g-import-not-at-top

  if output_path is None:
    output_path = SAMPLE_DIR / "seismic_wavefield_propagation.gif"
  out_path = Path(output_path)

  steps = tuple(range(45, 450, 30))
  sim = simulate_marmousi_wavefields(snapshot_steps=steps)
  vp = sim["vp"]
  nx, nz, n_pad = sim["nx"], sim["nz"], sim["n_pad"]
  dx = sim["dx"]
  extent_km = [0.0, nx * dx * 1e-3, nz * dx * 1e-3, 0.0]
  sponge_x0 = n_pad * dx * 1e-3
  sponge_z0 = n_pad * dx * 1e-3
  sponge_w = (nx - 2 * n_pad) * dx * 1e-3
  sponge_h = (nz - 2 * n_pad) * dx * 1e-3

  vmax = float(np.percentile(np.abs(sim["snaps_champ"][195]), 99.2)) * 1.1
  frames: list[Image.Image] = []

  for step in steps:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=110)
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(
        f"2D Acoustic Seismic Wave Propagation on Marmousi-II Salt Lens (t = {step} ms)\n"
        "Baseline 2nd-Order 5-Pt + Quadratic Sponge vs. Champion 8th-Order Isotropic 25-Pt DRP + Cubic Sponge",
        fontsize=10.5,
        fontweight="bold",
        y=0.96,
    )

    for ax, wf, title in [
        (ax1, sim["snaps_base"][step], "Baseline 2nd-Order 5-Pt (Dispersion & Boundary Refl)"),
        (ax2, sim["snaps_champ"][step], "Champion 8th-Order Isotropic 25-Pt DRP + Lax-Wendroff"),
    ]:
      ax.imshow(
          wf,
          extent=extent_km,
          cmap="seismic",
          vmin=-vmax,
          vmax=vmax,
          aspect="auto",
      )
      ax.contour(
          np.linspace(0.0, nx * dx * 1e-3, nx),
          np.linspace(0.0, nz * dx * 1e-3, nz),
          vp,
          levels=[2200.0, 2900.0, 4200.0],
          colors="#0F172A",
          linewidths=0.65,
          alpha=0.45,
      )
      ax.add_patch(
          Rectangle(
              (sponge_x0, sponge_z0),
              sponge_w,
              sponge_h,
              facecolor="none",
              edgecolor="#0F172A",
              linestyle="--",
              linewidth=1.1,
          )
      )
      ax.set_title(title, fontsize=9.0, fontweight="bold")
      ax.set_xlabel("Lateral Distance X (km)", fontsize=8.0)
      ax.set_ylabel("Depth Z (km)", fontsize=8.0)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))

  out_path.parent.mkdir(parents=True, exist_ok=True)
  frames[0].save(
      out_path,
      save_all=True,
      append_images=frames[1:],
      duration=160,
      loop=0,
  )
  return out_path


def postprocess(top_programs: list[Any] | None = None) -> None:
  """Post-processes top evolved seismic stencil programs and generates static/animated plots."""
  champ_metrics: Mapping[str, Any] = CHAMPION_METRICS_PAYLOAD
  if top_programs:
    best = top_programs[0]
    cand_dir = getattr(best, "candidate_dir", None) or (
        best.get("candidate_dir") if isinstance(best, Mapping) else None
    )
    if cand_dir:
      metrics_path = Path(cand_dir) / "seismic_metrics.json"
      if metrics_path.is_file():
        try:
          loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
          if "marmousi" in loaded and "subsalt_fault" in loaded:
            champ_metrics = loaded
        except (OSError, json.JSONDecodeError):
          pass

  visualize_seismic_comparison(
      baseline_metrics=BASELINE_METRICS_PAYLOAD,
      champion_metrics=champ_metrics,
      output_path=SAMPLE_DIR / "seismic_wavefield_stencil_comparison.png",
  )
  generate_seismic_propagation_gif(
      output_path=SAMPLE_DIR / "seismic_wavefield_propagation.gif",
  )


if __name__ == "__main__":
  print(json.dumps(evaluate(), indent=2))

