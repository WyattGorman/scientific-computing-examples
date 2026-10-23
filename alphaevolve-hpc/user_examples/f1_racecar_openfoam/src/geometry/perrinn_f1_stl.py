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
"""Watertight 6-region 3D STL generator & FIA 2017 legality checker for PERRINN F1.

Synthesizes a closed, 2-manifold multi-solid ASCII STL representing the half-car
(y >= 0, extending cleanly to y = -0.05 m across the y = 0 symmetry plane for
snappyHexMesh intersection) of the 2017 PERRINN Open-Source Formula 1 Car:
  1. `chassis`: Low 2017 FIA nose cone, survival cell, halo fairing, undercut
     sidepods, airbox, and Coanda coke-bottle engine cover.
  2. `underbody`: Front T-tray splitter, concave ground-effect Venturi tunnel,
     outer vortex-sealing floor skirt, FIA Art. 3.5 rear diffuser, Gurney lip,
     and twin longitudinal vortex strakes.
  3. `frontWing`: 2017 FIA delta-swept mainplane (with Y250 neutral central
     section), slotted upper cascade flap, and outwash-angled endplate.
  4. `rearWing`: 2017 FIA swept dual-element inverted rear wing with Gurney
     tabs, central Y150 diffuser-extraction monkey-seat, and full-depth endplate.
  5. `frontWheel`: 2017 FIA 305 mm x 670 mm Pirelli front tire with contact plinth.
  6. `rearWheel`: 2017 FIA 405 mm x 670 mm Pirelli rear tire with contact plinth.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REQUIRED_PARAMETERS: Tuple[str, ...] = (
    "front_ride_height_mm",
    "rear_ride_height_mm",
    "diffuser_angle_deg",
    "diffuser_strake_height_mm",
    "fw_flap_angle_deg",
    "fw_outwash_toe_deg",
    "rw_aoa_deg",
    "sidepod_undercut_ratio",
)

EXPECTED_SOLIDS: Tuple[str, ...] = (
    "chassis",
    "underbody",
    "frontWing",
    "rearWing",
    "frontWheel",
    "rearWheel",
)


def compute_fia_legality_metrics(
    params: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
  """Validates candidate parameters against bounds and FIA 2017 regulations."""
  if not isinstance(params, dict):
    return {"valid": False, "reason": "Candidate parameters must be a dict."}

  bounds = spec["design_bounds"]
  clean: Dict[str, float] = {}
  for key in REQUIRED_PARAMETERS:
    if key not in params:
      return {"valid": False, "reason": f"Missing required parameter '{key}'."}
    try:
      val = float(params[key])
    except (TypeError, ValueError):
      return {"valid": False, "reason": f"Non-numeric parameter '{key}'."}
    if not math.isfinite(val):
      return {"valid": False, "reason": f"Non-finite parameter '{key}'={val}."}
    lo, hi = float(bounds[key][0]), float(bounds[key][1])
    if val < lo - 1e-6 or val > hi + 1e-6:
      return {
          "valid": False,
          "reason": f"Parameter '{key}'={val:.4f} outside bounds [{lo}, {hi}].",
      }
    clean[key] = val

  rules = spec["fia_2017_guardrails"]
  h_f = clean["front_ride_height_mm"]
  h_r = clean["rear_ride_height_mm"]
  diff_deg = clean["diffuser_angle_deg"]
  strake_h = clean["diffuser_strake_height_mm"]
  undercut = clean["sidepod_undercut_ratio"]

  # 1. Dynamic plank bottoming & positive rake check
  min_hf = float(rules["min_front_plank_clearance_mm"])
  if h_f < min_hf:
    return {
        "valid": False,
        "reason": (
            f"Plank bottoming violation: front_ride_height_mm={h_f:.2f} mm < "
            f"{min_hf:.1f} mm dynamic FIA Titanium skid-block clearance."
        ),
    }

  min_rake_delta = float(rules["min_positive_rake_delta_mm"])
  if h_r < h_f + min_rake_delta:
    return {
        "valid": False,
        "reason": (
            f"Negative/flat rake hazard: rear_ride_height_mm={h_r:.2f} mm must "
            f"exceed front_ride_height_mm={h_f:.2f} mm by >= {min_rake_delta:.1f} mm."
        ),
    }

  # 2. FIA 2017 Art. 3.5.3 Rear Diffuser 175 mm Box Height Limit
  diff_len = float(rules["diffuser_expansion_length_mm"])
  step_off = float(rules["diffuser_step_offset_mm"])
  max_diff_box = float(rules["max_diffuser_box_height_mm"])
  diffuser_box_height_mm = step_off + diff_len * math.tan(math.radians(diff_deg))
  if diffuser_box_height_mm > max_diff_box:
    return {
        "valid": False,
        "reason": (
            f"FIA Art. 3.5.3 diffuser box violation: diffuser_box_height_mm="
            f"{diffuser_box_height_mm:.2f} mm > {max_diff_box:.1f} mm "
            f"(diffuser_angle_deg={diff_deg:.2f} deg)."
        ),
    }

  # 3. Diffuser longitudinal strake ground-scrape clearance
  min_strake_clear_limit = float(rules["min_strake_ground_clearance_mm"])
  z_ref_at_kick_mm = h_f + (3250.0 / 3475.0) * (h_r - h_f)
  strake_clearance_mm = z_ref_at_kick_mm + 10.0 - strake_h
  if strake_clearance_mm < min_strake_clear_limit:
    return {
        "valid": False,
        "reason": (
            f"Diffuser strake ground-strike violation: clearance="
            f"{strake_clearance_mm:.2f} mm < {min_strake_clear_limit:.1f} mm."
        ),
    }

  # 4. V6 Turbo-Hybrid + ERS Sidepod Radiator Cooling Duct Area
  min_rad_area = float(rules["min_sidepod_radiator_inlet_area_m2"])
  radiator_area_m2 = 0.145 * (1.0 - 0.82 * undercut)
  if radiator_area_m2 < min_rad_area:
    return {
        "valid": False,
        "reason": (
            f"Power-unit thermal starvation: radiator_inlet_area_m2="
            f"{radiator_area_m2:.4f} m^2 < {min_rad_area:.4f} m^2 "
            f"(sidepod_undercut_ratio={undercut:.3f})."
        ),
    }

  rake_angle_deg = math.degrees(math.atan2(h_r - h_f, 3475.0))
  return {
      "valid": True,
      "reason": "OK",
      "clean_parameters": clean,
      "rake_angle_deg": round(rake_angle_deg, 4),
      "diffuser_box_height_mm": round(diffuser_box_height_mm, 3),
      "strake_clearance_mm": round(strake_clearance_mm, 3),
      "radiator_inlet_area_m2": round(radiator_area_m2, 5),
  }


def _closed_loft_triangles(
    stations: Sequence[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Builds a closed, outward-oriented triangle mesh across cross-sections."""
  tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  m = len(stations)
  k_pts = stations[0].shape[0]
  for i in range(m - 1):
    a = stations[i]
    b = stations[i + 1]
    for k in range(k_pts):
      kn = (k + 1) % k_pts
      tris.append((a[k], a[kn], b[kn]))
      tris.append((a[k], b[kn], b[k]))

  c0 = np.mean(stations[0], axis=0)
  for k in range(k_pts):
    kn = (k + 1) % k_pts
    tris.append((c0, stations[0][kn], stations[0][k]))

  c1 = np.mean(stations[-1], axis=0)
  for k in range(k_pts):
    kn = (k + 1) % k_pts
    tris.append((c1, stations[-1][k], stations[-1][kn]))

  vol6 = sum(float(np.dot(np.cross(v0, v1), v2)) for v0, v1, v2 in tris)
  if vol6 < 0.0:
    tris = [(v0, v2, v1) for v0, v1, v2 in tris]
  return tris


def _airfoil_xz(
    chord: float,
    camber: float,
    thickness: float,
    aoa_deg: float,
    x_le: float,
    z_le: float,
    invert: bool = True,
    gurney_frac: float = 0.0,
    k_pts: int = 28,
) -> np.ndarray:
  """Generates a closed cambered NACA-4 airfoil section with optional Gurney lip."""
  half = k_pts // 2
  bu = np.linspace(0.0, math.pi, half + 1)[:-1]
  bl = np.linspace(math.pi, 0.0, half + 1)[:-1]
  pts = []
  rad = math.radians(aoa_deg)
  ca, sa = math.cos(rad), math.sin(rad)
  for is_up, betas in [(True, bu), (False, bl)]:
    for b in betas:
      xc = 0.5 * (1.0 - math.cos(b))
      yt = 5.0 * thickness * (
          0.2969 * math.sqrt(max(xc, 1e-6))
          - 0.1260 * xc
          - 0.3516 * xc**2
          + 0.2843 * xc**3
          - 0.1036 * xc**4
      )
      p = 0.45
      if xc < p:
        yc = (camber / p**2) * (2.0 * p * xc - xc**2)
      else:
        yc = (camber / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc**2)
      if invert:
        yc = -yc
      gurney = 0.0
      if invert and is_up and gurney_frac > 0.0 and xc > 0.86:
        gurney = gurney_frac * ((xc - 0.86) / 0.14) ** 1.5
      zl = (yc + (yt if is_up else -yt) + gurney) * chord
      xl = xc * chord
      if invert:
        xr = xl * ca - zl * sa
        zr = xl * sa + zl * ca
      else:
        xr = xl * ca + zl * sa
        zr = -xl * sa + zl * ca
      pts.append((x_le + xr, z_le + zr))
  return np.array(pts, dtype=float)


def _emit_solid(
    solid_name: str,
    tri_groups: Sequence[Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> str:
  """Serializes closed triangle groups into a named ASCII STL solid block."""
  lines = [f"solid {solid_name}"]
  for tris in tri_groups:
    for v0, v1, v2 in tris:
      n = np.cross(v1 - v0, v2 - v0)
      ln = float(np.linalg.norm(n))
      if ln < 1e-14:
        continue
      n = n / ln
      lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
      lines.append("    outer loop")
      lines.append(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}")
      lines.append(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}")
      lines.append(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}")
      lines.append("    endloop")
      lines.append("  endfacet")
  lines.append(f"endsolid {solid_name}")
  return "\n".join(lines)


def build_perrinn_f1_stl(
    params: Dict[str, float],
    stl_path: Path,
) -> Dict[str, Any]:
  """Builds the 6-region watertight 3D PERRINN 2017 F1 STL and returns stats."""
  h_f = float(params["front_ride_height_mm"]) * 1e-3
  h_r = float(params["rear_ride_height_mm"]) * 1e-3
  diff_deg = float(params["diffuser_angle_deg"])
  strake_h = float(params["diffuser_strake_height_mm"]) * 1e-3
  fw_flap_deg = float(params["fw_flap_angle_deg"])
  fw_toe_deg = float(params["fw_outwash_toe_deg"])
  rw_aoa_deg = float(params["rw_aoa_deg"])
  undercut = float(params["sidepod_undercut_ratio"])

  wb = 3.475

  def z_rake(x: float) -> float:
    t = float(np.clip(x / wb, -0.25, 1.15))
    return h_f + (h_r - h_f) * t

  solids_txt: List[str] = []
  solid_counts: Dict[str, int] = {}
  k_ch = 28
  angles = np.linspace(0.0, 2.0 * math.pi, k_ch, endpoint=False)

  # 1. CHASSIS
  x_ch = [-0.90, -0.55, -0.15, 0.28, 0.55, 0.95, 1.35, 1.80, 2.35, 2.95, 3.42, 3.78]
  ch_prof = [
      (0.065, 0.075, 0.130, 0.0),
      (0.125, 0.105, 0.245, 0.0),
      (0.210, 0.125, 0.395, 0.0),
      (0.255, 0.040, 0.495, 0.0),
      (0.290, 0.022, 0.535, 0.1),
      (0.330, 0.022, 0.560, 0.3),
      (0.660, 0.022, 0.610, 1.0),
      (0.630, 0.022, 0.820, 1.0),
      (0.480, 0.022, 0.720, 0.95),
      (0.260, 0.022, 0.610, 0.75),
      (0.135, 0.095, 0.530, 0.20),
      (0.080, 0.165, 0.460, 0.0),
  ]
  ch_stations = []
  for x, (ym, zb_rel, zt, uw) in zip(x_ch, ch_prof):
    zf = z_rake(x)
    zb = zf + zb_rel
    zc = 0.5 * (zb + zt)
    hz = 0.5 * (zt - zb)
    pts = []
    for th in angles:
      ct, st = math.cos(th), math.sin(th)
      y_norm = 0.5 * (1.0 + ct)
      ym_eff = ym
      if uw > 0.0 and st < 0.0:
        ym_eff = ym * (1.0 - 0.60 * uw * undercut * (-st))
      y = -0.05 + (ym_eff + 0.05) * (y_norm**0.88)
      z_shape = math.copysign(abs(st) ** (0.60 if st < 0 else 0.88), st)
      z = zc + hz * z_shape
      pts.append((x, y, z))
    ch_stations.append(np.array(pts, dtype=float))
  ch_tris = [_closed_loft_triangles(ch_stations)]
  solid_counts["chassis"] = sum(len(g) for g in ch_tris)
  solids_txt.append(_emit_solid("chassis", ch_tris))

  # 2. UNDERBODY
  x_ub = [0.22, 0.42, 0.72, 1.25, 1.95, 2.55, 2.95, 3.30, 3.60, 3.825]
  ub_stations = []
  diff_rad = math.radians(diff_deg)
  for x in x_ub:
    zf = z_rake(x)
    bellmouth = (
        0.018 * max(0.0, (0.55 - x) / 0.33) ** 1.5 if x < 0.55 else 0.0
    )
    if x <= 2.55:
      diff_rise = 0.0
      ym = 0.42 if x < 0.50 else (0.64 if x < 0.80 else 0.70)
      thick = 0.054
    elif x <= 3.30:
      frac_pre = (x - 2.55) / 0.75
      diff_rise = (
          0.040
          * (frac_pre**1.5)
          * (math.tan(diff_rad) / math.tan(math.radians(13.0)))
      )
      ym = 0.70
      thick = 0.052
    else:
      pre_end = 0.040 * (math.tan(diff_rad) / math.tan(math.radians(13.0)))
      dx_d = x - 3.30
      diff_rise = pre_end + dx_d * math.tan(diff_rad)
      ym = 0.70 - 0.16 * min(1.0, dx_d / 0.22)
      thick = 0.036

    gurney_deck = 0.028 * ((x - 3.50) / 0.325) ** 1.6 if x > 3.50 else 0.0
    pts = []
    for th in angles:
      ct, st = math.cos(th), math.sin(th)
      y_norm = 0.5 * (1.0 + ct)
      y = -0.05 + (ym + 0.05) * y_norm
      ye = max(0.0, y)
      tunnel_arch = 0.0
      if 0.07 < ye < ym - 0.04 and x > 0.48:
        eta = (ye - 0.07) / max(0.10, (ym - 0.11))
        tunnel_arch = (0.024 + 0.022 * undercut) * math.sin(math.pi * eta) ** 1.15
        if x > 2.55:
          tunnel_arch += 0.50 * diff_rise * math.sin(math.pi * eta) ** 0.8
      edge_skirt = -0.013 * (y_norm**5) if x >= 0.72 else 0.0

      z_bot_local = (
          zf
          + bellmouth
          + diff_rise
          * (0.68 + 0.32 * math.sin(0.5 * math.pi * min(1.0, ye / 0.45)))
          + tunnel_arch
          + edge_skirt
      )
      z_top_local = max(
          z_bot_local + 0.024, zf + thick + 0.35 * diff_rise + gurney_deck
      )
      if st >= 0.0:
        z = 0.5 * (z_bot_local + z_top_local) + 0.5 * (
            z_top_local - z_bot_local
        ) * (st**0.55)
      else:
        z = 0.5 * (z_bot_local + z_top_local) - 0.5 * (
            z_top_local - z_bot_local
        ) * ((-st) ** 0.45)
      pts.append((x, y, z))
    ub_stations.append(np.array(pts, dtype=float))
  ub_tris = [_closed_loft_triangles(ub_stations)]

  for y_st in [0.20, 0.38]:
    st_stations = []
    for x in [2.70, 3.25, 3.55, 3.81]:
      zf = z_rake(x)
      dx_d = max(0.0, x - 3.30)
      z_roof = zf + 0.03 + dx_d * math.tan(diff_rad)
      z_bot = max(0.014, z_roof - strake_h - 0.020)
      zc = 0.5 * (z_bot + z_roof)
      hz = 0.5 * (z_roof - z_bot)
      pts = [
          (x, y_st + 0.012 * math.cos(th), zc + hz * math.sin(th))
          for th in angles
      ]
      st_stations.append(np.array(pts, dtype=float))
    ub_tris.append(_closed_loft_triangles(st_stations))
  solid_counts["underbody"] = sum(len(g) for g in ub_tris)
  solids_txt.append(_emit_solid("underbody", ub_tris))

  # 3. FRONT WING
  fw_tris = []
  ys_fw = np.linspace(-0.05, 0.90, 12)
  main_st, flap_st = [], []
  for y in ys_fw:
    ye = max(0.0, float(y))
    s_frac = ye / 0.90
    sweep_x = 0.16 * (s_frac**1.15)
    y250_blend = float(np.clip((ye - 0.18) / 0.10, 0.0, 1.0))
    mp_xz = _airfoil_xz(
        chord=0.27,
        camber=0.035 + 0.065 * y250_blend,
        thickness=0.13,
        aoa_deg=(
            4.0
            + (4.0 + 0.25 * fw_flap_deg) * y250_blend
            - 0.25 * fw_toe_deg * (s_frac**2)
        ),
        x_le=-0.95 + sweep_x,
        z_le=h_f + 0.026 + 0.018 * s_frac,
        invert=True,
        gurney_frac=0.03 * y250_blend,
        k_pts=28,
    )
    main_st.append(np.column_stack([mp_xz[:, 0], np.full(28, y), mp_xz[:, 1]]))

    fl_xz = _airfoil_xz(
        chord=0.22,
        camber=0.045 + 0.095 * y250_blend,
        thickness=0.13,
        aoa_deg=(
            5.0
            + (fw_flap_deg - 4.0) * y250_blend
            - 0.35 * fw_toe_deg * (s_frac**1.5)
        ),
        x_le=-0.68 + sweep_x,
        z_le=h_f + 0.076 + 0.028 * s_frac,
        invert=True,
        gurney_frac=0.045 * y250_blend,
        k_pts=28,
    )
    flap_st.append(np.column_stack([fl_xz[:, 0], np.full(28, y), fl_xz[:, 1]]))
  fw_tris.append(_closed_loft_triangles(main_st))
  fw_tris.append(_closed_loft_triangles(flap_st))

  toe_rad = math.radians(fw_toe_deg)
  ep_stations = []
  for z_ep in np.linspace(h_f + 0.016, h_f + 0.25, 6):
    pts = []
    for th in angles:
      xl = 0.25 * math.cos(th)
      yl = 0.016 * math.sin(th)
      xr = -0.54 + xl * math.cos(toe_rad) - yl * math.sin(toe_rad)
      yr = 0.895 + xl * math.sin(toe_rad) + yl * math.cos(toe_rad)
      pts.append((xr, yr, float(z_ep)))
    ep_stations.append(np.array(pts, dtype=float))
  fw_tris.append(_closed_loft_triangles(ep_stations))
  solid_counts["frontWing"] = sum(len(g) for g in fw_tris)
  solids_txt.append(_emit_solid("frontWing", fw_tris))

  # 4. REAR WING
  rw_tris = []
  ys_rw = np.linspace(-0.05, 0.475, 9)
  rw_main_st, rw_flap_st = [], []
  mp_aoa = 0.54 * rw_aoa_deg
  mp_te_z = 0.71 + 0.34 * math.sin(math.radians(mp_aoa))
  mp_te_x = 3.55 + 0.34 * math.cos(math.radians(mp_aoa))
  for y in ys_rw:
    ye = max(0.0, float(y))
    sf = ye / 0.475
    rw_mp = _airfoil_xz(
        chord=0.34,
        camber=0.115,
        thickness=0.13,
        aoa_deg=mp_aoa,
        x_le=3.55 + 0.03 * sf,
        z_le=0.71,
        invert=True,
        gurney_frac=0.025,
        k_pts=28,
    )
    rw_main_st.append(
        np.column_stack([rw_mp[:, 0], np.full(28, y), rw_mp[:, 1]])
    )

    rw_fl = _airfoil_xz(
        chord=0.21,
        camber=0.135,
        thickness=0.13,
        aoa_deg=rw_aoa_deg,
        x_le=mp_te_x - 0.015 + 0.03 * sf,
        z_le=mp_te_z + 0.046,
        invert=True,
        gurney_frac=0.05,
        k_pts=28,
    )
    rw_flap_st.append(
        np.column_stack([rw_fl[:, 0], np.full(28, y), rw_fl[:, 1]])
    )
  rw_tris.append(_closed_loft_triangles(rw_main_st))
  rw_tris.append(_closed_loft_triangles(rw_flap_st))

  ms_st = []
  for y in np.linspace(-0.05, 0.16, 5):
    ms_xz = _airfoil_xz(
        chord=0.18,
        camber=0.11,
        thickness=0.14,
        aoa_deg=0.68 * rw_aoa_deg,
        x_le=3.68,
        z_le=0.47,
        invert=True,
        gurney_frac=0.03,
        k_pts=28,
    )
    ms_st.append(np.column_stack([ms_xz[:, 0], np.full(28, y), ms_xz[:, 1]]))
  rw_tris.append(_closed_loft_triangles(ms_st))

  rw_ep_st = []
  for z_ep in np.linspace(0.35, 0.95, 7):
    sweep_back = 0.07 * ((float(z_ep) - 0.35) / 0.60)
    pts = [
        (
            3.78 + sweep_back + 0.27 * math.cos(th),
            0.475 + 0.016 * math.sin(th),
            float(z_ep),
        )
        for th in angles
    ]
    rw_ep_st.append(np.array(pts, dtype=float))
  rw_tris.append(_closed_loft_triangles(rw_ep_st))
  solid_counts["rearWing"] = sum(len(g) for g in rw_tris)
  solids_txt.append(_emit_solid("rearWing", rw_tris))

  # 5 & 6. FRONT & REAR WHEELS
  for solid_name, x_axle, y_in, y_out in [
      ("frontWheel", 0.000, 0.695, 1.000),
      ("rearWheel", 3.475, 0.595, 1.000),
  ]:
    r_tire = 0.335
    z_hub = 0.335
    ys_w = np.linspace(y_in, y_out, 7)
    w_stations = []
    k_w = 32
    w_ang = np.linspace(0.0, 2.0 * math.pi, k_w, endpoint=False)
    for y in ys_w:
      edge_dist = min(float(y) - y_in, y_out - float(y))
      bevel = 0.94 if edge_dist < 1e-4 else (0.985 if edge_dist < 0.06 else 1.0)
      r_loc = r_tire * bevel
      pts = []
      for th in w_ang:
        x_p = x_axle + r_loc * math.cos(th)
        z_p = z_hub + r_loc * math.sin(th)
        if z_p < 0.022:
          z_p = -0.020
        pts.append((x_p, float(y), z_p))
      w_stations.append(np.array(pts, dtype=float))
    w_tris = [_closed_loft_triangles(w_stations)]
    solid_counts[solid_name] = sum(len(g) for g in w_tris)
    solids_txt.append(_emit_solid(solid_name, w_tris))

  stl_path.parent.mkdir(parents=True, exist_ok=True)
  stl_path.write_text("\n".join(solids_txt) + "\n", encoding="utf-8")
  return {
      "stl_path": str(stl_path),
      "solid_triangle_counts": solid_counts,
      "total_triangles": int(sum(solid_counts.values())),
  }


def verify_stl_watertight(stl_path: Path) -> Dict[str, Any]:
  """Pure-Python 2-manifold edge pairing & signed-volume check on ASCII STL."""
  if not stl_path.is_file():
    return {"watertight": False, "reason": f"Missing STL file: {stl_path}"}

  text = stl_path.read_text(encoding="utf-8")
  solids_found: List[str] = []
  current_solid: str | None = None
  edge_counts: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
  vol6 = 0.0
  verts_buf: List[Tuple[float, float, float]] = []
  total_facets = 0

  def _qkey(v: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (int(round(v[0] * 1e5)), int(round(v[1] * 1e5)), int(round(v[2] * 1e5)))

  for raw in text.splitlines():
    line = raw.strip()
    if line.startswith("solid "):
      current_solid = line.split(None, 1)[1]
      solids_found.append(current_solid)
      edge_counts.clear()
      vol6 = 0.0
    elif line.startswith("vertex "):
      parts = line.split()
      verts_buf.append((float(parts[1]), float(parts[2]), float(parts[3])))
      if len(verts_buf) == 3:
        v0, v1, v2 = verts_buf
        verts_buf.clear()
        total_facets += 1
        k0, k1, k2 = _qkey(v0), _qkey(v1), _qkey(v2)
        for a, b in ((k0, k1), (k1, k2), (k2, k0)):
          edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1
        vol6 += (
            (v0[1] * v1[2] - v0[2] * v1[1]) * v2[0]
            + (v0[2] * v1[0] - v0[0] * v1[2]) * v2[1]
            + (v0[0] * v1[1] - v0[1] * v1[0]) * v2[2]
        )
    elif line.startswith("endsolid"):
      unpaired = sum(
          1
          for (a, b), cnt in edge_counts.items()
          if edge_counts.get((b, a), 0) != cnt
      )
      if unpaired > 0:
        return {
            "watertight": False,
            "reason": f"Solid '{current_solid}' has {unpaired} unpaired edges.",
        }
      if vol6 <= 0.0:
        return {
            "watertight": False,
            "reason": f"Solid '{current_solid}' has non-positive volume.",
        }

  missing = [s for s in EXPECTED_SOLIDS if s not in solids_found]
  if missing:
    return {
        "watertight": False,
        "reason": f"Missing required STL solids: {missing}",
    }

  return {
      "watertight": True,
      "solids": solids_found,
      "total_facets": total_facets,
  }
