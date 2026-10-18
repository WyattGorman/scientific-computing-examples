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

"""OpenFOAM-14 2D incompressible airfoil CFD evaluator (airFoil2D tutorial).

Generates a structured 6-block C-grid mesh (`system/blockMeshDict`) with curved
B-spline edges around a 6-parameter Kulfan CST airfoil, validates mesh quality
with `blockMesh` and `checkMesh`, and runs the OpenFOAM-14 `incompressibleFluid`
steady-state RANS solver (`foamRun` / `simpleFoam`) with the Spalart-Allmaras
turbulence model matching `tutorials/incompressibleFluid/airFoil2D`.
"""

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Sequence
import numpy as np

from src.kulfan_converter.kulfan_to_coord import CST_shape, _compute_surface_y

logger = logging.getLogger(__name__)

# Default flow conditions from OpenFOAM-14 tutorials/incompressibleFluid/airFoil2D:
# U = (25.75, 3.62, 0) m/s -> |U| = 26.0029 m/s, AoA = arctan2(3.62, 25.75) = 8.0024 deg
DEFAULT_FLUID_VELOCITY = np.array([25.75, 3.62, 0.0], dtype=float)
KINEMATIC_VISCOSITY = 1.0e-5  # m^2/s (Re ~ 2.6e6 for 1 m chord)
NU_TILDA_INF = 0.14  # m^2/s (Spalart-Allmaras freestream turbulence variable)
NUT_INF = 0.14  # m^2/s (freestream turbulent eddy viscosity)
MAX_SIMULATIONS_PER_EVAL = 16  # Max successful OpenFOAM simulations per candidate program
MAX_TOTAL_ATTEMPTS_PER_EVAL = 100  # Max total calls to funct() including pre-screen rejections

# Structural thickness constraints (normalized by unit chord c = 1.0 m)
MIN_INTERIOR_THICKNESS = 0.002  # >= 0.2% chord minimum interior thickness (x/c in [0.01, 0.99])
MIN_MAX_THICKNESS = 0.08        # >= 8.0% chord minimum structural wing-box thickness
MAX_MAX_THICKNESS = 0.24        # <= 24.0% chord maximum structural thickness

# Allowed parameter bounds for the 6 Kulfan CST airfoil shape weights
CST_BOUNDS = [
    (-0.35, -0.05),
    (-0.30, 0.20),
    (-0.30, 0.25),
    (0.08, 0.50),
    (0.05, 0.55),
    (0.05, 0.50),
]


@dataclass
class Parameters:
  """Configuration parameters and telemetry tracker for an OpenFOAM evaluation run."""

  run_name: str
  cases_folder: Path
  is_debug: bool
  csv_path: Path
  fluid_velocity: np.ndarray = field(
      default_factory=lambda: DEFAULT_FLUID_VELOCITY.copy()
  )
  end_time: int = 500
  sim_count: int = 0
  successful_sim_count: int = 0
  max_successful_sims: int = MAX_SIMULATIONS_PER_EVAL
  max_total_sims: int = MAX_TOTAL_ATTEMPTS_PER_EVAL
  allow_multiple_evals: bool = True
  last_diagnostics: dict[str, Any] = field(default_factory=dict)
  best_x: np.ndarray | None = None
  best_score: float = float("inf")
  best_diagnostics: dict[str, Any] = field(default_factory=dict)
  eval_history: list[dict[str, Any]] = field(default_factory=list)
  _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class GeometryCheckResult(tuple):
  """A `(is_valid, metrics_dict)` tuple whose truthiness is `bool(is_valid)`.

  Allows callers to use either `if check_geometry(x):` or
  `is_valid, metrics = check_geometry(x)`.
  """

  def __bool__(self) -> bool:
    return bool(self[0])


def check_geometry(x: np.ndarray) -> GeometryCheckResult:
  """Fast analytical pre-screening of 6 Kulfan CST airfoil parameters.

  Verifies that:
    1. `x` is a finite 6-element vector within `CST_BOUNDS`.
    2. Minimum interior thickness `y_u(x) - y_l(x)` on `x/c in [0.01, 0.99]` is `>= 0.005`.
    3. Maximum thickness lies within `[0.08, 0.24]` (`8% <= t/c <= 24%`).

  Returns:
    `GeometryCheckResult` tuple of `(is_valid, metrics_dict)` with `min_thickness`,
    `max_thickness`, `max_camber`, where `bool(check_geometry(x)) == is_valid`.
  """
  x_arr = np.asarray(x, dtype=float).flatten()
  if x_arr.shape != (6,) or not np.all(np.isfinite(x_arr)):
    return GeometryCheckResult(
        (False, {"min_thickness": 0.0, "max_thickness": 0.0, "max_camber": 0.0})
    )

  in_bounds = all(
      low - 1e-5 <= float(val) <= high + 1e-5
      for val, (low, high) in zip(x_arr, CST_BOUNDS)
  )
  wl = x_arr[0:3]
  wu = x_arr[3:6]

  x_check = np.linspace(0.01, 0.99, 100)
  yu_check = _compute_surface_y(x_check, wu, 0.0)
  yl_check = _compute_surface_y(x_check, wl, 0.0)
  thickness = yu_check - yl_check
  camber = 0.5 * (yu_check + yl_check)
  min_thickness = float(np.min(thickness))
  max_thickness = float(np.max(thickness))
  max_camber = float(np.max(np.abs(camber)))

  is_valid = bool(
      in_bounds
      and min_thickness >= MIN_INTERIOR_THICKNESS
      and MIN_MAX_THICKNESS <= max_thickness <= MAX_MAX_THICKNESS
  )
  return GeometryCheckResult((
      is_valid,
      {
          "min_thickness": min_thickness,
          "max_thickness": max_thickness,
          "max_camber": max_camber,
      },
  ))


def _log_to_csv(
    csv_path: Path,
    run_name: str,
    no_clipping: bool,
    block_mesh: bool,
    check_mesh: bool,
    simple: bool,
    cd: float,
    cl: float,
) -> None:
  """Appends simulation metrics and stage statuses to the tracking CSV file."""
  file_exists = csv_path.exists()
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow([
          "run_name",
          "no_clipping",
          "block_mesh",
          "check_mesh",
          "simple",
          "cd",
          "cl",
      ])
    writer.writerow([
        run_name,
        no_clipping,
        block_mesh,
        check_mesh,
        simple,
        f"{cd:.6f}",
        f"{cl:.6f}",
    ])


def is_openfoam_available() -> bool:
  """Checks whether OpenFOAM (`blockMesh` and `foamRun`/`simpleFoam`) is installed."""
  if Path("/opt/openfoam14/etc/bashrc").exists():
    return True
  return shutil.which("blockMesh") is not None


def _run_openfoam_cmd(
    cmd: str, cwd: Path, timeout_sec: int = 300
) -> subprocess.CompletedProcess[str]:
  """Executes an OpenFOAM command inside bash after sourcing OpenFOAM's bashrc."""
  bashrc_candidates = [
      "/opt/openfoam14/etc/bashrc",
      "/opt/openfoam13/etc/bashrc",
      "/opt/openfoam12/etc/bashrc",
      "/opt/openfoam11/etc/bashrc",
      "/usr/lib/openfoam/openfoam2406/etc/bashrc",
      "/usr/lib/openfoam/openfoam2312/etc/bashrc",
  ]
  source_prefix = ""
  for rc in bashrc_candidates:
    if Path(rc).exists():
      source_prefix = f"source {rc} >/dev/null 2>&1 || true; "
      break

  full_cmd = f"{source_prefix}set -o pipefail; {cmd}"
  return subprocess.run(
      ["/bin/bash", "-c", full_cmd],
      cwd=str(cwd),
      capture_output=True,
      text=True,
      timeout=timeout_sec,
      check=False,
  )


def _format_spline_points(
    xs: np.ndarray, ys: np.ndarray, z_val: float
) -> str:
  """Formats a list of 3D spline control points for OpenFOAM blockMeshDict."""
  lines = [
      f"            ({x_pt:.8f} {y_pt:.8f} {z_val:.8f})"
      for x_pt, y_pt in zip(xs, ys)
  ]
  return "\n".join(lines)


def _write_block_mesh_dict(
    case_dir: Path, wl: np.ndarray, wu: np.ndarray
) -> None:
  """Writes a structured 6-block C-grid `system/blockMeshDict` with airfoil splines."""
  x_m = 0.30
  r_far = 8.0
  x_wake = 12.0
  z0 = -0.5
  z1 = 0.5

  y_u_m = float(_compute_surface_y(np.array([x_m]), wu, 0.0)[0])
  y_l_m = float(_compute_surface_y(np.array([x_m]), wl, 0.0)[0])

  xy_coords = [
      (x_wake, r_far),      # 0 / 12: top-right wake outlet
      (x_wake, 0.0),        # 1 / 13: wake centerline outlet
      (x_wake, -r_far),     # 2 / 14: bottom-right wake outlet
      (1.0, -r_far),        # 3 / 15: bottom under trailing edge
      (x_m, -r_far),        # 4 / 16: bottom under split point (C-arc start)
      (x_m - r_far, 0.0),   # 5 / 17: upstream farfield nose on x-axis
      (x_m, r_far),         # 6 / 18: top above split point (C-arc end)
      (1.0, r_far),         # 7 / 19: top above trailing edge
      (0.0, 0.0),           # 8 / 20: airfoil Leading Edge (LE)
      (x_m, y_l_m),         # 9 / 21: airfoil lower surface split point
      (1.0, 0.0),           # 10 / 22: airfoil Trailing Edge (TE)
      (x_m, y_u_m),         # 11 / 23: airfoil upper surface split point
  ]

  vertices_lines = []
  for idx, (xv, yv) in enumerate(xy_coords):
    vertices_lines.append(f"    ({xv:.8f} {yv:.8f} {z0:.8f}) // {idx}")
  for idx, (xv, yv) in enumerate(xy_coords):
    vertices_lines.append(f"    ({xv:.8f} {yv:.8f} {z1:.8f}) // {idx + 12}")

  n_nose_pts = 30
  theta_nose = np.linspace(0.0, 0.5 * math.pi, n_nose_pts + 2)[1:-1]
  x_nose = x_m * (1.0 - np.cos(theta_nose))
  yu_nose = _compute_surface_y(x_nose, wu, 0.0)
  yl_nose = _compute_surface_y(x_nose, wl, 0.0)

  n_rear_pts = 40
  theta_rear = np.linspace(0.0, math.pi, n_rear_pts + 2)[1:-1]
  x_rear = x_m + 0.5 * (1.0 - x_m) * (1.0 - np.cos(theta_rear))
  yu_rear = _compute_surface_y(x_rear, wu, 0.0)
  yl_rear = _compute_surface_y(x_rear, wl, 0.0)

  arc_x = x_m - r_far * math.cos(0.25 * math.pi)
  arc_y_up = r_far * math.sin(0.25 * math.pi)
  arc_y_low = -r_far * math.sin(0.25 * math.pi)

  n_nose = 30
  n_rear = 45
  n_wake = 50
  n_radial = 40
  g_radial = 180.0
  g_wake = 15.0
  inv_g_wake = 1.0 / g_wake

  block_mesh_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  14
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale 1;

vertices
(
{chr(10).join(vertices_lines)}
);

blocks
(
    // Block 0: Upper front nose (LE 8 -> Upper split 11, radial outward to 5 -> 6)
    hex (8 11 6 5 20 23 18 17) ({n_nose} {n_radial} 1) simpleGrading (1 {g_radial} 1)
    // Block 1: Upper rear airfoil (Upper split 11 -> TE 10, radial outward to 6 -> 7)
    hex (11 10 7 6 23 22 19 18) ({n_rear} {n_radial} 1) simpleGrading (1 {g_radial} 1)
    // Block 2: Upper wake (TE 10 -> Outlet 1, radial outward to 7 -> 0)
    hex (10 1 0 7 22 13 12 19) ({n_wake} {n_radial} 1) simpleGrading ({g_wake} {g_radial} 1)
    // Block 3: Lower front nose (Lower split 9 -> LE 8, radial outward to 4 -> 5)
    hex (9 8 5 4 21 20 17 16) ({n_nose} {n_radial} 1) simpleGrading (1 {g_radial} 1)
    // Block 4: Lower rear airfoil (TE 10 -> Lower split 9, radial outward to 3 -> 4)
    hex (10 9 4 3 22 21 16 15) ({n_rear} {n_radial} 1) simpleGrading (1 {g_radial} 1)
    // Block 5: Lower wake (Outlet 1 -> TE 10, radial outward to 2 -> 3)
    hex (1 10 3 2 13 22 15 14) ({n_wake} {n_radial} 1) simpleGrading ({inv_g_wake:.8f} {g_radial} 1)
);

edges
(
    // Upstream C-grid circular arcs
    arc 5 6 ({arc_x:.8f} {arc_y_up:.8f} {z0:.8f})
    arc 4 5 ({arc_x:.8f} {arc_y_low:.8f} {z0:.8f})
    arc 17 18 ({arc_x:.8f} {arc_y_up:.8f} {z1:.8f})
    arc 16 17 ({arc_x:.8f} {arc_y_low:.8f} {z1:.8f})

    // Airfoil upper nose splines (8 -> 11 and 20 -> 23)
    spline 8 11
    (
{_format_spline_points(x_nose, yu_nose, z0)}
    )
    spline 20 23
    (
{_format_spline_points(x_nose, yu_nose, z1)}
    )

    // Airfoil upper rear splines (11 -> 10 and 23 -> 22)
    spline 11 10
    (
{_format_spline_points(x_rear, yu_rear, z0)}
    )
    spline 23 22
    (
{_format_spline_points(x_rear, yu_rear, z1)}
    )

    // Airfoil lower nose splines (8 -> 9 and 20 -> 21)
    spline 8 9
    (
{_format_spline_points(x_nose, yl_nose, z0)}
    )
    spline 20 21
    (
{_format_spline_points(x_nose, yl_nose, z1)}
    )

    // Airfoil lower rear splines (9 -> 10 and 21 -> 22)
    spline 9 10
    (
{_format_spline_points(x_rear, yl_rear, z0)}
    )
    spline 21 22
    (
{_format_spline_points(x_rear, yl_rear, z1)}
    )
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (5 6 18 17)
            (6 7 19 18)
            (4 5 17 16)
            (3 4 16 15)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (7 0 12 19)
            (0 1 13 12)
            (1 2 14 13)
            (2 3 15 14)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (8 11 23 20)
            (11 10 22 23)
            (9 8 20 21)
            (10 9 21 22)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (8 11 6 5)
            (11 10 7 6)
            (10 1 0 7)
            (9 8 5 4)
            (10 9 4 3)
            (1 10 3 2)
            (20 17 18 23)
            (23 18 19 22)
            (22 19 12 13)
            (21 16 17 20)
            (22 15 16 21)
            (13 14 15 22)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
  system_dir = case_dir / "system"
  system_dir.mkdir(parents=True, exist_ok=True)
  (system_dir / "blockMeshDict").write_text(block_mesh_dict, encoding="utf-8")


def _write_openfoam_case_files(
    case_dir: Path, fluid_velocity: np.ndarray, end_time: int
) -> None:
  """Writes the OpenFOAM-14 `airFoil2D` 0/, constant/, and system/ files."""
  zero_dir = case_dir / "0"
  const_dir = case_dir / "constant"
  sys_dir = case_dir / "system"
  zero_dir.mkdir(parents=True, exist_ok=True)
  const_dir.mkdir(parents=True, exist_ok=True)
  sys_dir.mkdir(parents=True, exist_ok=True)

  ux = float(fluid_velocity[0])
  uy = float(fluid_velocity[1])
  uz = float(fluid_velocity[2]) if len(fluid_velocity) > 2 else 0.0
  u_mag = float(math.hypot(ux, uy))
  if u_mag < 1e-8:
    raise ValueError(f"Fluid velocity magnitude is zero: {fluid_velocity}")

  drag_x = ux / u_mag
  drag_y = uy / u_mag
  lift_x = -uy / u_mag
  lift_y = ux / u_mag

  # 0/U
  (zero_dir / "U").write_text(
      f"""FoamFile
{{
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({ux:.6f} {uy:.6f} {uz:.6f});
boundaryField
{{
    inlet
    {{
        type            freestreamVelocity;
        freestreamValue $internalField;
    }}
    outlet
    {{
        type            freestreamVelocity;
        freestreamValue $internalField;
    }}
    walls
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
      encoding="utf-8",
  )

  # 0/p
  (zero_dir / "p").write_text(
      """FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type            freestreamPressure;
        freestreamValue $internalField;
    }
    outlet
    {
        type            freestreamPressure;
        freestreamValue $internalField;
    }
    walls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
""",
      encoding="utf-8",
  )

  # 0/nut
  (zero_dir / "nut").write_text(
      f"""FoamFile
{{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nut;
}}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {NUT_INF};
boundaryField
{{
    inlet
    {{
        type            freestream;
        freestreamValue $internalField;
    }}
    outlet
    {{
        type            freestream;
        freestreamValue $internalField;
    }}
    walls
    {{
        type            nutUSpaldingWallFunction;
        value           uniform 0;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
      encoding="utf-8",
  )

  # 0/nuTilda
  (zero_dir / "nuTilda").write_text(
      f"""FoamFile
{{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nuTilda;
}}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {NU_TILDA_INF};
boundaryField
{{
    inlet
    {{
        type            freestream;
        freestreamValue $internalField;
    }}
    outlet
    {{
        type            freestream;
        freestreamValue $internalField;
    }}
    walls
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
      encoding="utf-8",
  )

  # constant/momentumTransport
  (const_dir / "momentumTransport").write_text(
      """FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      momentumTransport;
}
simulationType  RAS;
RAS
{
    model           SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;
}
""",
      encoding="utf-8",
  )

  # constant/physicalProperties
  (const_dir / "physicalProperties").write_text(
      f"""FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      physicalProperties;
}}
viscosityModel  constant;
rho             1;
nu              {KINEMATIC_VISCOSITY:.6e};
""",
      encoding="utf-8",
  )

  # constant/transportProperties
  (const_dir / "transportProperties").write_text(
      f"""FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {KINEMATIC_VISCOSITY:.6e};
""",
      encoding="utf-8",
  )

  # system/fvSchemes
  (sys_dir / "fvSchemes").write_text(
      """FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
ddtSchemes
{
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         cellMDLimited Gauss linear 1;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,nuTilda) bounded Gauss linearUpwind grad(nuTilda);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear limited corrected 0.5;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         limited corrected 0.5;
}
wallDist
{
    method          meshWave;
}
""",
      encoding="utf-8",
  )

  # system/fvSolution
  (sys_dir / "fvSolution").write_text(
      """FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }
    "(U|nuTilda)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
    residualControl
    {
        p               1e-5;
        U               1e-5;
        "(k|epsilon|omega|nuTilda)" 1e-5;
    }
}
relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        nuTilda         0.7;
    }
}
""",
      encoding="utf-8",
  )

  # system/controlDict
  (sys_dir / "controlDict").write_text(
      f"""FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
solver          incompressibleFluid;
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {int(end_time)};
deltaT          1;
writeControl    timeStep;
writeInterval   {int(end_time)};
purgeWrite      1;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forceCoeffs1
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        log             yes;
        patches         (walls);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0.25 0 0);
        liftDir         ({lift_x:.8f} {lift_y:.8f} 0);
        dragDir         ({drag_x:.8f} {drag_y:.8f} 0);
        pitchAxis       (0 0 1);
        magUInf         {u_mag:.8f};
        lRef            1.0;
        Aref            1.0;
    }}
}}
""",
      encoding="utf-8",
  )


def _parse_force_coeffs(case_dir: Path) -> dict[str, Any]:
  """Parses converged Cl, Cd, Cm from postProcessing/forceCoeffs1/0/*.dat."""
  post_dir = case_dir / "postProcessing" / "forceCoeffs1" / "0"
  if not post_dir.exists():
    return {
        "converged": False,
        "reason": "No postProcessing/forceCoeffs1/0 directory produced.",
    }

  dat_files = sorted(post_dir.glob("*.dat"))
  if not dat_files:
    return {
        "converged": False,
        "reason": "No .dat file found in postProcessing/forceCoeffs1/0.",
    }

  dat_file = dat_files[0]
  col_map = {"Time": 0, "Cm": 1, "Cd": 2, "Cl": 3}
  rows: list[tuple[float, float, float, float]] = []

  for raw_line in dat_file.read_text(
      encoding="utf-8", errors="replace"
  ).splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if line.startswith("#"):
      header_tokens = line.lstrip("#").split()
      if (
          "Time" in header_tokens
          and "Cd" in header_tokens
          and "Cl" in header_tokens
      ):
        col_map = {tok: i for i, tok in enumerate(header_tokens)}
      continue
    parts = line.split()
    try:
      t_val = float(parts[col_map.get("Time", 0)])
      cm_idx = col_map.get("Cm", col_map.get("CmPitch", 1))
      cd_idx = col_map.get("Cd", 2)
      cl_idx = col_map.get("Cl", 3)
      cm_val = float(parts[cm_idx])
      cd_val = float(parts[cd_idx])
      cl_val = float(parts[cl_idx])
      if (
          math.isfinite(t_val)
          and math.isfinite(cm_val)
          and math.isfinite(cd_val)
          and math.isfinite(cl_val)
      ):
        rows.append((t_val, cm_val, cd_val, cl_val))
    except (ValueError, IndexError):
      continue

  if len(rows) < 30:
    return {
        "converged": False,
        "iterations": len(rows),
        "reason": f"Solver terminated early after only {len(rows)} iterations.",
    }

  tail = rows[-min(20, len(rows)) :]
  cm_vals = np.array([r[1] for r in tail], dtype=float)
  cd_vals = np.array([r[2] for r in tail], dtype=float)
  cl_vals = np.array([r[3] for r in tail], dtype=float)

  cm_mean = float(np.mean(cm_vals))
  cd_mean = float(np.mean(cd_vals))
  cl_mean = float(np.mean(cl_vals))
  cl_std = float(np.std(cl_vals))
  cd_std = float(np.std(cd_vals))

  if cd_mean <= 1.0e-4 or cd_mean > 2.0 or abs(cl_mean) > 5.0:
    return {
        "converged": False,
        "iterations": len(rows),
        "cl": cl_mean,
        "cd": cd_mean,
        "cm": cm_mean,
        "reason": (
            f"Unphysical aerodynamic coefficients (Cl={cl_mean:.4f}, Cd={cd_mean:.4f})."
        ),
    }

  if cd_std > 0.25 * abs(cd_mean) or cl_std > 0.25 * max(abs(cl_mean), 0.1):
    return {
        "converged": False,
        "iterations": len(rows),
        "cl": cl_mean,
        "cd": cd_mean,
        "cm": cm_mean,
        "reason": (
            f"Force coefficients failed to settle (Cl std={cl_std:.4f}, Cd std={cd_std:.4f})."
        ),
    }

  return {
      "converged": True,
      "iterations": len(rows),
      "cl": cl_mean,
      "cd": cd_mean,
      "cm": cm_mean,
      "cl_std": cl_std,
      "cd_std": cd_std,
  }


def funct(x: np.ndarray, parameters: Parameters) -> float:
  """Evaluates a 6-parameter Kulfan CST airfoil using OpenFOAM-14 (`airFoil2D`).

  Thread-safe: multiple candidate shapes can be evaluated concurrently via
  `evaluate_batch(X_batch, parameters)` or `ThreadPoolExecutor`.

  Args:
    x: 1D numpy array of 6 Kulfan CST parameters:
      - x[0:3] = wl (lower surface Bernstein weights)
      - x[3:6] = wu (upper surface Bernstein weights)
    parameters: Execution configuration and telemetry tracker.

  Returns:
    Negative lift-to-drag ratio `-Cl/Cd` for minimization (or `0.0` on failure / budget exhaustion).
  """
  is_polar = parameters.run_name.startswith("polar_aoa_")
  with parameters._lock:
    if not is_polar:
      if (
          not parameters.allow_multiple_evals
          and parameters.sim_count >= 1
      ):
        return 0.0
      if (
          parameters.successful_sim_count >= parameters.max_successful_sims
          or parameters.sim_count >= parameters.max_total_sims
      ):
        return 0.0
    parameters.sim_count += 1
    sim_id = parameters.sim_count

  x_arr = np.asarray(x, dtype=float).flatten()
  if x_arr.shape != (6,) or not np.all(np.isfinite(x_arr)):
    with parameters._lock:
      parameters.last_diagnostics = {
          "stage": "parameter_validation",
          "no_clipping": False,
          "block_mesh": False,
          "check_mesh": False,
          "simple": False,
          "error_message": f"Expected finite 6-element CST vector, got {x_arr}.",
      }
      _log_to_csv(
          parameters.csv_path,
          parameters.run_name,
          False,
          False,
          False,
          False,
          1.0,
          0.0,
      )
    return 0.0

  wl = x_arr[0:3]
  wu = x_arr[3:6]

  # Step 1: Validate airfoil geometry via Kulfan CST coordinates & structural rules
  try:
    airfoil = CST_shape(wl=wl, wu=wu, dz=0.0, N=200)
    _ = airfoil.airfoil_coor()
    no_clipping, geom_metrics = check_geometry(x_arr)
    min_thickness = geom_metrics["min_thickness"]
    max_thickness = geom_metrics["max_thickness"]
    max_camber = geom_metrics["max_camber"]
  except Exception as e:
    with parameters._lock:
      parameters.last_diagnostics = {
          "stage": "geometry_validation",
          "no_clipping": False,
          "block_mesh": False,
          "check_mesh": False,
          "simple": False,
          "error_message": f"CST geometry generation failed: {e}",
      }
      _log_to_csv(
          parameters.csv_path,
          parameters.run_name,
          False,
          False,
          False,
          False,
          1.0,
          0.0,
      )
    return 0.0

  if not no_clipping:
    with parameters._lock:
      parameters.last_diagnostics = {
          "stage": "geometry_validation",
          "no_clipping": False,
          "block_mesh": False,
          "check_mesh": False,
          "simple": False,
          "min_thickness": min_thickness,
          "max_thickness": max_thickness,
          "max_camber": max_camber,
          "error_message": (
              f"Invalid airfoil thickness profile: min_thickness={min_thickness:.4f} "
              f"(must be >= {MIN_INTERIOR_THICKNESS}), max_thickness={max_thickness:.4f} "
              f"(must be in [{MIN_MAX_THICKNESS}, {MAX_MAX_THICKNESS}])."
          ),
      }
      _log_to_csv(
          parameters.csv_path,
          parameters.run_name,
          False,
          False,
          False,
          False,
          1.0,
          0.0,
      )
    return 0.0

  # Step 2: Prepare isolated OpenFOAM case directory for this simulation
  case_name = f"{parameters.run_name}_{sim_id}"
  case_dir = Path(parameters.cases_folder) / case_name
  if case_dir.exists():
    shutil.rmtree(case_dir)
  case_dir.mkdir(parents=True, exist_ok=True)

  try:
    _write_block_mesh_dict(case_dir, wl=wl, wu=wu)
    _write_openfoam_case_files(
        case_dir,
        fluid_velocity=parameters.fluid_velocity,
        end_time=parameters.end_time,
    )

    # Step 3: Run blockMesh
    bm_res = _run_openfoam_cmd(
        "blockMesh > log.blockMesh 2>&1", cwd=case_dir, timeout_sec=60
    )
    bm_log = (
        (case_dir / "log.blockMesh").read_text(
            encoding="utf-8", errors="replace"
        )
        if (case_dir / "log.blockMesh").exists()
        else (bm_res.stdout + bm_res.stderr)
    )
    block_mesh_ok = bm_res.returncode == 0 and "FOAM FATAL" not in bm_log
    if not block_mesh_ok:
      with parameters._lock:
        parameters.last_diagnostics = {
            "stage": "blockMesh",
            "no_clipping": True,
            "block_mesh": False,
            "check_mesh": False,
            "simple": False,
            "max_thickness": max_thickness,
            "max_camber": max_camber,
            "error_message": (
                f"blockMesh failed (rc={bm_res.returncode}): {bm_log[-600:]}"
            ),
        }
        _log_to_csv(
            parameters.csv_path,
            parameters.run_name,
            True,
            False,
            False,
            False,
            1.0,
            0.0,
        )
      return 0.0

    # Step 4: Run checkMesh
    cm_res = _run_openfoam_cmd(
        "checkMesh > log.checkMesh 2>&1", cwd=case_dir, timeout_sec=60
    )
    cm_log = (
        (case_dir / "log.checkMesh").read_text(
            encoding="utf-8", errors="replace"
        )
        if (case_dir / "log.checkMesh").exists()
        else (cm_res.stdout + cm_res.stderr)
    )
    check_mesh_ok = (
        cm_res.returncode == 0
        and "Failed" not in cm_log
        and "FOAM FATAL" not in cm_log
        and "Mesh OK" in cm_log
    )
    if not check_mesh_ok:
      with parameters._lock:
        parameters.last_diagnostics = {
            "stage": "checkMesh",
            "no_clipping": True,
            "block_mesh": True,
            "check_mesh": False,
            "simple": False,
            "max_thickness": max_thickness,
            "max_camber": max_camber,
            "error_message": f"checkMesh rejected mesh quality: {cm_log[-600:]}",
        }
        _log_to_csv(
            parameters.csv_path,
            parameters.run_name,
            True,
            True,
            False,
            False,
            1.0,
            0.0,
        )
      return 0.0

    # Step 5: Run OpenFOAM incompressible steady-state RANS solver
    solver_cmd = (
        "(command -v foamRun >/dev/null 2>&1 && foamRun || simpleFoam) > log.foamRun 2>&1"
    )
    sim_res = _run_openfoam_cmd(solver_cmd, cwd=case_dir, timeout_sec=240)
    sim_log = (
        (case_dir / "log.foamRun").read_text(encoding="utf-8", errors="replace")
        if (case_dir / "log.foamRun").exists()
        else (sim_res.stdout + sim_res.stderr)
    )

    if sim_res.returncode != 0 or "FOAM FATAL" in sim_log:
      with parameters._lock:
        parameters.last_diagnostics = {
            "stage": "foamRun",
            "no_clipping": True,
            "block_mesh": True,
            "check_mesh": True,
            "simple": False,
            "max_thickness": max_thickness,
            "max_camber": max_camber,
            "error_message": (
                f"OpenFOAM solver failed (rc={sim_res.returncode}): {sim_log[-600:]}"
            ),
        }
        _log_to_csv(
            parameters.csv_path,
            parameters.run_name,
            True,
            True,
            True,
            False,
            1.0,
            0.0,
        )
      return 0.0

    # Step 6: Parse converged aerodynamic coefficients from forceCoeffs1
    parsed = _parse_force_coeffs(case_dir)
    if not parsed.get("converged", False):
      with parameters._lock:
        parameters.last_diagnostics = {
            "stage": "convergence_check",
            "no_clipping": True,
            "block_mesh": True,
            "check_mesh": True,
            "simple": False,
            "max_thickness": max_thickness,
            "max_camber": max_camber,
            "iterations": parsed.get("iterations", 0),
            "error_message": parsed.get("reason", "Solver did not converge."),
        }
        _log_to_csv(
            parameters.csv_path,
            parameters.run_name,
            True,
            True,
            True,
            False,
            1.0,
            0.0,
        )
      return 0.0

    cl = float(parsed["cl"])
    cd = float(parsed["cd"])
    cm = float(parsed["cm"])
    ld_ratio = cl / cd

    score = -ld_ratio if ld_ratio > 0.0 else 0.0
    ux, uy = float(parameters.fluid_velocity[0]), float(
        parameters.fluid_velocity[1]
    )
    aoa_deg = math.degrees(math.atan2(uy, ux))

    diag = {
        "stage": "completed",
        "no_clipping": True,
        "block_mesh": True,
        "check_mesh": True,
        "simple": True,
        "cl": cl,
        "cd": cd,
        "cm": cm,
        "ld_ratio": ld_ratio,
        "aoa_deg": aoa_deg,
        "max_thickness": max_thickness,
        "max_camber": max_camber,
        "iterations": parsed.get("iterations", 0),
    }

    with parameters._lock:
      if not is_polar and score < 0.0:
        if parameters.successful_sim_count >= parameters.max_successful_sims:
          return 0.0
        parameters.successful_sim_count += 1
        succ_idx = parameters.successful_sim_count
      else:
        succ_idx = parameters.successful_sim_count

      parameters.last_diagnostics = diag
      if score < 0.0:
        parameters.eval_history.append({
            "sim_id": sim_id,
            "successful_idx": succ_idx,
            "x": x_arr.tolist(),
            "ld_ratio": ld_ratio,
            "cl": cl,
            "cd": cd,
            "cm": cm,
            "max_thickness": max_thickness,
            "max_camber": max_camber,
            "iterations": parsed.get("iterations", 0),
        })
        if score < parameters.best_score:
          parameters.best_score = float(score)
          parameters.best_x = x_arr.copy()
          parameters.best_diagnostics = dict(diag)

      _log_to_csv(
          parameters.csv_path,
          parameters.run_name,
          True,
          True,
          True,
          True,
          cd,
          cl,
      )
    return float(score)

  finally:
    if case_dir.exists():
      shutil.rmtree(case_dir, ignore_errors=True)


def evaluate_batch(
    candidates: Sequence[np.ndarray] | np.ndarray,
    parameters: Parameters,
    max_workers: int = 4,
) -> list[float]:
  """Evaluates a batch of 6-element Kulfan CST airfoil candidates in parallel using OpenFOAM-14.

  Args:
    candidates: Sequence or 2D array of shape (N, 6) containing Kulfan CST vectors.
    parameters: Shared Parameters instance tracking simulation budget and telemetry.
    max_workers: Number of parallel OpenFOAM subprocesses to run concurrently.

  Returns:
    List of scores (`-Cl/Cd` for valid airfoils, `0.0` for invalid or over-budget shapes).
  """
  cand_list = [np.asarray(c, dtype=float).flatten() for c in candidates]
  if not cand_list:
    return []
  workers = max(1, min(int(max_workers), len(cand_list), 8))
  with ThreadPoolExecutor(max_workers=workers) as executor:
    return list(executor.map(lambda x: funct(x, parameters), cand_list))
