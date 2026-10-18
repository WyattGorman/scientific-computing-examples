# 3D Liquid-Cooling Heat Sink Optimization with OpenFOAM-14

This example uses AlphaEvolve and OpenFOAM-14 Conjugate Heat Transfer (CHT) to design a 3D copper pin-fin liquid-cooling cold plate for an EDA semiconductor die with a non-uniform compute hotspot.

## Overview

Modern AI and HPC accelerators exhibit severe spatial power non-uniformity: dense compute core clusters generate intense localized heat fluxes while surrounding cache/IO periphery runs cooler. Uniform heat-sink pin arrays either under-cool the central hotspot or over-restrict coolant flow across the entire channel.

AlphaEvolve evolves a Python generator `design_heatsink(parameters)` that outputs an `(N, 5)` array of 3D elliptical copper pin-fins `[x, y, dx, dy, height]`. Each candidate design is converted into a watertight 3D STL surface (`fluidToSolid.stl`), meshed into coupled fluid and solid regions with `blockMesh`, `snappyHexMesh`, and `splitMeshRegions`, and solved to steady state with `foamMultiRun`.

## Physics & Domain Setup

| Property | Value |
| :--- | :--- |
| **OpenFOAM Version** | OpenFOAM-14 (`foamMultiRun` multi-region CHT) |
| **Channel Dimensions** | `24 mm` (L) $\times$ `16 mm` (W) $\times$ `4.5 mm` (H) water channel |
| **Copper Baseplate** | `1.5 mm` thick (`z in [-1.5, 0] mm`, $\kappa = 385\text{ W/m}\cdot\text{K}$) |
| **Central Compute Hotspot** | `8 mm` $\times$ `7 mm` (`x in [9, 17] mm`), $q = 2.0\text{ GW/m}^3$ ($\approx 101\text{ W}$) |
| **Surrounding Die Periphery** | `16 mm` $\times$ `13 mm` (`x in [4, 20] mm`), $q = 0.4\text{ GW/m}^3$ ($\approx 55\text{ W}$) |
| **Coolant Inlet** | Water at $U_{\text{in}} = 0.50\text{ m/s}$, $T_{\text{in}} = 300.0\text{ K}$ ($26.85\,^\circ\text{C}$) |
| **Pressure-Drop Budget** | $\Delta p \le 200.0\text{ Pa}$ |
| **Primary Metric** | `thermal_score = 400.0 - (T_max_K + dp_penalty_K)` (higher is better) |
| **Baseline (3x4 uniform pins)** | $T_{\max} = 374.89\text{ K}$ ($101.74\,^\circ\text{C}$), $\Delta p = 122.3\text{ Pa}$, `thermal_score = 25.11` |

## Directory Structure

```text
user_examples/heatsink_openfoam/
├── evaluator.Dockerfile      # Ubuntu 24.04 + OpenFOAM-14 + Python 3.12 container
├── evaluator.py              # AlphaEvolve evaluation & visualization hooks
├── experiment.yaml           # Declarative experiment configuration
├── main.py                   # Seed candidate generator (# EVOLVE-BLOCK)
├── README.md                 # Documentation
└── src/
    ├── geometry/
    │   └── stl_generator.py  # Watertight 3D elliptical pin-fin STL generator
    └── optimization/
        └── optimization.py   # OpenFOAM-14 multi-region CHT case writer & runner
```

## Design Space & Constraints

Each row `[x, y, dx, dy, height]` of the returned `(N, 5)` NumPy array (`4 <= N <= 20`) specifies one 3D copper pin-fin in meters:
- `x in [0.0030, 0.0210]`: Streamwise center along the 24 mm plate.
- `y in [-0.0065, 0.0065]`: Spanwise center across the 16 mm channel.
- `dx in [0.0008, 0.0030]`: Streamwise elliptical diameter.
- `dy in [0.0008, 0.0024]`: Spanwise elliptical diameter.
- `height in [0.0010, 0.0040]`: Pin height extending into the 4.5 mm water channel.
- **Clearance**: Every pair of pins must maintain at least `0.5 mm` (`0.0005 m`) surface-to-surface clearance so `snappyHexMesh` produces clean hex-dominant fluid passages.
- **Single Simulation Rule**: `evaluate(parameters)` runs strictly **1** OpenFOAM simulation per candidate evaluation.

## Running on Google Cloud

Deploy the experiment infrastructure using Cluster Toolkit (`gcluster`) from the `alphaevolve` root directory:

```bash
gcluster deploy alpha-evolve-experiment.yaml \
  --vars project_id=<YOUR_PROJECT_ID> \
  --vars region=us-central1 \
  --vars zone=us-central1-a \
  --vars deployment_name=ae-exp-heatsink-foam \
  --vars user_experiment_name=heatsink-foam \
  --vars example_dir=user_examples/heatsink_openfoam \
  --auto-approve
```

Once deployed, submit the validation check or launch the evolution loop:

```bash
bash scripts/submit_experiment.sh \
  --config=gs://<YOUR_PROJECT_ID>-alphaevolve-bucket/heatsink-foam/config/variables-heatsink-foam.env \
  --validate

bash scripts/submit_experiment.sh \
  --config=gs://<YOUR_PROJECT_ID>-alphaevolve-bucket/heatsink-foam/config/variables-heatsink-foam.env \
  --max_programs_generated=20 \
  --max_programs_evaluated=20 \
  --concurrency=4
```

## Outputs

After the controller finishes, post-processing uploads the best design summary and layout plot to Cloud Storage:
- `gs://<BUCKET>/heatsink-foam/results/best_heatsink_summary.json`
- `gs://<BUCKET>/heatsink-foam/results/optimized_heatsink.png`
