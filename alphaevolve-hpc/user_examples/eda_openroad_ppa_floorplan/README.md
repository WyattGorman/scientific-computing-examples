# Semiconductor Physical Design: OpenROAD 7nm ASAP7 Macro Floorplanning & PPA Optimization

This example demonstrates how to use AlphaEvolve to evolve a **dataflow-aware macro floorplanning and pin-orientation algorithm** (`compute_macro_floorplan` in `main.py`) for a 7nm ASAP7 RISC-V Vector + NPU Accelerator SoC tile (`asap7_riscv_npu_tile`).

## Problem Overview

In hierarchical ASIC physical design, hard SRAM macro placement (`(x_um, y_um, orientation, halo_um)`) dictates the achievable **Power, Performance, and Area (PPA)** of the entire block:
- Placing SRAM macros near the center of the die fragments the standard-cell core, blocks the central clock spine (`TritonCTS` from `IO_CLK_PLL`), and forces long wire detours across M2–M5 routing tracks.
- Placing SRAM macros near the die perimeter with the wrong orientation (`R0` on the East boundary) points active signal pins into the die edge (`PIN_BLOCKED_BY_BOUNDARY`), causing severe local GCell routing congestion (`FastRoute`).

## Two-Stage OpenROAD PPA Evaluation Cascade (`src/openroad_runner.py`)

1. **Stage 1 — Floorplan DRC Legality & Global Routing Congestion Gate**:
   - Verifies core boundary containment (`[20, 20, 1180, 1180] um`), zero macro-macro overlaps, minimum channel spacing (`>= 18.0 um`), zero sliver notches, and active pin-edge clearance (`>= 42.0 um`).
   - Generates OpenROAD `macro_placement.tcl` commands and solves the 2D `RePlAce` electrostatic density-spreading global placement for the 8 standard-cell functional clusters (45,000 gates) and 24x24 GCell `FastRoute` RUDY congestion map.
2. **Stage 2 — `TritonCTS` Clock Skew & 3-Corner `OpenSTA` Signoff**:
   - Evaluates Worst Negative Slack (`WNS`), Total Negative Slack (`TNS`), and clock-tree skew across all three 7nm ASAP7 PVT corners (`ss_0p63v_100c`, `tt_0p70v_25c`, `ff_0p77v_m40c`) at a `500 ps` (`2.0 GHz`) clock target, along with dynamic switching + repeater + CTS + leakage power (`mW`).

## Quickstart

```bash
# Run local baseline verification
python3 evaluator.py

# Launch evolutionary search
ae-run -c experiment.yaml
```
