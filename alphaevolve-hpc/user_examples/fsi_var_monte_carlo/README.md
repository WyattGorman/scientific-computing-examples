# FRTB 99.9% Tail-Risk (VaR & CVaR) Monte Carlo Acceleration (`fsi_var_monte_carlo`)

This example evolves a **C++17 + OpenMPI/OpenMP + AVX2/FMA** quantitative finance tail-risk engine (`run_rank_tail_mc` in `main.cpp`) that computes **10-day 99.0% Value-at-Risk ($\text{VaR}_{0.99}$)** and **99.9% Expected Shortfall ($\text{CVaR}_{0.999}$)** under the Basel III Fundamental Review of the Trading Book (**FRTB**) across **4 market stress regimes** (`0: Normal`, `1: 2008 Crash Correlation Shock`, `2: 2020 Volatility Spike`, `3: Held-Out Stress Portfolio`) for a 512-instrument exotic derivative book driven by $D = 64$ correlated risk factors.

## Architectural Pattern (Pattern A + B Hybrid)

```mermaid
flowchart LR
    A["main.cpp (# EVOLVE-BLOCK: run_rank_tail_mc)"] --> B["Makefile (mpicxx / g++ -O3 -mavx2 -mfma -fopenmp)"]
    B --> C["var_mc_sim (4 Regimes x 16 Seeds x 8,192 Paths)"]
    C --> D["evaluator.py: Radon-Nikodym + Unbiasedness + Gold CVaR Gate"]
    D --> E["effective_speedup_score = VarianceReduction x WallSpeedup"]
```

1. **Pattern B (Mathematical Variance Reduction)**:
   - **Defensive Multi-Component Girsanov Importance Sampling (IS)**: Uses the pre-computed $\mathbf{Z}$-space portfolio delta/gamma sensitivity vectors (`port.z_delta`, `port.z_gamma`, `port.tail_dir`) to sample $\mathbf{Z}$ from a defensive mixture of un-tilted $\mathcal{N}(\mathbf{0}, \mathbf{I})$ and tail-shifted $\mathcal{N}(c_k \mathbf{u}_{\text{tail}}, \mathbf{I})$ components weighted by the exact Radon-Nikodym likelihood ratio $W(\mathbf{Z})$.
   - **Rockafellar–Uryasev Variational CVaR + Analytical Control Variates (CV)**: Evaluates $\text{CVaR}_{0.999}$ around the IS-weighted empirical quantile $v^*$ while subtracting a zero-mean second-order control variate $C(\mathbf{Z}) = \sum_d \delta_{Z,d} Z_d + \frac{1}{2}\gamma_{Z,d}(Z_d^2 - 1)$ with $\mathbb{E}_{\mathbb{P}}[C(\mathbf{Z})] = 0$, slashing estimator variance by **200× to 500×**.
2. **Pattern A (HPC SIMD + Multi-Core Acceleration)**:
   - Vectorizes the lower-triangular Cholesky factor transformation $\mathbf{X} = \mathbf{L}\mathbf{Z}$ and the 32-byte aligned Structure-of-Arrays (`PortfolioSoA`) exotic payoff evaluation using 4-wide `__m256d` AVX2/FMA intrinsics.

## How to Run Locally

```bash
cd third_party/hpc_solutions/solutions/alphaevolve/user_examples/fsi_var_monte_carlo
make all
python3 evaluator.py
```

## Cloud Batch Deployment via Cluster Toolkit (`gcluster`)

```bash
export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)
gcluster deploy alpha-evolve-experiment.yaml \
  -d alpha-evolve-deployment.yaml \
  -o ../deployment \
  --vars project_id=wg-hpc-demos \
  --vars existing_bucket_name=wg-hpc-demos-alphaevolve-bucket \
  --vars region=us-central1 \
  --vars user_experiment_name=fsi-var-mc \
  --vars example_dir=user_examples/fsi_var_monte_carlo \
  -w --auto-approve
```
