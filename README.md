# FTS-xRooFit-Demo: Focused Test Statistics Implementation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![ROOT](https://img.shields.io/badge/ROOT-6.24%2B-orange)](https://root.cern)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)


- FTS uses Bayesian-style focus weights but the statistic is entirely frequentist.
- FTS does not inherit the chi-square asymptotics of the LRT; you must calibrate it explicitly.
- Confidence intervals must be obtained via a Neyman construction (RooStats toys or the pure-Python belts provided here), not via chi-square approximations.

## Overview

This repository implements **Focused Test Statistics (FTS)** as a statistical method for enhancing sensitivity in parameter estimation and hypothesis testing in particle physics analyses. FTS incorporates physics-motivated "focus functions" to concentrate statistical power in specific parameter regions of interest while remaining a **frequentist** construction—no Bayesian priors or Wilks-style asymptotics are assumed.

## Project Structure

### 📁 Core Implementation
```
src/
├── fts_core.py                 # Main FTS algorithm and optimization classes
├── publication_plotting.py     # Professional plotting utilities
├── asimov_utils.py            # Asimov dataset generation tools
├── config.py                  # Environment configuration
├── demo_replacement_guide.py  # Integration guide for existing analyses
└── setup_environment.py       # Automated environment setup
```

### 📓 Interactive Notebooks
```
notebooks/
├── FTS_plus.ipynb            # 🎯 MAIN DEMO: Complete FTS implementation
├── validation_plots.ipynb    # Validation analysis and comparison plots
└── fts_core.py               # Core algorithms (notebook copy)
```

### 📋 Examples & Scripts
```
examples/
├── simple_integration.py     # Minimal standalone FTS example
└── html_exact_compat.py      # xRooFit environment compatibility layer

scripts/
├── validate_optimization.py  # Performance validation and testing
├── neyman_fts_qr.py          # Quantile-regressed Neyman calibration
├── neyman_roostats.py        # RooStats Feldman–Cousins baseline (μ only)
└── interval_json_to_npz.py   # Convert Neyman JSON outputs to NPZ
```

### 📊 Output Directory
```
results/                       # Auto-generated analysis outputs
├── *.png                     # Generated plots and figures
├── *.json                    # Analysis results and validation reports
└── validation_summary.json   # Overall validation metrics
```

## File Execution Guide

### 🚀 Main Demonstrations

#### `notebooks/FTS_plus.ipynb` → Complete Analysis Suite
**Execution:** Jupyter notebook interface
```bash
jupyter notebook notebooks/FTS_plus.ipynb
```
**Generates:**
- Statistical model setup and validation
- FTS vs LRT comparison analysis
- Performance benchmarks and timing metrics
- Interactive plots with hypothesis testing results
- Comprehensive analysis summary

#### `notebooks/validation_plots.ipynb` → Validation Analysis
**Execution:** Jupyter notebook interface
**Generates:**
- `results/fts_validation_verified.png` - Three-panel validation plots
- `results/fts_validation_report.json` - Detailed validation metrics
- Constant offset verification (FTS = LRT + C)
- Focus function normalization checks
- Statistical property validation

### 🔧 Command-Line Tools

#### `examples/simple_integration.py` → Basic FTS Demo
**Execution:**
```bash
python examples/simple_integration.py
```
**Output:**
```
Simple FTS Integration Example
=====================================
Mock Statistical Model Setup Complete
Focus Function: Gaussian(μ=1.0, σ=0.5)
FTS Test Statistic: 2.847
Traditional LRT: 3.496
Enhancement Factor: 18.6% improvement in focus region
```

#### `scripts/validate_optimization.py` → Performance Testing
**Execution:**
```bash
python scripts/validate_optimization.py [--quick]
```
**Generates:**
- Performance benchmark results
- Cache efficiency metrics
- Numerical accuracy validation
- Error handling verification
- Console report with PASS/FAIL status

#### `scripts/neyman_fts_qr.py` → Quantile-Regressed Neyman Calibration
**Execution:**
```bash
python scripts/neyman_fts_qr.py \
    --observed 8 \
    --signal 5.0 \
    --background 3.0 \
    --mu-min 0.0 \
    --mu-max 5.0 \
    --n-toys 2000 \
    --output results/fts_neyman_qr.json
```
**Generates:**
- FTS and LRS critical curves across a μ grid
- Optional quantile-regressed smoothing for toys
- Coverage and interval-length diagnostics stored in JSON

#### `scripts/neyman_roostats.py` → RooStats Feldman–Cousins baseline
**Execution:**
```bash
python scripts/neyman_roostats.py \
    --observed 8 \
    --signal 5.0 \
    --background 3.0 \
    --mu-min 0.0 \
    --mu-max 5.0 \
    --nbins 60 \
    --confidence-level 0.95 \
    --output results/fc_interval_mu.json
```
**Generates:**
- JSON file containing the observed Feldman–Cousins interval and sampled belt points
- Baseline configuration for the no-nuisance counting model; re-enable systematics after verifying the core workflow
- Adapt the internal `build_workspace` helper to import your own `ModelConfig` once available

JSON schema：
```json
{
  "method": "LRS",
  "construction": "FeldmanCousins",
  "ordering": "likelihood_ratio",
  "confidence_level": 0.95,
  "poi": "mu",
  "grid": {
    "mu": [0.0, 0.1, ...],
    "lower": [...],
    "upper": [...]
  },
  "observed_interval": {"lower": 0.7, "upper": 2.1},
  "model": {
    "tag": "counts_no_nuisance_v1",
    "observed": 8.0,
    "signal": 5.0,
    "background": 3.0
  },
  "settings": {
    "mu_bounds": [0.0, 5.0],
    "nbins": 60,
    "seed": 12345
  }
}
```

#### `scripts/interval_json_to_npz.py` → Convert Neyman payloads to NPZ
**Execution:**
```bash
python scripts/interval_json_to_npz.py \
    --fts-json results/fts_neyman_qr.json \
    --lrs-json results/fc_interval_mu.json \
    --out results/fts_interval_metrics.npz
```
**Generates:**
- Consolidated NPZ arrays with coverage and interval metrics
- Ready-to-plot inputs for notebooks or downstream scripts

### 🔗 Environment Setup

#### `examples/html_exact_compat.py` → Auto-Configuration
**Usage:** Import in Python scripts/notebooks
```python
import sys
sys.path.insert(0, 'examples')
import html_exact_compat  # Automatically configures xRooFit environment
```
**Functions:**
- Detects ROOT and xRooFit installations
- Configures library paths (macOS/Linux)
- Sets up Python module paths
- Validates environment readiness

## Installation & Dependencies

### Prerequisites
- **Python 3.8+** with scientific computing stack
- **ROOT 6.24+** with Python bindings
- **xRooFit** (build from CERN GitLab)

### Quick Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install ROOT (macOS)
brew install root

# 3. Build xRooFit
git clone https://gitlab.cern.ch/will/xroofit.git
cmake -S xroofit -B xroofit_build
cmake --build xroofit_build

# 4. Verify installation
python examples/simple_integration.py
```

## Core Algorithm Features

### FTS Test Statistic
```
T_f(D; μ₀) = -2 log[L(μ₀|D) / ∫ L(μ|D) f(μ) dμ]
```

**Key Properties:**
- **Constant Offset**: FTS and LRT differ by a configuration-dependent constant (reported numerically per run)
- **Focus Enhancement**: Higher sensitivity in focus regions
- **Numerical Stability**: Log-sum-exp implementation
- **Optimized Caching**: Eliminates redundant calculations

### Focus Functions

#### Gaussian Focus (Recommended)
```python
from src.fts_core import ProductionFocusFunction

focus = ProductionFocusFunction(
    mu_focus=1.0,      # Center of interest
    sigma_focus=0.5,   # Width of focus region
    normalize=True     # Ensure ∫f(μ)dμ = 1
)
```

#### Top-Hat Focus
```python
focus = ProductionFocusFunction(
    mu_focus=1.0,
    sigma_focus=1.0,
    weight_type='tophat'
)
```

## Usage Examples

### Basic FTS Calculation
```python
from src.fts_core import fts_ts_obs, ProductionFocusFunction

# Setup focus function
focus = ProductionFocusFunction(mu_focus=1.0, sigma_focus=0.5)

# Calculate FTS test statistic
fts_value = fts_ts_obs(
    nll_calc=your_nll_calculator,
    dataset="observed_data",
    mu0=1.0,
    focus_obj=focus,
    n_grid=101
)
```

### Publication Plotting
```python
from src.publication_plotting import plot_fts_lrs_paper_style

fig, axes = plot_fts_lrs_paper_style(
    mu_grid, T_fts_obs, T_lrt_obs,
    C_fts_68, C_fts_95, C_lrt_68, C_lrt_95,
    savepath='results/my_analysis.png'
)
```

## Validation Metrics

The implementation includes comprehensive validation:

| Test | Expected Result | Interpretation |
|------|----------------|----------------|
| **Constant Offset** | Reported per configuration (numeric + Laplace) | FTS-LRT offset consistency |
| **Focus Normalization** | ∫f(μ)dμ = 1 ± 1e-6 | Proper probability weighting |
| **Cache Efficiency** | Hit rate > 50% | Performance optimization |
| **Error Handling** | Proper exceptions | Robust implementation |

## Mathematical Background

**Frequentist Method**: FTS is a frequentist test statistic. It uses Bayesian-style weights ("focus functions") as design choices inside the statistic, but it does not perform Bayesian inference and it does not rely on Bayesian priors for coverage. Coverage validation is carried out explicitly via a Neyman construction.

**FTS Theory**: Enhances statistical power by incorporating weights that concentrate sensitivity in parameter regions of interest (e.g., around a physics-motivated μ). The focus function is normalized (∫f(μ)dμ=1) and appears in the FTS denominator as an average over μ, computed with numerically stable quadrature.

**Relationship to LRT**: Unlike the LRT, FTS does not inherit standard LRT asymptotic properties. Confidence intervals for both LRS and FTS should therefore be calibrated via a Neyman construction rather than relying on Wilks approximations (see the FTS paper for details).

## Confidence Intervals (Neyman Construction)

To obtain properly calibrated confidence intervals for μ, use a Neyman construction. This is computationally heavier than looking at p-value histograms, but it provides the correct belt and coverage.

1. **Step 0 – Nuisances off:** For speed, first disable all nuisance parameters and calibrate only the parameter of interest μ. Once the pipeline is validated, re-enable systematics and repeat.
2. **Step 1 – RooStats (LRS baseline):** Run the Feldman–Cousins implementation in RooStats. A ready-to-run helper is provided in `scripts/neyman_roostats.py`, which builds a nuisance-free counting model, performs the Feldman–Cousins construction, and writes the resulting belt to `results/`. Adapt the script to your workspace by replacing the simple counting model with your own `ModelConfig`.
3. **Step 2 – FTS calibration:** If a direct RooStats `TestStatistic` implementation is not available for FTS, use the pure-Python calibrator in `src/neyman.py` to build the belt with toys, optionally accelerating the fit with quantile regression.

See the code snippet below for a lightweight, pure-Python workflow (μ only, no nuisances):

```python
from src.fts_core import OptimizedFTSCalculator
from src.neyman import calibrate_belts, default_quantile_levels

# Required user hooks (domain-specific):
#   - generate_toy(mu_true, i) -> dataset_id
#   - nll_calc.get_nll_at_mu(dataset_id, mu) -> NLL

calc = OptimizedFTSCalculator()
mu_grid = np.linspace(0.0, 3.0, 61)
alphas = (0.3173, 0.0455)  # 68% and 95% CL

crit = calibrate_belts(
    calc=calc,
    nll_calc=persistent_nll_calc,           # your existing NLL calculator
    focus=ProductionFocusFunction(1.0, 0.5),
    mu_grid=mu_grid,
    alphas=alphas,
    n_toys=2000,                            # increase as needed
    generate_toy=generate_toy,              # user-provided hook
    method="empirical",                    # or "quantile_regression" if sklearn available
    seed=1234,
)

# Returns dicts: crit['lrs'][alpha] -> array over mu_grid
#                crit['fts'][alpha] -> array over mu_grid
```

## Troubleshooting

### Common Issues

**ROOT/xRooFit not found:**
```bash
# Verify ROOT installation
root --version
python -c "import ROOT; print('ROOT OK')"

# Check xRooFit build
ls -la xroofit_build/libxRooFit.*
```

**Import errors:**
```bash
# Ensure correct working directory
cd FTS-xRooFit-Demo1
python examples/simple_integration.py
```

### Notes on Systematics and Nuisances

- To speed up early validation, freeze all nuisance parameters (no systematics) and calibrate only μ. Then re-enable systematics and repeat the calibration for the final results.
- If using a RooFit/xRooFit workspace, you can freeze all non-POI parameters (e.g., all but `mu`) by setting them `Constant(true)`. A convenience helper is provided in `src/neyman.py` (best-effort; adjust to your workspace conventions).

## References

- **FTS Paper**: [arXiv:2507.17831](https://arxiv.org/abs/2507.17831)
- **xRooFit**: [GitLab Repository](https://gitlab.cern.ch/will/xroofit)
- **ROOT**: [Official Documentation](https://root.cern/)

## License

MIT License - See [LICENSE](./LICENSE) for details.

---

**Quick Start**: Run `python examples/simple_integration.py` for a basic demonstration, then explore `notebooks/FTS_plus.ipynb` for the complete interactive experience.
