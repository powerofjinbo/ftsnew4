#!/usr/bin/env python3
"""
Neyman Construction Utilities (μ-only, no-nuisance starting point)
=================================================================

Lightweight, pure-Python calibration for confidence belts of LRS and FTS.

Design goals:
- Start with the parameter of interest μ and no nuisance parameters for speed.
- Provide a simple hook-based interface so existing likelihood evaluators can be used.
- Reuse the optimized FTS implementation to avoid redundant denominator work.
- Optionally speed up with quantile regression if scikit-learn is available.

Usage outline inside a notebook:

    from src.fts_core import OptimizedFTSCalculator
    from src.neyman import calibrate_belts

    calc = OptimizedFTSCalculator()
    mu_grid = np.linspace(0.0, 3.0, 61)

    crit = calibrate_belts(
        calc=calc,
        nll_calc=persistent_nll_calc,     # must implement get_nll_at_mu(dataset_id, mu)
        focus=ProductionFocusFunction(1.0, 0.5),
        mu_grid=mu_grid,
        alphas=(0.3173, 0.0455),
        n_toys=2000,
        generate_toy=generate_toy,        # user hook: generate_toy(mu_true, i) -> dataset_id
        method="empirical",               # or "quantile_regression" if sklearn available
        seed=1234,
    )

    # crit['lrs'][alpha], crit['fts'][alpha] are arrays over mu_grid

This module is deliberately conservative and does not assume xRooFit/RooStats
availability. A best-effort helper to freeze nuisances in a ROOT workspace is
provided for convenience when running with RooFit.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):
        return it


# Public defaults for common CL levels
default_quantile_levels: Tuple[float, float] = (0.3173, 0.0455)


def _empirical_quantile(a: np.ndarray, q: float) -> float:
    """Numerically stable empirical quantile for 0<q<1.

    Uses numpy.quantile with linear interpolation.
    """
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")
    try:
        return float(np.quantile(a, 1.0 - q, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(a, 1.0 - q, interpolation="linear"))


def _maybe_fit_qr(mu: np.ndarray, y: np.ndarray, alpha: float):
    """Fit a 1D quantile regressor y(mu) at quantile (1-alpha) if sklearn exists.

    Returns a callable f(mu_array) -> y_hat_array, or None if sklearn unavailable.
    """
    try:
        from sklearn.linear_model import QuantileRegressor
    except Exception:
        return None

    # Simple linear model in μ is often sufficient for smooth belts
    qr = QuantileRegressor(quantile=1.0 - float(alpha), alpha=0.0, solver="highs")
    X = mu.reshape(-1, 1)
    qr.fit(X, y)

    def predict_fn(mu_arr: np.ndarray) -> np.ndarray:
        return qr.predict(mu_arr.reshape(-1, 1))

    return predict_fn


def calibrate_belt(
    *,
    calc,
    nll_calc,
    focus,
    mu_grid: Sequence[float],
    alpha: float,
    n_toys: int,
    generate_toy: Callable[[float, int], str],
    method: str = "empirical",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrate critical curves C_alpha(mu) for both LRS and FTS at one alpha.

    Args:
        calc: OptimizedFTSCalculator instance (from src.fts_core)
        nll_calc: object with get_nll_at_mu(dataset_id, mu)
        focus: focus object with attributes mu_focus, sigma_focus, normalize and weight(mu)
        mu_grid: array of μ₀ values where to calibrate
        alpha: tail probability (e.g., 0.3173 for 68% CL)
        n_toys: number of pseudo-experiments per μ₀
        generate_toy: callback generating a toy dataset for a given μ_true and index -> dataset_id
        method: "empirical" (default) or "quantile_regression"
        seed: RNG seed for reproducibility (drives only local numpy RNG)

    Returns:
        (C_lrs_alpha, C_fts_alpha) each shaped like mu_grid
    """
    rng = np.random.default_rng(seed)
    mu_grid = np.asarray(mu_grid, dtype=float)
    C_lrs = np.empty_like(mu_grid)
    C_fts = np.empty_like(mu_grid)

    # Empirical calibration at each μ independently
    # For each μ₀: draw toys D~p(D|μ₀) and compute T(D; μ₀)
    for j, mu0 in enumerate(tqdm(mu_grid, desc=f"Neyman α={alpha:.4f}")):
        lrs_vals = np.empty(n_toys, dtype=float)
        fts_vals = np.empty(n_toys, dtype=float)

        for i in range(n_toys):
            # Generate toy dataset under the null μ₀
            dataset_id = generate_toy(float(mu0), i)

            # Compute both TS with one denominator evaluation (per dataset)
            fts_i, lrs_i = calc.compute_fts_and_lrt(
                nll_calc=nll_calc,
                dataset_id=dataset_id,
                mu0=float(mu0),
                focus_obj=focus,
                theta_bounds=(-100.0, 100.0),
                n_grid=401,
                verbose=False,
            )
            lrs_vals[i] = lrs_i
            fts_vals[i] = fts_i

        # Empirical tail critical values (upper-tail by default)
        C_lrs[j] = _empirical_quantile(lrs_vals, alpha)
        C_fts[j] = _empirical_quantile(fts_vals, alpha)

    # Optional quantile regression smoothing across μ (post-fit smoothing)
    if method == "quantile_regression":
        pred_lrs = _maybe_fit_qr(mu_grid, C_lrs, alpha)
        pred_fts = _maybe_fit_qr(mu_grid, C_fts, alpha)
        if pred_lrs is not None:
            C_lrs = pred_lrs(mu_grid)
        if pred_fts is not None:
            C_fts = pred_fts(mu_grid)

    return C_lrs, C_fts


def calibrate_belts(
    *,
    calc,
    nll_calc,
    focus,
    mu_grid: Sequence[float],
    alphas: Sequence[float] = default_quantile_levels,
    n_toys: int = 2000,
    generate_toy: Callable[[float, int], str],
    method: str = "empirical",
    seed: Optional[int] = None,
) -> Dict[str, Dict[float, np.ndarray]]:
    """Calibrate multiple α-level belts for LRS and FTS.

    Returns a nested dict: { 'lrs': {alpha: arr}, 'fts': {alpha: arr} }
    where each array has shape (len(mu_grid),).
    """
    out: Dict[str, Dict[float, np.ndarray]] = {"lrs": {}, "fts": {}}
    for alpha in alphas:
        C_lrs, C_fts = calibrate_belt(
            calc=calc,
            nll_calc=nll_calc,
            focus=focus,
            mu_grid=mu_grid,
            alpha=float(alpha),
            n_toys=int(n_toys),
            generate_toy=generate_toy,
            method=method,
            seed=None if seed is None else int(seed) + int(10_000 * alpha),
        )
        out["lrs"][float(alpha)] = C_lrs
        out["fts"][float(alpha)] = C_fts
    return out


# -----------------------------------------------------------------------------
# RooFit convenience: freeze all nuisance parameters (best-effort helper)
# -----------------------------------------------------------------------------

def freeze_all_nuisances(workspace, poi_name: str = "mu") -> int:
    """Set all non-POI parameters in a RooFit/xRooFit workspace to Constant(True).

    Returns the number of variables frozen. If ROOT is not available or the
    workspace does not expose expected APIs, this is a no-op and returns 0.
    """
    try:
        import ROOT  # noqa: F401
    except Exception:
        return 0

    frozen = 0
    try:
        # Try xRooFit-style dict of parameters
        if hasattr(workspace, "pars"):
            pars = workspace.pars()
            for name, var in pars.items():
                if name == poi_name:
                    continue
                try:
                    if hasattr(var, "isConstant") and hasattr(var, "setConstant"):
                        if not var.isConstant():
                            var.setConstant(True)
                            frozen += 1
                except Exception:
                    pass
            return frozen

        # Fallback: iterate over all RooRealVar in workspace
        if hasattr(workspace, "allVars"):
            it = workspace.allVars().createIterator()
            var = it.Next()
            while var:
                name = getattr(var, "GetName", lambda: "")( ) if hasattr(var, "GetName") else ""
                if name != poi_name and hasattr(var, "isConstant") and hasattr(var, "setConstant"):
                    try:
                        if not var.isConstant():
                            var.setConstant(True)
                            frozen += 1
                    except Exception:
                        pass
                var = it.Next()
            return frozen
    except Exception:
        return frozen

    return frozen


# -----------------------------------------------------------------------------
# Inversion utilities: from critical curves to CIs for one dataset
# -----------------------------------------------------------------------------

def _find_intervals_precise(mu_grid: Sequence[float],
                            ts_obs: Sequence[float],
                            crit: Sequence[float]) -> List[List[float]]:
    """Return acceptance intervals [mu_lo, mu_hi] via linear-interpolated boundaries.

    Defines acceptance by test inversion: accept μ if TS_obs(μ) ≤ C(μ).
    Supports multiple disjoint segments. No smoothing or clipping.
    """
    mu = np.asarray(mu_grid, float)
    obs = np.asarray(ts_obs, float)
    c = np.asarray(crit, float)

    d = obs - c
    accepted = d <= 0
    if not np.any(accepted):
        return []

    edges = np.where(accepted[:-1] != accepted[1:])[0]
    points: List[float] = []

    if accepted[0]:
        points.append(mu[0])
    for i in edges:
        d1, d2 = d[i], d[i + 1]
        if (d2 - d1) != 0:
            t = d1 / (d1 - d2)
            x = mu[i] + t * (mu[i + 1] - mu[i])
        else:
            x = mu[i]
        points.append(float(x))
    if accepted[-1]:
        points.append(mu[-1])

    points = sorted(points)
    out: List[List[float]] = []
    for i in range(0, len(points), 2):
        if i + 1 < len(points):
            out.append([points[i], points[i + 1]])
    return out


def compute_ts_curves(
    *,
    calc,
    nll_calc,
    dataset_id: str,
    focus,
    mu_grid: Sequence[float],
    theta_bounds: Tuple[float, float] = (-100.0, 100.0),
    n_grid: int = 401,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute observed T_lrs(μ) and T_fts(μ) arrays for one dataset.

    Uses a single denominator per dataset and reuses cached NLL grid.
    """
    mu_grid = np.asarray(mu_grid, float)
    T_lrs = np.empty_like(mu_grid)
    T_fts = np.empty_like(mu_grid)
    # Warm-up to ensure denominator cached
    calc.precompute_denominator(nll_calc, dataset_id, focus, theta_bounds, n_grid, verbose=False)
    for j, mu0 in enumerate(mu_grid):
        fts_j, lrs_j = calc.compute_fts_and_lrt(
            nll_calc=nll_calc,
            dataset_id=dataset_id,
            mu0=float(mu0),
            focus_obj=focus,
            theta_bounds=theta_bounds,
            n_grid=n_grid,
            verbose=False,
        )
        T_lrs[j] = lrs_j
        T_fts[j] = fts_j
    return T_lrs, T_fts


def invert_intervals_for_dataset(
    *,
    calc,
    nll_calc,
    dataset_id: str,
    focus,
    mu_grid: Sequence[float],
    crit_lrs: Dict[float, np.ndarray],
    crit_fts: Dict[float, np.ndarray],
    theta_bounds: Tuple[float, float] = (-100.0, 100.0),
    n_grid: int = 401,
) -> Dict[str, Dict[float, List[List[float]]]]:
    """Compute CIs via inversion for one dataset.

    Returns: { 'lrs': {alpha: [[lo,hi], ...]}, 'fts': {alpha: [[lo,hi], ...]} }
    """
    T_lrs, T_fts = compute_ts_curves(
        calc=calc,
        nll_calc=nll_calc,
        dataset_id=dataset_id,
        focus=focus,
        mu_grid=mu_grid,
        theta_bounds=theta_bounds,
        n_grid=n_grid,
    )

    out = {"lrs": {}, "fts": {}}
    for a, C in crit_lrs.items():
        out["lrs"][float(a)] = _find_intervals_precise(mu_grid, T_lrs, C)
    for a, C in crit_fts.items():
        out["fts"][float(a)] = _find_intervals_precise(mu_grid, T_fts, C)
    return out
